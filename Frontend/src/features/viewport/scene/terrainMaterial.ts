import * as THREE from 'three';
import CustomShaderMaterial from 'three-custom-shader-material/vanilla';
import { lut, type ColormapId } from '@/theme/colormaps';
import type { DrapeLayer } from '@/store/view';

/** Row order of the LUT atlas texture. */
export const LUT_ROWS: ColormapId[] = ['turbo', 'terrain', 'viridis', 'cividis', 'grey', 'diverging', 'slope', 'confidence'];
export const LAYER_INDEX: Record<DrapeLayer, number> = { tint: 0, optical: 1, height: 2, hillshade: 3, slope: 4, reference: 5, error: 6, classes: 7 };

export function createLutAtlas(): THREE.DataTexture {
  const data = new Uint8Array(256 * LUT_ROWS.length * 4);
  LUT_ROWS.forEach((id, row) => data.set(lut(id), row * 256 * 4));
  const tex = new THREE.DataTexture(data, 256, LUT_ROWS.length, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}

/** Heights (relative to `base`) as a filterable half-float texture, row 0 first (flipY = false). */
export function createHeightTexture(data: Float32Array, width: number, height: number, base: number): THREE.DataTexture {
  const half = new Uint16Array(data.length);
  for (let i = 0; i < data.length; i++) {
    const v = data[i];
    half[i] = THREE.DataUtils.toHalfFloat(Number.isFinite(v) ? v - base : 0);
  }
  const tex = new THREE.DataTexture(half, width, height, THREE.RedFormat, THREE.HalfFloatType);
  tex.magFilter = THREE.LinearFilter;
  tex.minFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/** Class ids as an R8 texture; nearest filtering so neighbouring ids never blend into a third. */
export function createClassTexture(data: Uint8Array, width: number, height: number): THREE.DataTexture {
  const tex = new THREE.DataTexture(data, width, height, THREE.RedFormat, THREE.UnsignedByteType);
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.flipY = false;
  tex.needsUpdate = true;
  return tex;
}

/** 256×1 RGBA palette indexed by class id. */
export function createClassPalette(rgba: Uint8Array): THREE.DataTexture {
  const tex = new THREE.DataTexture(rgba, 256, 1, THREE.RGBAFormat, THREE.UnsignedByteType);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.NearestFilter;
  tex.minFilter = THREE.NearestFilter;
  tex.needsUpdate = true;
  return tex;
}

export interface TerrainUniforms {
  [k: string]: THREE.IUniform;
  uHeight: THREE.IUniform<THREE.Texture | null>;
  uRgb: THREE.IUniform<THREE.Texture | null>;
  uLut: THREE.IUniform<THREE.Texture | null>;
  uRef: THREE.IUniform<THREE.Texture | null>;
  uHasRef: THREE.IUniform<number>;
  uClass: THREE.IUniform<THREE.Texture | null>;
  uClassLut: THREE.IUniform<THREE.Texture | null>;
  uHasClass: THREE.IUniform<number>;
  uLayer: THREE.IUniform<number>;
  uRange: THREE.IUniform<THREE.Vector2>;
  uCmapRow: THREE.IUniform<number>;
  uTint: THREE.IUniform<number>;
  uHillshade: THREE.IUniform<number>;
  uTexel: THREE.IUniform<THREE.Vector2>;
  uGsd: THREE.IUniform<number>;
  uSunDir: THREE.IUniform<THREE.Vector3>;
  uSlopeMax: THREE.IUniform<number>;
  uContour: THREE.IUniform<number>;
  uBase: THREE.IUniform<number>;
  uErrRange: THREE.IUniform<number>;
  uCompare: THREE.IUniform<number>;
  uSwipe: THREE.IUniform<number>;
  uViewport: THREE.IUniform<THREE.Vector2>;
}

export function createUniforms(): TerrainUniforms {
  return {
    uHeight: { value: null },
    uRgb: { value: null },
    uLut: { value: null },
    uRef: { value: null },
    uHasRef: { value: 0 },
    uClass: { value: null },
    uClassLut: { value: null },
    uHasClass: { value: 0 },
    uLayer: { value: 0 },
    uRange: { value: new THREE.Vector2(0, 1) },
    uCmapRow: { value: 1 },
    uTint: { value: 0.55 },
    uHillshade: { value: 0.45 },
    uTexel: { value: new THREE.Vector2(1, 1) },
    uGsd: { value: 0.5 },
    uSunDir: { value: new THREE.Vector3(0, 1, 0) },
    uSlopeMax: { value: 60 },
    uContour: { value: 0 },
    uBase: { value: 0 },
    uErrRange: { value: 5 },
    uCompare: { value: 0 },
    uSwipe: { value: 0.5 },
    uViewport: { value: new THREE.Vector2(1, 1) },
  };
}

const vertexShader = /* glsl */ `
  attribute float shade;
  varying vec2 vGrid;
  varying float vShade;
  void main() {
    vGrid = uv;
    vShade = shade;
  }
`;

const fragmentShader = /* glsl */ `
  uniform sampler2D uHeight;
  uniform sampler2D uRgb;
  uniform sampler2D uLut;
  uniform sampler2D uRef;
  uniform float uHasRef;
  uniform sampler2D uClass;
  uniform sampler2D uClassLut;
  uniform float uHasClass;
  uniform int uLayer;
  uniform vec2 uRange;
  uniform float uCmapRow;
  uniform float uTint;
  uniform float uHillshade;
  uniform vec2 uTexel;
  uniform float uGsd;
  uniform vec3 uSunDir;
  uniform float uSlopeMax;
  uniform float uContour;
  uniform float uBase;
  uniform float uErrRange;
  uniform float uCompare;
  uniform float uSwipe;
  uniform vec2 uViewport;
  varying vec2 vGrid;
  varying float vShade;

  const float LUT_ROWS = 8.0;
  vec3 cmap(float row, float t) {
    return texture2D(uLut, vec2(clamp(t, 0.0, 1.0) * (255.0 / 256.0) + 0.5 / 256.0, (row + 0.5) / LUT_ROWS)).rgb;
  }
  float hAt(vec2 g) { return texture2D(uHeight, g).r; }

  vec2 gradient(vec2 g) {
    float hE = hAt(g + vec2(uTexel.x, 0.0));
    float hW = hAt(g - vec2(uTexel.x, 0.0));
    float hS = hAt(g + vec2(0.0, uTexel.y));
    float hN = hAt(g - vec2(0.0, uTexel.y));
    return vec2((hE - hW) / (2.0 * uGsd), (hS - hN) / (2.0 * uGsd)); // (east, south)
  }

  float hillshade(vec2 grad) {
    vec3 n = normalize(vec3(-grad.x, 1.0, -grad.y));
    return clamp(dot(n, normalize(uSunDir)), 0.0, 1.0);
  }

  vec3 layerColor(int layer, vec2 g) {
    float h = hAt(g);
    float t = (h - uRange.x) / max(uRange.y - uRange.x, 1e-4);
    vec2 grad = gradient(g);
    float hs = hillshade(grad);
    float relief = mix(1.0, 0.35 + 0.9 * hs, uHillshade);
    vec3 rgb = texture2D(uRgb, g).rgb;
    if (layer == 1) return rgb;
    if (layer == 2) return cmap(uCmapRow, t) * relief;
    if (layer == 3) return vec3(hs);
    if (layer == 4) {
      float slope = degrees(atan(length(grad)));
      return cmap(6.0, slope / uSlopeMax) * mix(1.0, 0.55 + 0.6 * hs, uHillshade * 0.6);
    }
    if (layer == 5) {
      float r = texture2D(uRef, g).r;
      float tr = (r - uRange.x) / max(uRange.y - uRange.x, 1e-4);
      return uHasRef > 0.5 ? cmap(uCmapRow, tr) : vec3(0.5);
    }
    if (layer == 6) {
      float e = h - texture2D(uRef, g).r;
      return uHasRef > 0.5 ? cmap(5.0, 0.5 + 0.5 * e / uErrRange) : vec3(0.5);
    }
    if (layer == 7) {
      if (uHasClass < 0.5) return rgb;
      float id = floor(texture2D(uClass, g).r * 255.0 + 0.5);
      vec3 c = texture2D(uClassLut, vec2((id + 0.5) / 256.0, 0.5)).rgb;
      // keep a little of the photo underneath so misclassifications are visible against it
      return mix(rgb, c, 0.7) * relief;
    }
    // 0: optical + height tint — ground stays photographic, raised structures take the colormap.
    float a = uTint * smoothstep(0.02, 0.35, t);
    return mix(rgb, cmap(uCmapRow, t), a) * mix(1.0, relief, 0.6);
  }

  void main() {
    int layer = uLayer;
    if (uCompare > 0.5 && uHasRef > 0.5 && gl_FragCoord.x > uSwipe * uViewport.x) {
      layer = (layer == 6 || layer == 5) ? 2 : 5;
    }
    vec3 color = layerColor(layer, vGrid);
    if (uContour > 0.0) {
      float f = (hAt(vGrid) + uBase) / uContour;
      float d = abs(fract(f - 0.5) - 0.5) / max(fwidth(f), 1e-5);
      float line = 1.0 - clamp(d - 0.5, 0.0, 1.0);
      color = mix(color, vec3(0.08, 0.09, 0.11), line * 0.7);
    }
    csm_DiffuseColor = vec4(color * vShade, 1.0);
  }
`;

export type TerrainMaterialKind = 'lit' | 'flat';

/** Terrain drape material. `lit` = PBR (3D view, casts/receives shadows); `flat` = unlit (2D map views). */
export function createTerrainMaterial(uniforms: TerrainUniforms, kind: TerrainMaterialKind) {
  const common = { vertexShader, fragmentShader, uniforms };
  if (kind === 'lit') {
    return new CustomShaderMaterial({
      baseMaterial: THREE.MeshStandardMaterial,
      ...common,
      roughness: 0.93,
      metalness: 0,
      side: THREE.FrontSide,
    }) as unknown as THREE.MeshStandardMaterial & { uniforms: TerrainUniforms };
  }
  return new CustomShaderMaterial({
    baseMaterial: THREE.MeshBasicMaterial,
    ...common,
    side: THREE.DoubleSide,
  }) as unknown as THREE.MeshBasicMaterial & { uniforms: TerrainUniforms };
}

/** Direction *towards* the sun in scene coordinates (x east, y up, z south). Azimuth clockwise from north. */
export function sunDirection(azimuthDeg: number, elevationDeg: number, out = new THREE.Vector3()) {
  const az = THREE.MathUtils.degToRad(azimuthDeg);
  const el = THREE.MathUtils.degToRad(elevationDeg);
  return out.set(Math.sin(az) * Math.cos(el), Math.sin(el), -Math.cos(az) * Math.cos(el)).normalize();
}
