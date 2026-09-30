import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { transfer } from 'comlink';
import { useFrame, useThree } from '@react-three/fiber';
import { useComputedColorScheme } from '@mantine/core';
import CustomShaderMaterial from 'three-custom-shader-material/vanilla';
import type { Scene } from '@/domain/types';
import { simFactor } from '@/lib/usecases/floodSim';
import { analysisGridFor } from '@/features/usecases/actions';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { useUseCases } from '@/store/usecases';
import { SIM_GLSL, SIM_UNIFORMS_GLSL, simUniforms, useFloodSim } from '@/store/floodSim';
import { floodSimWorker } from '@/workers/clients';
import { sceneBase } from '../terrainState';

/** Simulated seconds per real second at most: the flood plays as a time-lapse (8 minutes a second). */
const SIM_SPEED = 480;
/** Worker compute per round trip, ms. */
const BUDGET_MS = 10;
/** Round trips the water must stay still before stepping pauses (until the level moves again). */
const CALM_FRAMES = 8;
const PUBLISH_MS = 200;

interface SimLayout {
  w: number;
  h: number;
  cellM: number;
  /** Scene pixels per simulation cell. */
  F: number;
}

/** Scene-grid pixel index of each simulation cell's centre along one axis, plus the grid's own edges, so the
 *  surface reaches the terrain border exactly. World = (pixel-corner position − size/2) · gsd, as the terrain. */
function axis(n: number, F: number, size: number, gsd: number) {
  const pts = [{ pos: 0.5, cell: 0 }];
  for (let k = 0; k < n; k++) {
    const p = (k + 0.5) * F;
    if (p > 0.5 && p < size - 0.5) pts.push({ pos: p, cell: k });
  }
  pts.push({ pos: size - 0.5, cell: Math.min(n - 1, Math.floor((size - 0.5) / F)) });
  return pts.map((p) => ({ world: (p.pos - size / 2) * gsd, cell: p.cell }));
}

function surfaceGeometry(scene: Scene, l: SimLayout) {
  const xs = axis(l.w, l.F, scene.heights.width, scene.gsd);
  const zs = axis(l.h, l.F, scene.heights.height, scene.gsd);
  const nx = xs.length;
  const pos = new Float32Array(nx * zs.length * 3);
  const cell = new Float32Array(nx * zs.length * 2);
  zs.forEach((z, r) =>
    xs.forEach((x, c) => {
      const v = r * nx + c;
      pos.set([x.world, 0, z.world], v * 3);
      cell.set([x.cell, z.cell], v * 2);
    }),
  );
  const index: number[] = [];
  for (let r = 0; r < zs.length - 1; r++) {
    for (let c = 0; c < nx - 1; c++) {
      const a = r * nx + c;
      index.push(a, a + nx, a + 1, a + 1, a + nx, a + nx + 1);
    }
  }
  // the real normal comes from the water surface in the vertex shader
  const up = new Float32Array(pos.length);
  for (let i = 1; i < up.length; i += 3) up[i] = 1;
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(up, 3));
  g.setAttribute('cell', new THREE.BufferAttribute(cell, 2));
  g.setIndex(index);
  return g;
}

const vertexShader = /* glsl */ `
  attribute vec2 cell;
  ${SIM_UNIFORMS_GLSL}
  uniform float uCellM;
  varying float vDepth;
  varying vec2 vVel;
  varying vec3 vPos;
  varying vec3 vNrm;
  ${SIM_GLSL}

  // a neighbour's surface, or this one where nothing is drawn (so the shore does not tilt the normal)
  float surfOr(ivec2 c, float own) {
    vec4 s = simTexel(c);
    return s.r >= 0.0 ? s.g : own;
  }

  void main() {
    ivec2 c = ivec2(cell);
    vec4 s = simTexel(c);
    float e = surfOr(c + ivec2(1, 0), s.g);
    float w = surfOr(c - ivec2(1, 0), s.g);
    float n = surfOr(c - ivec2(0, 1), s.g);
    float so = surfOr(c + ivec2(0, 1), s.g);
    vNrm = normalize(vec3((w - e) * uExag / (2.0 * uCellM), 1.0, (n - so) * uExag / (2.0 * uCellM)));
    csm_Position = vec3(position.x, s.g * uExag, position.z);
    csm_Normal = vNrm;
    vDepth = s.r;
    vVel = s.ba;
    vPos = csm_Position;
  }
`;

const fragmentShader = /* glsl */ `
  uniform float uTime;
  uniform vec3 uSky;
  varying float vDepth;
  varying vec2 vVel;
  varying vec3 vPos;
  varying vec3 vNrm;

  float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
  }
  float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), u.x), mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), u.x), u.y);
  }
  // slope of a travelling sine wave; faded out before it is finer than the pixels it covers
  vec2 wave(vec2 p, vec2 dir, float len, float amp, float px) {
    float k = 6.2831853 / len;
    float fade = 1.0 - smoothstep(0.12 * len, 0.5 * len, px);
    return dir * (amp * k * cos(dot(p, dir) * k - uTime * 1.25 * sqrt(k)) * fade);
  }
  vec2 ripples(vec2 p, float px) {
    return wave(p, vec2(0.8, 0.6), 1.7, 0.035, px) + wave(p, vec2(-0.45, 0.89), 3.1, 0.05, px) +
      wave(p, vec2(0.96, -0.28), 5.9, 0.07, px) + wave(p, vec2(-0.7, -0.71), 11.0, 0.1, px);
  }

  void main() {
    if (vDepth < 0.01) discard;
    vec2 p = vPos.xz;
    float px = length(fwidth(p));
    float speed = length(vVel);

    // ripples carried by the current: two phases cross-faded so the pattern never smears
    float T = 2.5;
    float t0 = fract(uTime / T);
    float t1 = fract(uTime / T + 0.5);
    float w0 = 1.0 - abs(2.0 * t0 - 1.0);
    vec2 flow = vVel * 1.6;
    vec2 g = ripples(p - flow * t0 * T, px) * w0 + ripples(p - flow * t1 * T + 37.0, px) * (1.0 - w0);
    g *= 0.5 + clamp(speed / 1.2, 0.0, 1.5);
    vec3 nW = normalize(vNrm + vec3(-g.x, 0.0, -g.y));
    csm_FragNormal = normalize((viewMatrix * vec4(nW, 0.0)).xyz);

    // silty flood water: light is absorbed with depth, so the ground fades out below it
    float k = 1.0 - exp(-vDepth * 0.55);
    vec3 col = mix(vec3(0.36, 0.5, 0.47), vec3(0.05, 0.16, 0.24), k);

    // foam on the thin advancing front and on fast water
    float fn = mix(vnoise(p * 0.45 - flow * uTime * 0.35), 0.5, smoothstep(0.6, 2.5, px));
    float front = 1.0 - smoothstep(0.03, 0.35, vDepth);
    float fast = smoothstep(0.9, 2.8, speed);
    float foam = clamp((0.85 * front + 0.75 * fast) * smoothstep(0.35, 0.7, fn), 0.0, 1.0);
    col = mix(col, vec3(0.88, 0.9, 0.88), foam);

    vec3 V = normalize(vViewPosition);
    float fres = 0.02 + 0.98 * pow(1.0 - clamp(dot(csm_FragNormal, V), 0.0, 1.0), 5.0);
    float alpha = max(mix(0.52, 0.9, k), foam);
    alpha = mix(alpha, 1.0, fres * 0.5) * smoothstep(0.01, 0.06, vDepth);
    csm_DiffuseColor = vec4(col, alpha);
    csm_Emissive = uSky * fres * (1.0 - foam) * 0.55;
    csm_Roughness = mix(0.05, 0.75, foam);
  }
`;

/** The flood as water: steps the hydraulics (floodSim.worker) while the flood scenario is on screen, feeds its
 *  state to every water-drawing material (simUniforms), and in 3D draws the water surface at its real height,
 *  so anything lower than the water is under it. */
export function FloodWater() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const open = useUseCases((s) => s.open);
  const active = useUseCases((s) => s.active);
  const on = useUseCases((s) => s.floodOverlay);
  const flood = useUseCases((s) => s.flood);
  const epoch = useUseCases((s) => s.floodEpoch);
  const shadows = useView((s) => s.shadows);
  const scheme = useComputedColorScheme('light');
  const invalidate = useThree((s) => s.invalidate);
  const [layout, setLayout] = useState<SimLayout | null>(null);
  const enabled = !!scene && open && active === 'flood' && on && !!flood;
  const is3d = mode === 'dsm3d';

  // the simulation loop: one worker round trip per frame, paused while the water is still and the level is not moving
  useEffect(() => {
    if (!enabled || !scene || !flood) return;
    let alive = true;
    let wake: (() => void) | null = null;
    const { grid } = analysisGridFor(scene);
    const s = simFactor(flood.w, flood.h);
    const l: SimLayout = { w: Math.ceil(flood.w / s), h: Math.ceil(flood.h / s), cellM: flood.cellM * s, F: grid.f * s };
    const { width: W, height: H } = scene.heights;
    const buildings = scene.objects?.buildings ?? [];
    const probes = Int32Array.from(buildings, (b) => {
      let cx = 0;
      let cy = 0;
      for (const [x, y] of b.poly) {
        cx += x / b.poly.length;
        cy += y / b.poly.length;
      }
      const sx = Math.min(l.w - 1, Math.max(0, Math.floor(cx / l.F)));
      const sy = Math.min(l.h - 1, Math.max(0, Math.floor(cy / l.F)));
      return sy * l.w + sx;
    });
    const tex = new THREE.DataTexture(new Float32Array(l.w * l.h * 4), l.w, l.h, THREE.RGBAFormat, THREE.FloatType);
    tex.magFilter = THREE.NearestFilter;
    tex.minFilter = THREE.NearestFilter;
    tex.flipY = false;
    tex.needsUpdate = true;
    const cellArea = l.cellM * l.cellM;
    const worker = floodSimWorker();
    const level = () => flood.baseLevel + useUseCases.getState().rise;

    void (async () => {
      await worker.init({ w: flood.w, h: flood.h, cellM: flood.cellM, ground: flood.ground, seedMask: flood.seedMask, s, level: level(), base: sceneBase(scene), absolute: scene.product === 'DSM', probes });
      if (!alive) return;
      simUniforms.uSim.value = tex;
      simUniforms.uSimSize.value.set(l.w, l.h);
      const k = 1 / (scene.gsd * l.F);
      simUniforms.uSimWorld.value.set(k, k, W / (2 * l.F) - 0.5, H / (2 * l.F) - 0.5);
      simUniforms.uSimUv.value.set(W / l.F, H / l.F, -0.5, -0.5);
      setLayout(l);
      let spare: Float32Array | null = null;
      let last = performance.now();
      let published = 0;
      let calm = 0;
      while (alive) {
        const now = performance.now();
        const realDt = Math.min(0.1, Math.max(1 / 60, (now - last) / 1000));
        last = now;
        const fr = await worker.advance(level(), BUDGET_MS, realDt * SIM_SPEED, spare ? transfer(spare, [spare.buffer]) : null);
        if (!alive) return;
        spare = tex.image.data as Float32Array;
        tex.image.data = fr.data;
        tex.needsUpdate = true;
        simUniforms.uHasSim.value = 1;
        invalidate();
        calm = fr.settled ? calm + 1 : 0;
        if (now - published > PUBLISH_MS || calm === CALM_FRAMES) {
          published = now;
          const wet = fr.wetCells;
          useFloodSim.getState().set({
            live: { share: fr.validCells ? wet / fr.validCells : 0, areaM2: wet * cellArea, meanDepthM: wet ? fr.sumDepthM / wet : 0, maxDepthM: fr.maxDepthM, stages: fr.stages, timeS: fr.timeS },
          });
        }
        if (calm >= CALM_FRAMES) {
          const rise = useUseCases.getState().rise;
          await new Promise<void>((resolve) => {
            const stop = useUseCases.subscribe((st) => st.rise !== rise && done());
            const done = () => {
              stop();
              wake = null;
              resolve();
            };
            wake = done;
          });
          calm = 0;
          last = performance.now();
        } else {
          await new Promise((r) => requestAnimationFrame(r));
        }
      }
    })().catch((e) => console.error('Flood simulation failed', e));

    return () => {
      alive = false;
      wake?.();
      simUniforms.uSim.value = null;
      simUniforms.uHasSim.value = 0;
      tex.dispose();
      useFloodSim.getState().set({ live: null });
      setLayout(null);
      invalidate();
    };
  }, [enabled, scene, flood, epoch, invalidate]);

  // the terrain drape defers to the surface only while it is drawn
  useEffect(() => {
    simUniforms.uWaterMesh.value = enabled && is3d && layout ? 1 : 0;
    invalidate();
  }, [enabled, is3d, layout, invalidate]);

  useEffect(() => {
    const apply = () => {
      simUniforms.uExag.value = useView.getState().exaggeration;
      invalidate();
    };
    apply();
    return useView.subscribe(apply);
  }, [invalidate]);

  const geometry = useMemo(() => (scene && layout ? surfaceGeometry(scene, layout) : null), [scene, layout]);
  useEffect(() => () => geometry?.dispose(), [geometry]);

  const { material, own } = useMemo(() => {
    const own = { uCellM: { value: 1 }, uTime: { value: 0 }, uSky: { value: new THREE.Color() } };
    const material = new CustomShaderMaterial({
      baseMaterial: THREE.MeshStandardMaterial,
      vertexShader,
      fragmentShader,
      uniforms: { ...simUniforms, ...own },
      transparent: true,
      depthWrite: false,
      roughness: 0.05,
      metalness: 0,
      side: THREE.DoubleSide,
    }) as unknown as THREE.MeshStandardMaterial;
    return { material, own };
  }, []);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => {
    own.uSky.value.set(scheme === 'dark' ? '#27344a' : '#bcd3ea');
    if (layout) own.uCellM.value = layout.cellM;
  }, [own, scheme, layout]);

  useFrame((_, dt) => {
    own.uTime.value += dt;
  });

  if (!enabled || !is3d || !geometry) return null;
  return <mesh name="flood-water" geometry={geometry} material={material} frustumCulled={false} renderOrder={2} receiveShadow={shadows} />;
}
