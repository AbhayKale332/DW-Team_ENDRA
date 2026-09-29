import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { notifications } from '@mantine/notifications';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { useSettings } from '@/store/settings';
import { lonLatAt } from '@/lib/georef';
import { CONTEXT_MARGIN, lonLatToGrid, sceneBBox } from '@/lib/osm';
import { BASEMAPS, planTiles, tileToLonLat, tileZoomFor, type TileId } from '@/lib/tiles';
import { groundLevel, verticalScale } from '@/features/viewport/terrainState';
import { useBasemapShown } from './basemapState';

/** Each tile is bent through the scene's CRS on this many cells per side (Web-Mercator tiles are not
 *  axis-aligned in a UTM grid), so neighbouring tiles and the terrain edge meet without gaps. */
const SEGMENTS = 8;
/** The coarse ring sits under the sharp one where they overlap (metres, before exaggeration). */
const OUTER_DROP_M = 0.3;

type ToGrid = (lon: number, lat: number) => [number, number] | null;

/** One tile as a grid of world-space vertices at height `y`; null if a corner cannot be reprojected. */
function tileGeometry(t: TileId, toGrid: ToGrid, W: number, H: number, gsd: number, y: number): THREE.BufferGeometry | null {
  const n = SEGMENTS + 1;
  const pos = new Float32Array(n * n * 3);
  const uv = new Float32Array(n * n * 2);
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const [lon, lat] = tileToLonLat(t.x + i / SEGMENTS, t.y + j / SEGMENTS, t.z);
      const g = toGrid(lon, lat);
      if (!g) return null;
      const k = j * n + i;
      pos[k * 3] = (g[0] - (W - 1) / 2) * gsd;
      pos[k * 3 + 1] = y;
      pos[k * 3 + 2] = (g[1] - (H - 1) / 2) * gsd;
      // tile row 0 is north; textures load with flipY, so v = 1 is the image's top
      uv[k * 2] = i / SEGMENTS;
      uv[k * 2 + 1] = 1 - j / SEGMENTS;
    }
  }
  const index: number[] = [];
  for (let j = 0; j < SEGMENTS; j++) {
    for (let i = 0; i < SEGMENTS; i++) {
      const a = j * n + i;
      index.push(a, a + n, a + 1, a + 1, a + n, a + n + 1);
    }
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
  g.setIndex(index);
  g.computeBoundingSphere();
  return g;
}

/** Unlit tile material with a hole over the processed grid: the terrain shows there, even where it dips
 *  below the surrounding ground level. */
function tileMaterial(map: THREE.Texture, ring: TileId['ring'], hole: { value: THREE.Vector2 }) {
  const m = new THREE.MeshBasicMaterial({
    map,
    toneMapped: false,
    side: THREE.DoubleSide,
    polygonOffset: true,
    polygonOffsetFactor: ring === 'inner' ? 1 : 3,
    polygonOffsetUnits: ring === 'inner' ? 2 : 8,
  });
  m.onBeforeCompile = (shader) => {
    shader.uniforms.uHole = hole;
    shader.vertexShader = `varying vec2 vDwXZ;\n${shader.vertexShader}`.replace(
      '#include <begin_vertex>',
      '#include <begin_vertex>\n  vDwXZ = (modelMatrix * vec4(transformed, 1.0)).xz;',
    );
    shader.fragmentShader = `uniform vec2 uHole;\nvarying vec2 vDwXZ;\n${shader.fragmentShader}`.replace(
      'void main() {',
      'void main() {\n  if (abs(vDwXZ.x) < uHole.x && abs(vDwXZ.y) < uHole.y) discard;',
    );
  };
  return m;
}

/** Map tiles around a georeferenced scene (CONTEXT_MARGIN scene sizes beyond every edge), shown for context
 *  only — they are never processed. Sharp tiles near the scene, coarser ones further out. */
export function BasemapLayer() {
  const scene = useScene((s) => s.scene);
  const shown = useBasemapShown();
  const provider = useSettings((s) => s.basemapProvider);
  const gl = useThree((s) => s.gl);
  const invalidate = useThree((s) => s.invalidate);
  const group = useRef<THREE.Group>(null);

  useEffect(() => {
    const root = group.current;
    const g = scene?.georef;
    if (!shown || !scene || !g || !root) return;
    const { width: W, height: H } = scene.heights;
    const gsd = scene.gsd;
    const toGrid = lonLatToGrid(g);
    const inner = sceneBBox(g, W, H, 1);
    const outer = sceneBBox(g, W, H, CONTEXT_MARGIN);
    const centre = lonLatAt(g, (W - 1) / 2, (H - 1) / 2);
    if (!toGrid || !inner || !outer || !centre) return;

    const p = BASEMAPS[provider];
    const tiles = planTiles(inner, outer, tileZoomFor(gsd, centre[1], p.maxZoom));
    // sharp ring first, nearest tiles first within each ring
    const dist = (t: TileId) => {
      const [lon, lat] = tileToLonLat(t.x + 0.5, t.y + 0.5, t.z);
      return (t.ring === 'inner' ? 0 : 1e6) + Math.hypot(lon - centre[0], lat - centre[1]);
    };
    tiles.sort((a, b) => dist(a) - dist(b));

    const hole = { value: new THREE.Vector2(Math.max(0, ((W - 1) / 2 - 0.25) * gsd), Math.max(0, ((H - 1) / 2 - 0.25) * gsd)) };
    const ground = groundLevel(scene);
    const anisotropy = gl.capabilities.getMaxAnisotropy();
    const loader = new THREE.TextureLoader();
    loader.setCrossOrigin('anonymous');
    const meshes: THREE.Mesh<THREE.BufferGeometry, THREE.MeshBasicMaterial>[] = [];
    let cancelled = false;
    let failed = 0;
    const onFail = () => {
      if (cancelled || ++failed !== tiles.length) return;
      notifications.show({ title: 'Basemap unavailable', message: `No ${p.label.toLowerCase()} tiles could be loaded (offline, or blocked by the tile server).`, color: 'red' });
    };

    for (const t of tiles) {
      const geom = tileGeometry(t, toGrid, W, H, gsd, t.ring === 'inner' ? ground : ground - OUTER_DROP_M);
      if (!geom) {
        onFail();
        continue;
      }
      loader.load(
        p.url(t.z, t.x, t.y),
        (tex) => {
          if (cancelled) {
            tex.dispose();
            geom.dispose();
            return;
          }
          tex.colorSpace = THREE.SRGBColorSpace;
          tex.anisotropy = anisotropy;
          const mesh = new THREE.Mesh(geom, tileMaterial(tex, t.ring, hole));
          mesh.name = `basemap-${t.z}-${t.x}-${t.y}`;
          mesh.renderOrder = t.ring === 'inner' ? -2 : -1;
          root.add(mesh);
          meshes.push(mesh);
          invalidate();
        },
        undefined,
        () => {
          geom.dispose();
          onFail();
        },
      );
    }

    return () => {
      cancelled = true;
      for (const m of meshes) {
        root.remove(m);
        m.geometry.dispose();
        m.material.map?.dispose();
        m.material.dispose();
      }
      invalidate();
    };
  }, [shown, scene, provider, gl, invalidate]);

  // exaggeration in 3D, flattened with the terrain in the 2D views
  useEffect(() => {
    const apply = () => {
      if (group.current) group.current.scale.y = verticalScale();
      invalidate();
    };
    apply();
    return useView.subscribe(apply);
  }, [invalidate]);

  return <group ref={group} name="basemap" visible={shown} />;
}
