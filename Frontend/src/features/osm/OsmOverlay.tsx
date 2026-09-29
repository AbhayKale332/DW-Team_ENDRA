import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { useScene } from '@/store/scene';
import { activeObjectKinds, objectKindsKey, useView } from '@/store/view';
import { sampleBilinear } from '@/lib/heights';
import { canGeolocate, OSM_KIND_COLORS, osmSegments, type OsmKind } from '@/lib/osm';
import { objectSurfaceIfReady } from '@/features/viewport/objectSurfaceClient';
import { sceneBase, verticalScale } from '@/features/viewport/terrainState';
import { loadOsm, osmFeaturesFor, useOsm } from './osmStore';

/** Drawn back to front, so roads cross over water and building outlines sit on top. */
const ORDER: OsmKind[] = ['water', 'waterway', 'path', 'road', 'rail', 'building'];
/** Lines float this far above the meshed surface (before exaggeration), clear of z-fighting. */
const LIFT_M = 0.4;

/** OpenStreetMap buildings, roads, railways and water as coloured lines draped on the terrain.
 *  Georeferenced scenes only; data is fetched when the overlay is first switched on. */
export function OsmOverlay() {
  const scene = useScene((s) => s.scene);
  const on = useView((s) => s.osm);
  const mode = useView((s) => s.mode);
  const meshBuilding = useScene((s) => s.meshBuilding);
  // re-drape when the meshed surface changes (3D object classes flatten it)
  const objectsKey = useView((s) => objectKindsKey(activeObjectKinds(s.objectKinds, scene?.objects)));
  const features = useOsm((s) => osmFeaturesFor(s, scene));
  const invalidate = useThree((s) => s.invalidate);
  const group = useRef<THREE.Group>(null);

  useEffect(() => {
    if (on && scene && canGeolocate(scene.georef)) void loadOsm(scene);
  }, [on, scene]);

  const geometry = useMemo(() => {
    if (!scene || !features?.length || meshBuilding) return null;
    const { width: W, height: H } = scene.heights;
    const gsd = scene.gsd;
    const surface = objectSurfaceIfReady(scene, useView.getState().objectKinds) ?? scene.heights.data;
    const grid = { data: surface, width: W, height: H };
    const base = sceneBase(scene);
    // subdivide every ~2 m so lines follow the ground instead of cutting through it
    const segs = osmSegments(features, W, H, Math.max(1, 2 / gsd));
    const pos: number[] = [];
    const col: number[] = [];
    const c = new THREE.Color();
    for (const kind of ORDER) {
      const list = segs.get(kind);
      if (!list?.length) continue;
      c.set(OSM_KIND_COLORS[kind]);
      for (let i = 0; i < list.length; i += 2) {
        const cc = list[i];
        const rr = list[i + 1];
        const h = sampleBilinear(grid, cc, rr);
        pos.push((cc - (W - 1) / 2) * gsd, (Number.isFinite(h) ? h : base) - base + LIFT_M, (rr - (H - 1) / 2) * gsd);
        col.push(c.r, c.g, c.b);
      }
    }
    if (!pos.length) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.Float32BufferAttribute(col, 3));
    g.computeBoundingSphere();
    return g;
    // objectsKey: the surface read above depends on it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, features, meshBuilding, objectsKey]);
  useEffect(() => () => geometry?.dispose(), [geometry]);

  // 2D map views: always on top (the mesh is flattened to ~0 there); 3D: hidden behind hills and buildings.
  const material = useMemo(() => new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.95, toneMapped: false }), []);
  useEffect(() => () => material.dispose(), [material]);
  useEffect(() => {
    material.depthTest = mode === 'dsm3d';
    material.needsUpdate = true;
    invalidate();
  }, [material, mode, invalidate]);

  useEffect(() => {
    const apply = () => {
      if (group.current) group.current.scale.y = verticalScale();
      invalidate();
    };
    apply();
    return useView.subscribe(apply);
  }, [invalidate, geometry, on]);

  if (!on || !geometry) return null;
  return (
    <group ref={group} name="osm-overlay" scale={[1, verticalScale(), 1]}>
      <lineSegments geometry={geometry} material={material} renderOrder={5} />
    </group>
  );
}
