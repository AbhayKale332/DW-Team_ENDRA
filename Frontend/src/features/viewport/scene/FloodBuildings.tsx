import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { useScene } from '@/store/scene';
import { activeObjectKinds, useView } from '@/store/view';
import { useUseCases } from '@/store/usecases';
import { buildingGeometry, type MeshArrays } from '@/lib/objectGeometry';
import { TIER_COLORS, type Tier } from '@/lib/usecases/flood';
import { loadObjectSurface } from '../objectSurfaceClient';
import { objectFrame } from './objectLayer';

const TIERS: Tier[] = ['critical', 'high', 'watch'];

function toGeometry(a: MeshArrays) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(a.positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(a.normals, 3));
  g.setIndex(new THREE.BufferAttribute(a.index, 1));
  return g;
}

const material = (color: string, wet: boolean) =>
  new THREE.MeshStandardMaterial({
    color: wet ? color : new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.5),
    emissive: wet ? color : '#000000',
    emissiveIntensity: wet ? 0.35 : 0,
    roughness: 0.7,
    metalness: 0,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  });

/** Buildings the flood model reaches, drawn over the building models: tier colour, solid where the water is
 *  already at the footprint and pale where it is still to come. Geometry is rebuilt only when a building crosses
 *  the water line, not on every slider tick. */
export function FloodBuildings() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const buildingsOn = useView((s) => activeObjectKinds(s.objectKinds, scene?.objects).buildings);
  const kinds = useView((s) => s.objectKinds);
  const meshBuilding = useScene((s) => s.meshBuilding);
  const open = useUseCases((s) => s.open);
  const active = useUseCases((s) => s.active);
  const on = useUseCases((s) => s.floodOverlay);
  const risks = useUseCases((s) => s.risks);
  const flood = useUseCases((s) => s.flood);
  const rise = useUseCases((s) => s.rise);
  const focus = useUseCases((s) => s.focusBuilding);
  const exaggeration = useView((s) => s.exaggeration);
  const invalidate = useThree((s) => s.invalidate);
  const [ground, setGround] = useState<Float32Array | null>(null);
  const enabled = !!scene?.objects && buildingsOn && open && active === 'flood' && on && !!flood && mode === 'dsm3d' && !meshBuilding;

  useEffect(() => {
    if (!scene || !enabled) return;
    let cancelled = false;
    void loadObjectSurface(scene, kinds).then((d) => !cancelled && setGround(d));
    return () => {
      cancelled = true;
    };
  }, [scene, enabled, kinds]);

  const mats = useMemo(() => {
    const m = {} as Record<string, THREE.Material>;
    for (const t of TIERS) {
      m[`${t}-wet`] = material(TIER_COLORS[t], true);
      m[`${t}-dry`] = material(TIER_COLORS[t], false);
    }
    return m;
  }, []);
  useEffect(() => () => Object.values(mats).forEach((m) => m.dispose()), [mats]);

  // which buildings are wet at this rise: the only thing that changes geometry
  const wetKey = useMemo(() => {
    if (!flood) return '';
    return risks.map((r) => (r.tier !== 'none' && r.arrival <= flood.baseLevel + rise ? '1' : '0')).join('');
  }, [risks, flood, rise]);

  const groups = useMemo(() => {
    if (!scene?.objects || !enabled || !ground) return [];
    const frame = objectFrame(scene, ground);
    const out: Array<{ key: string; geo: THREE.BufferGeometry[] }> = [];
    for (const tier of TIERS) {
      for (const wet of [true, false]) {
        const list = risks.filter((r, i) => r.tier === tier && (wetKey[i] === '1') === wet).map((r) => scene.objects!.buildings[r.index]);
        if (!list.length) continue;
        const { roof, walls } = buildingGeometry(list, frame);
        out.push({ key: `${tier}-${wet ? 'wet' : 'dry'}`, geo: [toGeometry(roof), toGeometry(walls)] });
      }
    }
    return out;
    // wetKey stands in for `rise`: geometry depends only on which buildings are wet
  }, [scene, enabled, ground, risks, wetKey]);
  useEffect(() => {
    invalidate();
    return () => groups.forEach((g) => g.geo.forEach((x) => x.dispose()));
  }, [groups, invalidate]);

  const focused = useMemo(() => {
    if (!scene?.objects || !ground || focus === null || !enabled) return null;
    const b = scene.objects.buildings[focus];
    if (!b) return null;
    const { roof, walls } = buildingGeometry([b], objectFrame(scene, ground));
    return [toGeometry(roof), toGeometry(walls)];
  }, [scene, ground, focus, enabled]);
  useEffect(() => {
    invalidate();
    return () => focused?.forEach((g) => g.dispose());
  }, [focused, invalidate]);
  const focusMat = useMemo(() => new THREE.MeshBasicMaterial({ color: '#3bc9db', polygonOffset: true, polygonOffsetFactor: -4, polygonOffsetUnits: -4, wireframe: true }), []);
  useEffect(() => () => focusMat.dispose(), [focusMat]);

  if (!enabled || !ground) return null;
  return (
    <group name="flood-buildings" scale={[1, exaggeration, 1]}>
      {groups.map((g) => g.geo.map((geo, i) => <mesh key={`${g.key}${i}`} geometry={geo} material={mats[g.key]} />))}
      {focused?.map((geo, i) => <mesh key={`f${i}`} geometry={geo} material={focusMat} />)}
    </group>
  );
}
