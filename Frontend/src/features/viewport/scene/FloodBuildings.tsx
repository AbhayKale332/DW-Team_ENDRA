import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import CustomShaderMaterial from 'three-custom-shader-material/vanilla';
import { useScene } from '@/store/scene';
import { activeObjectKinds, useView } from '@/store/view';
import { useUseCases } from '@/store/usecases';
import { SIM_GLSL, SIM_UNIFORMS_GLSL, simUniforms, useFloodSim } from '@/store/floodSim';
import { buildingGeometry, type MeshArrays } from '@/lib/objectGeometry';
import { stageDepth, TIER_COLORS, type Tier } from '@/lib/usecases/flood';
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

const vertexShader = /* glsl */ `
  varying vec3 vW;
  void main() {
    vW = (modelMatrix * vec4(position, 1.0)).xyz;
  }
`;

// Below the simulated water surface a wall or roof sinks into the water colour, more the deeper it is; a band of
// foam marks the water line.
const fragmentShader = /* glsl */ `
  ${SIM_UNIFORMS_GLSL}
  varying vec3 vW;
  ${SIM_GLSL}
  void main() {
    if (uHasSim > 0.5) {
      vec4 s = simAt(vW.xz * uSimWorld.xy + uSimWorld.zw);
      if (s.r > 0.01) {
        float under = s.g * uExag - vW.y; // world units below the surface
        float m = under / max(uExag, 1e-3);
        if (m > 0.0) {
          float k = 1.0 - exp(-m * 0.6);
          csm_DiffuseColor.rgb = mix(csm_DiffuseColor.rgb * 0.55, vec3(0.05, 0.16, 0.24), 0.3 + 0.6 * k);
          csm_Emissive *= 1.0 - k;
        }
        float band = 1.0 - smoothstep(0.0, max(fwidth(vW.y) * 1.5, 0.12 * uExag), abs(under));
        csm_DiffuseColor.rgb = mix(csm_DiffuseColor.rgb, vec3(0.9, 0.92, 0.9), band * 0.85);
      }
    }
  }
`;

const material = (color: string, wet: boolean) =>
  new CustomShaderMaterial({
    baseMaterial: THREE.MeshStandardMaterial,
    vertexShader,
    fragmentShader,
    uniforms: { ...simUniforms },
    color: wet ? color : new THREE.Color(color).lerp(new THREE.Color('#ffffff'), 0.5),
    emissive: wet ? color : '#000000',
    emissiveIntensity: wet ? 0.35 : 0,
    roughness: 0.7,
    metalness: 0,
    polygonOffset: true,
    polygonOffsetFactor: -2,
    polygonOffsetUnits: -2,
  }) as unknown as THREE.MeshStandardMaterial;

/** Buildings the flood model reaches, drawn over the building models: tier colour, solid where the simulated
 *  water is already at the footprint and pale where it is still to come, and sunk into the water colour below its
 *  surface. Geometry is rebuilt only when a building crosses the water line, not on every frame. */
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
  const live = useFloodSim((s) => s.live);
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

  // which buildings the water has reached: the only thing that changes geometry (the settled level until the
  // simulation reports)
  const wetKey = useMemo(() => {
    if (!flood) return '';
    const wet = (i: number) => {
      const r = risks[i];
      return live ? stageDepth(r, live.stages[r.index]) > 0 : r.arrival <= flood.baseLevel + rise;
    };
    return risks.map((r, i) => (r.tier !== 'none' && wet(i) ? '1' : '0')).join('');
  }, [risks, flood, rise, live]);

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
