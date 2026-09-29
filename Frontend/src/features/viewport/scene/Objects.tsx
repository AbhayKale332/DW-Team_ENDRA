import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import type { ObjectKinds, Scene } from '@/domain/types';
import { useScene } from '@/store/scene';
import { activeObjectKinds, anyObjectKind, objectKindsKey, useView } from '@/store/view';
import { loadObjectSurface } from '../objectSurfaceClient';
import { buildObjectLayer } from './objectLayer';
import { loadImageTexture } from './imageTexture';

/** Detected buildings, trees and water (scene.objects) as 3D models, per class as switched on in the tool
 *  palette / Tools menu / Layers tab. The terrain mesh is flattened under the same classes
 *  (lib/objectSurface), so a model and the bump it replaces never overlap. */
export function Objects() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  // switched on *and* present in this scene
  const kb = useView((s) => activeObjectKinds(s.objectKinds, scene?.objects).buildings);
  const kt = useView((s) => activeObjectKinds(s.objectKinds, scene?.objects).trees);
  const kw = useView((s) => activeObjectKinds(s.objectKinds, scene?.objects).water);
  const shadows = useView((s) => s.shadows);
  // Hidden while the terrain rebuilds: until then the old mesh still carries the bumps (or there is no
  // ground at all), and objects would float or clip.
  const meshBuilding = useScene((s) => s.meshBuilding);
  const gl = useThree((s) => s.gl);
  const invalidate = useThree((s) => s.invalidate);
  const group = useRef<THREE.Group>(null);

  const kinds = useMemo<ObjectKinds>(() => ({ buildings: kb, trees: kt, water: kw }), [kb, kt, kw]);
  const kindsKey = objectKindsKey(kinds);
  const on = !!scene?.objects && anyObjectKind(kinds);

  // the flattened surface the objects stand on, from the terrain worker (same selection as the mesh)
  const [surface, setSurface] = useState<{ scene: Scene; key: string; data: Float32Array } | null>(null);
  useEffect(() => {
    if (!scene || !on) return;
    let cancelled = false;
    void loadObjectSurface(scene, kinds).then((data) => !cancelled && setSurface({ scene, key: kindsKey, data }));
    return () => {
      cancelled = true;
    };
  }, [scene, on, kinds, kindsKey]);
  const ground = surface && surface.scene === scene && surface.key === kindsKey ? surface.data : null;

  const layer = useMemo(() => (scene && on && ground ? buildObjectLayer(scene, ground, kinds) : null), [scene, on, ground, kinds]);
  useEffect(() => () => layer?.dispose(), [layer]);

  // Roofs show the real roof: the optical image, projected straight down. Loaded once per scene.
  const [roofTex, setRoofTex] = useState<THREE.Texture | null>(null);
  const needsRoofs = !!scene?.objects?.buildings.length;
  useEffect(() => {
    if (!scene || !needsRoofs) return;
    let cancelled = false;
    let tex: THREE.Texture | null = null;
    loadImageTexture(scene.image, Math.min(gl.capabilities.maxTextureSize, 4096), gl.capabilities.getMaxAnisotropy())
      .then((t) => {
        if (cancelled) return t.dispose();
        tex = t;
        setRoofTex(t);
      })
      .catch((e) => console.warn('Roof texture unavailable', e));
    return () => {
      cancelled = true;
      setRoofTex(null);
      tex?.dispose();
    };
  }, [scene, needsRoofs, gl]);
  useEffect(() => {
    layer?.setRoofTexture(roofTex);
    invalidate();
  }, [layer, roofTex, invalidate]);

  useEffect(() => {
    layer?.setShadows(shadows);
    invalidate();
  }, [layer, shadows, invalidate]);

  // Follow the vertical exaggeration exactly as the terrain group does, so objects stay rooted.
  useEffect(() => {
    const apply = () => {
      if (group.current) group.current.scale.y = useView.getState().exaggeration;
      invalidate();
    };
    apply();
    return useView.subscribe(apply);
  }, [invalidate, layer, mode, meshBuilding]);

  if (!layer || mode !== 'dsm3d' || meshBuilding) return null;
  return (
    <group ref={group} name="objects-root" scale={[1, useView.getState().exaggeration, 1]}>
      <primitive object={layer.group} />
    </group>
  );
}
