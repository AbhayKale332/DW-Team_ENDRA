import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { Sky } from '@react-three/drei';
import { useComputedColorScheme } from '@mantine/core';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { sunDirection } from './terrainMaterial';
import { sceneExtent } from '../terrainState';
import { useBasemapShown } from '@/features/basemap/basemapState';

/** Sky, fog, sun and fill light. Daylight sky in the light theme, night backdrop in the dark theme. */
export function Environment() {
  const scheme = useComputedColorScheme('light');
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const az = useView((s) => s.sunAzimuth);
  const el = useView((s) => s.sunElevation);
  const shadows = useView((s) => s.shadows);
  const exaggeration = useView((s) => s.exaggeration);
  const basemap = useBasemapShown();
  const sun = useRef<THREE.DirectionalLight>(null);
  const three = useThree();
  const is3d = mode === 'dsm3d';

  const extent = scene ? sceneExtent(scene) : { x: 200, z: 200, h: 20 };
  const radius = Math.max(extent.x, extent.z) * 0.75 + extent.h * exaggeration;
  const dir = useMemo(() => sunDirection(az, el), [az, el]);

  useEffect(() => {
    const l = sun.current;
    if (!l) return;
    l.position.copy(dir).multiplyScalar(radius * 2.2);
    l.target.position.set(0, 0, 0);
    l.target.updateMatrixWorld();
    const c = l.shadow.camera as THREE.OrthographicCamera;
    c.left = -radius;
    c.right = radius;
    c.top = radius;
    c.bottom = -radius;
    c.near = radius * 0.2;
    c.far = radius * 4.5;
    c.updateProjectionMatrix();
    l.shadow.bias = -0.0004;
    l.shadow.normalBias = Math.max(0.02, radius / 4000);
    three.invalidate();
  }, [dir, radius, three]);

  const bg = scheme === 'dark' ? '#0c1016' : '#cfdbe8';
  const dist = Math.max(extent.x, extent.z);

  useEffect(() => {
    three.scene.background = is3d ? null : new THREE.Color(scheme === 'dark' ? '#0d1116' : '#e7ecf2');
    // the surroundings reach 2.5 scene sizes from the centre: keep them out of the fog
    three.scene.fog = is3d ? (basemap ? new THREE.Fog(bg, dist * 4, dist * 12) : new THREE.Fog(bg, dist * 2.2, dist * 7)) : null;
    three.invalidate();
  }, [is3d, scheme, bg, dist, basemap, three]);

  return (
    <>
      {is3d && scheme === 'light' && (
        <Sky distance={dist * 40 + 4000} sunPosition={[dir.x, Math.max(dir.y, 0.05), dir.z]} turbidity={6} rayleigh={1.2} mieCoefficient={0.004} mieDirectionalG={0.85} />
      )}
      {is3d && scheme === 'dark' && <color attach="background" args={['#0b0f15']} />}
      <hemisphereLight args={[scheme === 'dark' ? '#8ea3c7' : '#dbe8ff', scheme === 'dark' ? '#1a1c1f' : '#4b4a45', scheme === 'dark' ? 0.55 : 0.85]} />
      <directionalLight
        ref={sun}
        color={scheme === 'dark' ? '#d9e2ff' : '#fff4e2'}
        intensity={is3d ? (scheme === 'dark' ? 1.6 : 2.4) : 0}
        castShadow={is3d && shadows}
        shadow-mapSize={[2048, 2048]}
      />
      <ambientLight intensity={scheme === 'dark' ? 0.12 : 0.18} />
    </>
  );
}
