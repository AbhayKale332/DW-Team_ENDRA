import { useEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { Line } from '@react-three/drei';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { registerViewportApi, useCamera } from '@/store/camera';
import { sceneExtent } from '../terrainState';
import { headingOf } from './keys';

/** Cinematic closed drone path around the scene (ported from the v5 viewer), for demos and recordings. */
export function TourRig() {
  const camera = useThree((s) => s.camera);
  const scene = useScene((s) => s.scene);
  const exaggeration = useView((s) => s.exaggeration);
  const t = useRef(0);
  const look = useRef(new THREE.Vector3());
  const fwd = useRef(new THREE.Vector3());
  const lastHud = useRef(0);

  const curve = useMemo(() => {
    const e = scene ? sceneExtent(scene) : { x: 200, z: 200, h: 20 };
    const rad = Math.max(e.x, e.z) * 0.42;
    const baseAlt = e.h * exaggeration * 1.6 + Math.max(e.x, e.z) * 0.1;
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < 24; i++) {
      const a = (i / 24) * Math.PI * 2;
      const wob = 1 + 0.18 * Math.sin(a * 3);
      pts.push(new THREE.Vector3(rad * wob * Math.cos(a), baseAlt * (0.75 + 0.35 * Math.sin(a * 2)), rad * wob * Math.sin(a)));
    }
    return new THREE.CatmullRomCurve3(pts, true, 'catmullrom', 0.5);
  }, [scene, exaggeration]);

  const length = useMemo(() => curve.getLength(), [curve]);
  const points = useMemo(() => curve.getPoints(300), [curve]);

  useEffect(
    () =>
      registerViewportApi({
        reset: () => {
          t.current = 0;
        },
        zoom: (dir) => {
          const s = useCamera.getState();
          s.set({ tourSpeed: THREE.MathUtils.clamp(s.tourSpeed * (dir > 0 ? 1.25 : 0.8), 0.1, 8) });
        },
      }),
    [],
  );

  useFrame((_, dt) => {
    if (!scene) return;
    const { tourPlaying, tourSpeed } = useCamera.getState();
    const e = sceneExtent(scene);
    const mps = Math.max(e.x, e.z) / 22;
    if (tourPlaying) t.current = (t.current + (Math.min(dt, 0.1) * mps * tourSpeed) / length) % 1;
    const p = curve.getPointAt(t.current);
    camera.position.copy(p);
    // Look a little ahead, blended toward the scene centre so structures stay framed.
    const ahead = curve.getPointAt((t.current + 0.04) % 1);
    look.current.set(ahead.x * 0.35, e.h * exaggeration * 0.2, ahead.z * 0.35);
    camera.lookAt(look.current);
    const now = performance.now();
    if (now - lastHud.current > 150) {
      lastHud.current = now;
      camera.getWorldDirection(fwd.current);
      useCamera.getState().set({ heading: headingOf(fwd.current.x, fwd.current.z), isHome: false, camXZ: [camera.position.x, camera.position.z] });
    }
  });

  return <Line points={points} color="#2d6cdf" lineWidth={1.5} transparent opacity={0.35} />;
}
