import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { PointerLockControls } from '@react-three/drei';
import { useScene } from '@/store/scene';
import { useCamera, registerViewportApi } from '@/store/camera';
import { heightAtWorld } from '@/lib/pick';
import { visibleFrame, sceneExtent } from '../terrainState';
import { headingOf, usePressedKeys } from './keys';

export const EYE_HEIGHT = 1.7;

/** First-person navigation: pointer-lock mouse look, WASD, terrain following at eye height. */
export function WalkRig() {
  const camera = useThree((s) => s.camera);
  const scene = useScene((s) => s.scene);
  const keys = usePressedKeys(['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight']);
  const lift = useRef(0);
  const fwd = useRef(new THREE.Vector3());
  const right = useRef(new THREE.Vector3());
  const lastHud = useRef(0);

  // Start at the scene centre, looking north.
  useEffect(() => {
    const f = visibleFrame();
    if (!f || !scene) return;
    const ground = (heightAtWorld(f, 0, 0) ?? f.base) - f.base;
    camera.position.set(0, ground * f.scaleY + EYE_HEIGHT, sceneExtent(scene).z * 0.3);
    camera.lookAt(0, ground * f.scaleY + EYE_HEIGHT, -1e6);
    lift.current = 0;
    if ((camera as THREE.PerspectiveCamera).isPerspectiveCamera) {
      const pc = camera as THREE.PerspectiveCamera;
      pc.near = 0.1;
      pc.updateProjectionMatrix();
    }
  }, [camera, scene]);

  useEffect(
    () =>
      registerViewportApi({
        reset: () => {
          const f = visibleFrame();
          if (!f) return;
          lift.current = 0;
          camera.position.set(0, ((heightAtWorld(f, 0, 0) ?? f.base) - f.base) * f.scaleY + EYE_HEIGHT, 0);
        },
        zoom: (dir) => {
          const pc = camera as THREE.PerspectiveCamera;
          pc.fov = THREE.MathUtils.clamp(pc.fov - dir * 8, 20, 90);
          pc.updateProjectionMatrix();
        },
        faceNorth: () => camera.lookAt(camera.position.x, camera.position.y, camera.position.z - 1000),
      }),
    [camera],
  );

  useFrame((_, dt) => {
    const f = visibleFrame();
    if (!f || !scene) return;
    const k = keys.current;
    const speedMul = useCamera.getState().moveSpeed;
    const speed = (k.has('ShiftLeft') || k.has('ShiftRight') ? 22 : 6) * speedMul;
    camera.getWorldDirection(fwd.current);
    fwd.current.y = 0;
    fwd.current.normalize();
    right.current.crossVectors(fwd.current, camera.up).normalize();
    const step = Math.min(dt, 0.1) * speed;
    if (k.has('KeyW') || k.has('ArrowUp')) camera.position.addScaledVector(fwd.current, step);
    if (k.has('KeyS') || k.has('ArrowDown')) camera.position.addScaledVector(fwd.current, -step);
    if (k.has('KeyD') || k.has('ArrowRight')) camera.position.addScaledVector(right.current, step);
    if (k.has('KeyA') || k.has('ArrowLeft')) camera.position.addScaledVector(right.current, -step);
    if (k.has('Space')) lift.current += step;
    if (k.has('KeyC')) lift.current = Math.max(0, lift.current - step);
    // Keep inside the scene footprint.
    const e = sceneExtent(scene);
    camera.position.x = THREE.MathUtils.clamp(camera.position.x, -e.x / 2, e.x / 2);
    camera.position.z = THREE.MathUtils.clamp(camera.position.z, -e.z / 2, e.z / 2);
    const ground = ((heightAtWorld(f, camera.position.x, camera.position.z) ?? f.base) - f.base) * f.scaleY;
    const targetY = ground + EYE_HEIGHT + lift.current;
    camera.position.y += (targetY - camera.position.y) * Math.min(1, dt * 12);

    const now = performance.now();
    if (now - lastHud.current > 100) {
      lastHud.current = now;
      camera.getWorldDirection(fwd.current);
      useCamera.getState().set({
        heading: headingOf(fwd.current.x, fwd.current.z),
        isHome: false,
        camXZ: [camera.position.x, camera.position.z],
        flight: { altitude: camera.position.y / f.scaleY + f.base, agl: (camera.position.y - ground) / f.scaleY, speed: 0, pitch: 0, roll: 0 },
      });
    }
  });

  return <PointerLockControls selector="#dw-lock-target" />;
}
