import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useFrame, useThree } from '@react-three/fiber';
import { useScene } from '@/store/scene';
import { registerViewportApi, useCamera } from '@/store/camera';
import { heightAtWorld } from '@/lib/pick';
import { visibleFrame, sceneExtent } from '../terrainState';
import { headingOf, usePressedKeys } from './keys';

/** Arcade flight model: throttle, pitch, roll with coordinated (banked) turns, yaw, terrain collision.
 *
 *  W / S  throttle      ↑ / ↓  pitch      ← / →  roll (A / D also roll)      Q / E  yaw
 *  Shift  boost          R      level out  Gamepad: left stick pitch/roll, triggers throttle. */
export function FlightRig() {
  const camera = useThree((s) => s.camera);
  const scene = useScene((s) => s.scene);
  const keys = usePressedKeys(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'Space']);
  const state = useRef({ speed: 0, throttle: 0.5, pitchRate: 0, rollRate: 0 });
  const tmp = useRef({ fwd: new THREE.Vector3(), q: new THREE.Quaternion(), e: new THREE.Euler(0, 0, 0, 'YXZ'), up: new THREE.Vector3() });
  const lastHud = useRef(0);

  const cruise = () => {
    if (!scene) return 40;
    const e = sceneExtent(scene);
    return THREE.MathUtils.clamp(Math.max(e.x, e.z) / 18, 12, 250);
  };

  const spawn = () => {
    const f = visibleFrame();
    if (!f || !scene) return;
    const e = sceneExtent(scene);
    const alt = Math.max(e.h * f.scaleY * 1.8, Math.max(e.x, e.z) * 0.12);
    camera.position.set(0, alt, e.z * 0.62);
    camera.quaternion.setFromEuler(new THREE.Euler(-0.18, 0, 0, 'YXZ'));
    state.current = { speed: cruise(), throttle: 0.5, pitchRate: 0, rollRate: 0 };
  };

  useEffect(() => {
    spawn();
    const pc = camera as THREE.PerspectiveCamera;
    if (pc.isPerspectiveCamera) {
      pc.near = 0.5;
      pc.updateProjectionMatrix();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, scene]);

  useEffect(
    () =>
      registerViewportApi({
        reset: spawn,
        zoom: (dir) => {
          state.current.throttle = THREE.MathUtils.clamp(state.current.throttle + dir * 0.15, 0, 1);
        },
        faceNorth: () => {
          const { e } = tmp.current;
          e.setFromQuaternion(camera.quaternion, 'YXZ');
          e.y = 0;
          camera.quaternion.setFromEuler(e);
        },
      }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [camera, scene],
  );

  useFrame((_, rawDt) => {
    const f = visibleFrame();
    if (!f || !scene) return;
    const dt = Math.min(rawDt, 0.05);
    const k = keys.current;
    const s = state.current;
    const pad = navigator.getGamepads?.()[0] ?? null;
    const axis = (i: number) => (pad && Math.abs(pad.axes[i] ?? 0) > 0.12 ? pad.axes[i] : 0);

    if (k.has('KeyW')) s.throttle = Math.min(1, s.throttle + dt * 0.5);
    if (k.has('KeyS')) s.throttle = Math.max(0, s.throttle - dt * 0.5);
    if (pad) s.throttle = THREE.MathUtils.clamp(s.throttle + ((pad.buttons[7]?.value ?? 0) - (pad.buttons[6]?.value ?? 0)) * dt * 0.6, 0, 1);

    const pitchIn = (k.has('ArrowUp') ? 1 : 0) - (k.has('ArrowDown') ? 1 : 0) - axis(1);
    const rollIn = (k.has('ArrowLeft') || k.has('KeyA') ? 1 : 0) - (k.has('ArrowRight') || k.has('KeyD') ? 1 : 0) - axis(0);
    const yawIn = (k.has('KeyQ') ? 1 : 0) - (k.has('KeyE') ? 1 : 0);
    s.pitchRate += (pitchIn * 0.9 - s.pitchRate) * Math.min(1, dt * 4);
    s.rollRate += (rollIn * 1.6 - s.rollRate) * Math.min(1, dt * 4);

    const { q, e, fwd, up } = tmp.current;
    // Body-axis rotations.
    q.setFromAxisAngle(new THREE.Vector3(1, 0, 0), s.pitchRate * dt);
    camera.quaternion.multiply(q);
    q.setFromAxisAngle(new THREE.Vector3(0, 0, 1), s.rollRate * dt);
    camera.quaternion.multiply(q);
    // Coordinated turn: bank angle induces yaw about world up; Q/E add rudder.
    e.setFromQuaternion(camera.quaternion, 'YXZ');
    const bankYaw = Math.sin(e.z) * 0.9 + yawIn * 0.6;
    q.setFromAxisAngle(new THREE.Vector3(0, 1, 0), bankYaw * dt);
    camera.quaternion.premultiply(q);
    // Auto-level roll when there is no roll input; R levels fully.
    if (Math.abs(rollIn) < 0.01 || k.has('KeyR')) {
      e.setFromQuaternion(camera.quaternion, 'YXZ');
      e.z *= 1 - Math.min(1, dt * (k.has('KeyR') ? 6 : 0.8));
      if (k.has('KeyR')) e.x *= 1 - Math.min(1, dt * 6);
      camera.quaternion.setFromEuler(e);
    }

    const boost = k.has('ShiftLeft') || k.has('ShiftRight') ? 2.2 : 1;
    const target = cruise() * (0.25 + s.throttle * 1.75) * boost * useCamera.getState().moveSpeed;
    s.speed += (target - s.speed) * Math.min(1, dt * 1.5);
    camera.getWorldDirection(fwd);
    camera.position.addScaledVector(fwd, s.speed * dt);

    // Terrain collision + soft ceiling + soft boundary.
    const ext = sceneExtent(scene);
    const ground = ((heightAtWorld(f, camera.position.x, camera.position.z) ?? f.base) - f.base) * f.scaleY;
    const clearance = 2.5;
    if (camera.position.y < ground + clearance) {
      camera.position.y = ground + clearance;
      e.setFromQuaternion(camera.quaternion, 'YXZ');
      if (e.x < 0) e.x *= 0.5;
      camera.quaternion.setFromEuler(e);
    }
    const ceiling = Math.max(ext.x, ext.z) * 2 + ext.h * f.scaleY * 3;
    camera.position.y = Math.min(camera.position.y, ceiling);
    const lim = Math.max(ext.x, ext.z) * 1.6;
    camera.position.x = THREE.MathUtils.clamp(camera.position.x, -lim, lim);
    camera.position.z = THREE.MathUtils.clamp(camera.position.z, -lim, lim);

    const now = performance.now();
    if (now - lastHud.current > 100) {
      lastHud.current = now;
      e.setFromQuaternion(camera.quaternion, 'YXZ');
      up.set(0, 1, 0);
      useCamera.getState().set({
        heading: headingOf(fwd.x, fwd.z),
        isHome: false,
        camXZ: [camera.position.x, camera.position.z],
        flight: {
          altitude: camera.position.y / f.scaleY + f.base,
          agl: (camera.position.y - ground) / f.scaleY,
          speed: s.speed,
          pitch: THREE.MathUtils.radToDeg(e.x),
          roll: THREE.MathUtils.radToDeg(e.z),
        },
      });
    }
  });

  return null;
}
