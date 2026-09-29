import { useCallback, useEffect, useRef } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { CameraControls } from '@react-three/drei';
import CameraControlsImpl from 'camera-controls';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { registerViewportApi, useCamera } from '@/store/camera';
import { useReducedMotion } from '@mantine/hooks';
import { sceneExtent } from '../terrainState';
import { useBasemapShown } from '@/features/basemap/basemapState';

const ACTION = CameraControlsImpl.ACTION;
const HOME_AZIMUTH = 0.32;
const HOME_POLAR = 0.98;
const FLAT_POLAR = 1e-4;

/** Orbit navigation for the 3D view, and pan/zoom (north-up, rotation locked) for the 2D views. */
export function OrbitRig({ flat }: { flat: boolean }) {
  const ref = useRef<CameraControlsImpl>(null);
  const scene = useScene((s) => s.scene);
  const exaggeration = useView((s) => s.exaggeration);
  const size = useThree((s) => s.size);
  const camera = useThree((s) => s.camera);
  const reduced = useReducedMotion();
  const basemap = useBasemapShown();
  const homeRef = useRef<{ pos: THREE.Vector3; target: THREE.Vector3; zoom: number } | null>(null);

  const computeHome = useCallback(() => {
    const e = scene ? sceneExtent(scene) : { x: 200, z: 200, h: 20 };
    const span = Math.max(e.x, e.z);
    if (flat) {
      const zoom = Math.min(size.width / Math.max(e.x, 1), size.height / Math.max(e.z, 1)) * 0.88;
      return { pos: new THREE.Vector3(0, span * 2 + 100, 0), target: new THREE.Vector3(0, 0, 0), zoom };
    }
    const r = span * 1.05 + e.h * exaggeration * 1.4;
    const target = new THREE.Vector3(0, e.h * exaggeration * 0.15, 0);
    const pos = new THREE.Vector3(
      target.x + r * Math.sin(HOME_POLAR) * Math.sin(HOME_AZIMUTH),
      target.y + r * Math.cos(HOME_POLAR),
      target.z + r * Math.sin(HOME_POLAR) * Math.cos(HOME_AZIMUTH),
    );
    return { pos, target, zoom: 1 };
  }, [scene, flat, size.width, size.height, exaggeration]);

  // Configure constraints per view.
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const e = scene ? sceneExtent(scene) : { x: 200, z: 200, h: 20 };
    const span = Math.max(e.x, e.z, 10);
    c.smoothTime = reduced ? 0 : 0.22;
    c.draggingSmoothTime = reduced ? 0 : 0.1;
    c.dollyToCursor = true;
    if (flat) {
      c.minPolarAngle = FLAT_POLAR;
      c.maxPolarAngle = FLAT_POLAR;
      c.minAzimuthAngle = 0;
      c.maxAzimuthAngle = 0;
      c.minZoom = 0.02;
      c.maxZoom = 400;
      c.mouseButtons.left = ACTION.TRUCK;
      c.mouseButtons.right = ACTION.TRUCK;
      c.mouseButtons.wheel = ACTION.ZOOM;
      c.touches.one = ACTION.TOUCH_TRUCK;
      c.touches.two = ACTION.TOUCH_ZOOM_TRUCK;
    } else {
      c.minPolarAngle = 0;
      c.maxPolarAngle = Math.PI * 0.495;
      c.minAzimuthAngle = -Infinity;
      c.maxAzimuthAngle = Infinity;
      c.minDistance = Math.max(span * 0.01, 2);
      // far enough to frame the whole surroundings window when the basemap is shown
      c.maxDistance = span * (basemap ? 10 : 6) + e.h * 20;
      c.mouseButtons.left = ACTION.ROTATE;
      c.mouseButtons.right = ACTION.TRUCK;
      c.mouseButtons.wheel = ACTION.DOLLY;
      c.touches.one = ACTION.TOUCH_ROTATE;
      c.touches.two = ACTION.TOUCH_DOLLY_TRUCK;
    }
    // `camera`: drei rebuilds the controls when <Cameras> swaps the default camera (right after mount, and on
    // every 3D ⇄ 2D switch), so everything bound to ref.current must be bound again.
  }, [flat, scene, reduced, camera, basemap]);

  const goHome = useCallback(
    (transition: boolean) => {
      const c = ref.current;
      if (!c) return;
      const h = computeHome();
      homeRef.current = h;
      c.normalizeRotations();
      void c.setLookAt(h.pos.x, h.pos.y, h.pos.z, h.target.x, h.target.y, h.target.z, transition && !reduced);
      if (flat) void c.zoomTo(h.zoom, transition && !reduced);
    },
    [computeHome, flat, reduced],
  );

  // New scene or view switch → go home (no transition). Exaggeration only refreshes the home pose.
  useEffect(() => {
    goHome(false);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene?.id, flat, camera]);
  useEffect(() => {
    homeRef.current = computeHome();
  }, [computeHome]);

  // Heading + "at home" tracking for the compass and the reset button.
  useEffect(() => {
    const c = ref.current;
    if (!c) return;
    const pos = new THREE.Vector3();
    const tgt = new THREE.Vector3();
    let last = 0;
    const onUpdate = () => {
      const now = performance.now();
      if (now - last < 60) return;
      last = now;
      const heading = ((-THREE.MathUtils.radToDeg(c.azimuthAngle) % 360) + 360) % 360;
      const h = homeRef.current;
      let isHome = true;
      if (h) {
        c.getPosition(pos, true);
        c.getTarget(tgt, true);
        const tol = Math.max(h.pos.distanceTo(h.target) * 0.01, 0.05);
        isHome = pos.distanceTo(h.pos) < tol && tgt.distanceTo(h.target) < tol && (!flat || Math.abs(c.camera.zoom - h.zoom) / h.zoom < 0.01);
      }
      const cur = useCamera.getState();
      const mpp = flat ? 1 / Math.max(c.camera.zoom, 1e-6) : cur.mpp;
      if (Math.abs(cur.heading - heading) > 0.5 || cur.isHome !== isHome || Math.abs(cur.mpp - mpp) / mpp > 0.005)
        useCamera.getState().set({ heading: flat ? 0 : heading, isHome, mpp });
    };
    c.addEventListener('update', onUpdate);
    c.addEventListener('rest', onUpdate);
    onUpdate();
    return () => {
      c.removeEventListener('update', onUpdate);
      c.removeEventListener('rest', onUpdate);
    };
  }, [flat, camera]);

  // Imperative API for the DOM overlay controls.
  useEffect(() => {
    return registerViewportApi({
      zoom: (dir) => {
        const c = ref.current;
        if (!c) return;
        if (flat || (c.camera as THREE.OrthographicCamera).isOrthographicCamera) void c.zoom(c.camera.zoom * 0.3 * dir, !reduced);
        else void c.dolly(c.distance * 0.25 * dir, !reduced);
      },
      reset: () => goHome(true),
      faceNorth: () => {
        const c = ref.current;
        if (!c || flat) return;
        c.normalizeRotations();
        void c.rotateAzimuthTo(0, !reduced);
      },
      preset: (p) => {
        const c = ref.current;
        if (!c || flat) return;
        c.normalizeRotations();
        const t = !reduced;
        if (p === 'fit') return goHome(true);
        if (p === 'top') return void c.rotatePolarTo(0.001, t);
        const az = { north: 0, east: -Math.PI / 2, south: Math.PI, west: Math.PI / 2 }[p];
        void c.rotateTo(az, HOME_POLAR, t);
      },
      getPose: () => {
        const c = ref.current;
        if (!c) return null;
        const p = c.getPosition(new THREE.Vector3());
        const t = c.getTarget(new THREE.Vector3());
        return { position: p.toArray() as [number, number, number], target: t.toArray() as [number, number, number] };
      },
      setPose: ({ position, target }) => {
        void ref.current?.setLookAt(...position, ...target, !reduced);
      },
      orbitKey: (dx, dy) => {
        const c = ref.current;
        if (!c) return;
        if (flat) void c.truck(dx * 40 / c.camera.zoom, dy * 40 / c.camera.zoom, !reduced);
        else void c.rotate(THREE.MathUtils.degToRad(dx * 8), THREE.MathUtils.degToRad(dy * 5), !reduced);
      },
    });
  }, [flat, goHome, reduced]);

  return <CameraControls ref={ref} makeDefault />;
}
