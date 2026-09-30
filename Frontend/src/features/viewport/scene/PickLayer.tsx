import { useEffect } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { useScene } from '@/store/scene';
import { useTool } from '@/store/tool';
import { useCamera } from '@/store/camera';
import { pickHeightfield } from '@/lib/pick';
import { currentFrame } from '../terrainState';
import { useUseCases } from '@/store/usecases';
import { addTower, runFlood } from '@/features/usecases/actions';
import { addGcp, useGcp } from '@/features/gcp/gcpStore';

/** Converts pointer clicks/hover on the canvas into grid coordinates via height-field ray marching. */
export function PickLayer() {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);

  useEffect(() => {
    const el = gl.domElement;
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let down: { x: number; y: number; t: number } | null = null;
    let lastMove = 0;

    const pick = (clientX: number, clientY: number) => {
      const scene = useScene.getState().scene;
      const f = currentFrame();
      if (!scene || !f) return null;
      const r = el.getBoundingClientRect();
      ndc.set(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
      ray.setFromCamera(ndc, camera);
      const hit = pickHeightfield(f, ray.ray.origin, ray.ray.direction, scene.stats.max);
      if (!hit) return null;
      const { width, height } = scene.heights;
      return { col: THREE.MathUtils.clamp(hit.col, 0, width - 1), row: THREE.MathUtils.clamp(hit.row, 0, height - 1) };
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      down = { x: e.clientX, y: e.clientY, t: performance.now() };
    };
    const onUp = (e: PointerEvent) => {
      if (!down || e.button !== 0) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      const quick = performance.now() - down.t < 450;
      down = null;
      const { tool } = useTool.getState();
      const uc = useUseCases.getState();
      const gcp = useGcp.getState().picking;
      const armed = uc.placing || uc.pickingSource || gcp; // a placement is waiting for a click
      if ((tool === 'none' && !armed) || moved > 5 || !quick || useCamera.getState().mode !== 'orbit') return;
      const p = pick(e.clientX, e.clientY);
      if (!p) return;
      if (gcp) return void addGcp(p.col, p.row);
      if (uc.placing) return void addTower(p.col, p.row);
      if (uc.pickingSource) {
        uc.set({ floodPoint: p, floodSource: 'point', pickingSource: false });
        return void runFlood();
      }
      useTool.getState().addPoint(p);
    };
    const onMove = (e: PointerEvent) => {
      const now = performance.now();
      if (now - lastMove < 45 || down || useCamera.getState().mode !== 'orbit') return;
      lastMove = now;
      const hover = pick(e.clientX, e.clientY);
      const r = el.getBoundingClientRect();
      useTool.getState().set({ hover, hoverScreen: hover ? { x: e.clientX - r.left, y: e.clientY - r.top } : null });
    };
    const onLeave = () => useTool.getState().set({ hover: null, hoverScreen: null });

    el.addEventListener('pointerdown', onDown);
    el.addEventListener('pointerup', onUp);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerleave', onLeave);
    return () => {
      el.removeEventListener('pointerdown', onDown);
      el.removeEventListener('pointerup', onUp);
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerleave', onLeave);
    };
  }, [gl, camera]);

  return null;
}
