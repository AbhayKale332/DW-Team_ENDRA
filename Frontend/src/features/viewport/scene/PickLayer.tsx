import { useEffect } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { useScene } from '@/store/scene';
import { finishMeasure, useTool } from '@/store/tool';
import { useView } from '@/store/view';
import { useCamera } from '@/store/camera';
import { gridToWorld, pickHeightfield } from '@/lib/pick';
import { currentFrame } from '../terrainState';
import { useUseCases } from '@/store/usecases';
import { addTower, runFlood } from '@/features/usecases/actions';
import { addGcp, useGcp } from '@/features/gcp/gcpStore';
import { MEASURE_STEM, pinScale } from './pins';

/** Second click of a double-click: this soon and this close (CSS px) to the first. */
const DOUBLE_MS = 400;
const DOUBLE_PX = 6;
/** How close (CSS px) a click must land to the first corner to close a polygon. */
const CLOSE_PX = 12;

/** Converts pointer clicks/hover on the canvas into grid coordinates via height-field ray marching. A click (never a
 *  drag) adds a point; for the measure tool a double-click finishes and a click on the first corner closes a polygon. */
export function PickLayer() {
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);

  useEffect(() => {
    const el = gl.domElement;
    const ray = new THREE.Raycaster();
    const ndc = new THREE.Vector2();
    let down: { x: number; y: number; t: number } | null = null;
    let lastMove = 0;
    let lastClick: { x: number; y: number; t: number } | null = null;
    const v = new THREE.Vector3();

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

    // the first corner on screen, at the ground or at its pin's head (both look like "the first point")
    const onFirstCorner = (clientX: number, clientY: number) => {
      const { measureMode, measure, measureDone } = useTool.getState();
      const scene = useScene.getState().scene;
      const f = currentFrame();
      if (measureMode !== 'area' || measureDone || measure.length < 3 || !scene || !f) return false;
      const r = el.getBoundingClientRect();
      const [x, y, z] = gridToWorld(f, measure[0].col, measure[0].row);
      const stem = pinScale(scene, useView.getState().mode).stem * MEASURE_STEM;
      return [y, y + stem].some((yy) => {
        v.set(x, yy, z).project(camera);
        return Math.hypot(r.left + ((v.x + 1) / 2) * r.width - clientX, r.top + ((1 - v.y) / 2) * r.height - clientY) < CLOSE_PX;
      });
    };

    const onDown = (e: PointerEvent) => {
      if (e.button !== 0) return;
      // event times, not handler times: a busy frame must not turn a click into a "slow" one
      down = { x: e.clientX, y: e.clientY, t: e.timeStamp };
      // the canvas takes the focus, so Enter / Backspace / Esc reach the measure tool after using its panel
      if (useTool.getState().tool === 'measure' && document.activeElement !== el) el.focus({ preventScroll: true });
    };
    const onUp = (e: PointerEvent) => {
      if (!down || e.button !== 0) return;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      const quick = e.timeStamp - down.t < 450;
      down = null;
      const { tool } = useTool.getState();
      const uc = useUseCases.getState();
      const gcp = useGcp.getState().picking;
      const armed = uc.placing || uc.pickingSource || gcp; // a placement is waiting for a click
      if ((tool === 'none' && !armed) || moved > 5 || !quick || useCamera.getState().mode !== 'orbit') return;
      if (tool === 'measure' && !armed) {
        const now = e.timeStamp;
        const double = !!lastClick && now - lastClick.t < DOUBLE_MS && Math.hypot(e.clientX - lastClick.x, e.clientY - lastClick.y) < DOUBLE_PX;
        lastClick = double ? null : { x: e.clientX, y: e.clientY, t: now };
        // the first click of the pair already placed the last point
        if (double) return void finishMeasure();
        if (onFirstCorner(e.clientX, e.clientY)) return void finishMeasure();
      }
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
