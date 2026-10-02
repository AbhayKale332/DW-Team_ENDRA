import { useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { Html, Line } from '@react-three/drei';
import type { Scene } from '@/domain/types';
import { useTool, type GridPoint } from '@/store/tool';
import { gridToWorld, type TerrainFrame } from '@/lib/pick';
import { sampleBilinear } from '@/lib/heights';
import { formatHeight } from '@/lib/product';
import { cleanPoints, formatArea, formatLength, formatSigned, heightDiff, pathLength, polygonArea } from '@/lib/measure';
import { approx, heightSampler } from '@/features/analysis/measureReadout';
import { LIFT, MEASURE_PIN, MEASURE_STEM, Pin, drape } from './pins';
import classes from '@/features/viewport/overlays/overlays.module.css';

const LINE = '#ffd166';
const FIRST = '#2d6cdf';
const NEXT = '#e5793a';
type V3 = [number, number, number];

/** A read-out pinned to a point in the scene. DOM, so it stays crisp; below the floating panels, never in the way of a click. */
function Label({ at, children, strong, above }: { at: V3; children: React.ReactNode; strong?: boolean; above?: boolean }) {
  return (
    <Html position={at} center zIndexRange={[2, 0]} style={{ pointerEvents: 'none' }}>
      <div className={classes.measureLabel} data-strong={strong || undefined} data-above={above || undefined}>
        {children}
      </div>
    </Html>
  );
}

const lifted = (f: TerrainFrame, p: GridPoint, h?: number): V3 => {
  const [x, y, z] = gridToWorld(f, p.col, p.row, h);
  return [x, y + LIFT, z];
};
const mid = (a: GridPoint, b: GridPoint): GridPoint => ({ col: (a.col + b.col) / 2, row: (a.row + b.row) / 2 });

/** Polygon fill that follows the terrain: ear-clipped, then every triangle split evenly so it drapes over the surface
 *  (one depth for all, so shared edges match). Capped at ~65k triangles. */
function fillGeometry(f: TerrainFrame, pts: GridPoint[]): THREE.BufferGeometry | null {
  const ring = cleanPoints(pts, true);
  if (ring.length < 3) return null;
  const tris = THREE.ShapeUtils.triangulateShape(
    ring.map((p) => new THREE.Vector2(p.col, p.row)),
    [],
  );
  if (!tris.length) return null;
  let longest = 0;
  for (let i = 0; i < ring.length; i++) longest = Math.max(longest, Math.hypot(ring[i].col - ring[(i + 1) % ring.length].col, ring[i].row - ring[(i + 1) % ring.length].row));
  let depth = 0;
  while (depth < 6 && longest / 2 ** depth > 4 && tris.length * 4 ** (depth + 1) <= 65536) depth++;
  const pos: number[] = [];
  const emit = (a: GridPoint, b: GridPoint, c: GridPoint, d: number) => {
    if (d === 0) {
      for (const p of [a, b, c]) pos.push(...lifted(f, p));
      return;
    }
    const ab = mid(a, b);
    const bc = mid(b, c);
    const ca = mid(c, a);
    emit(a, ab, ca, d - 1);
    emit(ab, b, bc, d - 1);
    emit(ca, bc, c, d - 1);
    emit(ab, bc, ca, d - 1);
  };
  for (const [i, j, k] of tris) emit(ring[i], ring[j], ring[k], depth);
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
  return g;
}

/** Measure tool in 3D: points, the line or polygon with per-segment labels, the rubber band to the pointer, and for
 *  Height a vertical line (with the horizontal leg for a pair) labelled with the height or Δh. */
export function MeasureMarkers({ scene, frame, size, stem }: { scene: Scene; frame: TerrainFrame; size: number; stem: number }) {
  const active = useTool((s) => s.tool === 'measure');
  const mode = useTool((s) => s.measureMode);
  const pts = useTool((s) => s.measure);
  const done = useTool((s) => s.measureDone);
  // the rubber band needs the pointer; only subscribe to it while one is drawn
  const rubber = active && !done && pts.length > 0 && !(mode === 'height' && pts.length >= 2);
  const hover = useTool((s) => (rubber ? s.hover : null));
  const ax = approx(scene);

  const isArea = mode === 'area';
  const ring = isArea && pts.length >= 3;
  const line = useMemo(() => {
    if (mode === 'height' || pts.length < 2) return null;
    return drape(frame, done && ring ? [...pts, pts[0]] : pts, LIFT);
  }, [frame, mode, pts, done, ring]);
  const segments = useMemo(() => (mode === 'length' ? pathLength(pts, scene.gsd).segments : []), [mode, pts, scene.gsd]);
  const fill = useMemo(() => (ring ? fillGeometry(frame, pts) : null), [frame, pts, ring]);
  useEffect(() => () => fill?.dispose(), [fill]);

  const band = useMemo(() => {
    if (!rubber || !hover) return null;
    const last = pts[pts.length - 1];
    const path = drape(frame, [last, hover], LIFT);
    // the polygon's closing edge, previewed back to the first corner
    const close = isArea && pts.length >= 2 ? drape(frame, [hover, pts[0]], LIFT) : null;
    let text = ax + formatLength(pathLength([last, hover], scene.gsd).horizontal);
    if (mode === 'height') {
      const h = heightSampler(scene);
      text = `Δh ${formatSigned(h(hover.col, hover.row) - h(last.col, last.row))} · ${text}`;
    }
    return { path, close, at: lifted(frame, mid(last, hover)), text };
  }, [rubber, hover, pts, frame, isArea, mode, scene, ax]);

  const height = useMemo(() => {
    if (mode !== 'height' || !pts.length) return null;
    const h = heightSampler(scene);
    const [a, b] = pts;
    const ha = h(a.col, a.row);
    if (!b) {
      // a spot height: the stick from the ground under it (0 for heights above ground, the DEM terrain for a DSM)
      const ground = scene.product !== 'DSM' ? 0 : scene.terrain ? sampleBilinear(scene.terrain, a.col, a.row) : null;
      const top = lifted(frame, a, ha);
      return { lines: ground === null ? [] : [[lifted(frame, a, ground), top]], legs: [], at: top, text: formatHeight(scene, ha, 1), spot: true };
    }
    const d = heightDiff(a, b, ha, h(b.col, b.row), scene.gsd);
    // vertical at the higher point down to the lower one's level, and the level leg back to the lower point
    const [lo, hi, hLo, hHi] = d.dh >= 0 ? [a, b, d.a, d.b] : [b, a, d.b, d.a];
    const foot = lifted(frame, hi, hLo);
    const top = lifted(frame, hi, hHi);
    return { lines: [[foot, top]], legs: [[lifted(frame, lo, hLo), foot]], at: lifted(frame, hi, (hLo + hHi) / 2), text: `Δh ${formatSigned(d.dh)}`, spot: false };
  }, [mode, pts, scene, frame]);

  if (!active || !pts.length) return null;
  const pinSize = size * MEASURE_PIN;
  const pinStem = stem * MEASURE_STEM;
  return (
    <group name="measure">
      {pts.map((p, i) => (
        <Pin key={`m${i}`} frame={frame} p={p} color={i === 0 ? FIRST : NEXT} size={pinSize} stem={pinStem} />
      ))}
      {mode === 'height' &&
        pts.slice(0, 2).map((p, i) => (
          <Label key={`l${i}`} at={lifted(frame, p).map((v, k) => (k === 1 ? v + pinStem : v)) as V3} above>
            {i === 0 ? 'A' : 'B'}
            {height?.spot ? ` · ${height.text}` : ''}
          </Label>
        ))}
      {line && <Line points={line} color={LINE} lineWidth={3} depthTest={false} renderOrder={9} />}
      {fill && (
        <mesh geometry={fill} renderOrder={8}>
          <meshBasicMaterial color={LINE} transparent opacity={0.38} depthTest={false} depthWrite={false} side={THREE.DoubleSide} />
        </mesh>
      )}
      {mode === 'length' && segments.map((m, i) => (
        <Label key={`s${i}`} at={lifted(frame, mid(pts[i], pts[i + 1]))}>
          {ax + formatLength(m)}
        </Label>
      ))}
      {ring && (
        <Label at={lifted(frame, { col: pts.reduce((s, p) => s + p.col, 0) / pts.length, row: pts.reduce((s, p) => s + p.row, 0) / pts.length })} strong>
          {ax + formatArea(polygonArea(pts, scene.gsd))}
        </Label>
      )}
      {height?.lines.map((l, i) => <Line key={`v${i}`} points={l} color={LINE} lineWidth={3} depthTest={false} renderOrder={9} />)}
      {height?.legs.map((l, i) => <Line key={`h${i}`} points={l} color={LINE} lineWidth={2} dashed dashSize={size * 1.5} gapSize={size} depthTest={false} renderOrder={9} />)}
      {height && !height.spot && (
        <Label at={height.at} strong>
          {height.text}
        </Label>
      )}
      {band && (
        <>
          <Line points={band.path} color={LINE} lineWidth={2} dashed dashSize={size * 1.5} gapSize={size} depthTest={false} renderOrder={9} />
          {band.close && <Line points={band.close} color={LINE} lineWidth={1.5} dashed dashSize={size} gapSize={size} depthTest={false} renderOrder={9} transparent opacity={0.6} />}
          <Label at={band.at}>{band.text}</Label>
        </>
      )}
    </group>
  );
}
