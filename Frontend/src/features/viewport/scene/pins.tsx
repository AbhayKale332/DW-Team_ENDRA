import { Line } from '@react-three/drei';
import type { Scene } from '@/domain/types';
import type { ViewMode } from '@/store/view';
import type { GridPoint } from '@/store/tool';
import { gridToWorld, type TerrainFrame } from '@/lib/pick';
import { sceneExtent } from '../terrainState';

/** Shared marker parts: a pin on a stem, and a polyline draped over the terrain. */

export const LIFT = 0.25;

/** Pin sphere radius and stem height for a scene (stems only stand up in the 3D view). */
export function pinScale(scene: Scene, mode: ViewMode) {
  const e = sceneExtent(scene);
  const size = Math.max(e.x, e.z) / 180;
  return { size, stem: mode === 'dsm3d' ? size * 6 : 0 };
}

/** Measure points are a little smaller than the probe pin. */
export const MEASURE_PIN = 0.8;
export const MEASURE_STEM = 0.6;

export function Pin({ frame, p, color, size, stem }: { frame: TerrainFrame; p: GridPoint; color: string; size: number; stem: number }) {
  const [x, y, z] = gridToWorld(frame, p.col, p.row);
  return (
    <group position={[x, y, z]}>
      <mesh position={[0, stem, 0]} renderOrder={10}>
        <sphereGeometry args={[size, 20, 14]} />
        <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.95} />
      </mesh>
      <Line points={[[0, 0, 0], [0, stem, 0]]} color={color} lineWidth={2} depthTest={false} renderOrder={10} />
    </group>
  );
}

/** Samples a polyline across the terrain so the drawn line hugs the surface. */
export function drape(frame: TerrainFrame, pts: GridPoint[], lift: number): Array<[number, number, number]> {
  const out: Array<[number, number, number]> = [];
  for (let i = 0; i < pts.length - 1; i++) {
    const a = pts[i];
    const b = pts[i + 1];
    const n = Math.max(2, Math.ceil(Math.hypot(b.col - a.col, b.row - a.row)));
    for (let k = 0; k <= n; k++) {
      if (i > 0 && k === 0) continue;
      const t = k / n;
      const [x, y, z] = gridToWorld(frame, a.col + (b.col - a.col) * t, a.row + (b.row - a.row) * t);
      out.push([x, y + lift, z]);
    }
  }
  return out;
}
