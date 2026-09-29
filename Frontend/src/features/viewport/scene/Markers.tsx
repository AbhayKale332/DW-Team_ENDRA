import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { useScene } from '@/store/scene';
import { useTool, type GridPoint } from '@/store/tool';
import { useView } from '@/store/view';
import { gridToWorld, type TerrainFrame } from '@/lib/pick';
import { FLAT_SCALE, sceneBase, sceneExtent } from '../terrainState';

const LIFT = 0.25;

function Pin({ frame, p, color, size, stem }: { frame: TerrainFrame; p: GridPoint; color: string; size: number; stem: number }) {
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
function drape(frame: TerrainFrame, pts: GridPoint[], lift: number): Array<[number, number, number]> {
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

/** Probe pin, measurement line and profile polyline, drawn in scene space. */
export function Markers() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const exaggeration = useView((s) => s.exaggeration);
  const probe = useTool((s) => s.probe);
  const measure = useTool((s) => s.measure);
  const profile = useTool((s) => s.profile);
  const cursor = useTool((s) => s.profileCursor);

  const frame = useMemo<TerrainFrame | null>(
    () => (scene ? { grid: scene.heights, gsd: scene.gsd, base: sceneBase(scene), scaleY: mode === 'dsm3d' ? exaggeration : FLAT_SCALE } : null),
    [scene, mode, exaggeration],
  );
  const e = scene ? sceneExtent(scene) : null;
  const size = e ? Math.max(e.x, e.z) / 180 : 1;
  const stem = mode === 'dsm3d' ? size * 6 : 0;

  const profileLine = useMemo(() => (frame && profile.length > 1 ? drape(frame, profile, LIFT) : null), [frame, profile]);
  const measureLine = useMemo(() => {
    if (!frame || measure.length < 2) return null;
    return measure.map((p) => {
      const [x, y, z] = gridToWorld(frame, p.col, p.row);
      return [x, y + LIFT, z] as [number, number, number];
    });
  }, [frame, measure]);
  const cursorPt = useMemo(() => {
    if (!profileLine || cursor === null) return null;
    return profileLine[Math.min(profileLine.length - 1, Math.round(cursor * (profileLine.length - 1)))];
  }, [profileLine, cursor]);

  if (!frame) return null;
  return (
    <group name="markers">
      {probe && <Pin frame={frame} p={probe} color="#2d6cdf" size={size} stem={stem} />}
      {measure.map((p, i) => (
        <Pin key={`m${i}`} frame={frame} p={p} color={i === 0 ? '#2d6cdf' : '#e5793a'} size={size} stem={stem} />
      ))}
      {measureLine && <Line points={measureLine} color="#ffd166" lineWidth={3} depthTest={false} renderOrder={9} />}
      {profile.map((p, i) => (
        <Pin key={`p${i}`} frame={frame} p={p} color="#6a4cf0" size={size * 0.8} stem={stem * 0.6} />
      ))}
      {profileLine && <Line points={profileLine} color="#6a4cf0" lineWidth={3} depthTest={false} renderOrder={9} />}
      {cursorPt && (
        <mesh position={cursorPt} renderOrder={11}>
          <sphereGeometry args={[size * 0.9, 16, 12]} />
          <meshBasicMaterial color="#ffd166" depthTest={false} />
        </mesh>
      )}
    </group>
  );
}
