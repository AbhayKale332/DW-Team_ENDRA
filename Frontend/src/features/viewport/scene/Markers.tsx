import { useMemo } from 'react';
import { Line } from '@react-three/drei';
import { useScene } from '@/store/scene';
import { useTool } from '@/store/tool';
import { useView } from '@/store/view';
import { useGcp } from '@/features/gcp/gcpStore';
import type { TerrainFrame } from '@/lib/pick';
import { FLAT_SCALE, sceneBase } from '../terrainState';
import { LIFT, Pin, drape, pinScale } from './pins';
import { MeasureMarkers } from './MeasureMarkers';

/** Probe pin, measurement (see MeasureMarkers) and profile polyline, drawn in scene space. */
export function Markers() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const exaggeration = useView((s) => s.exaggeration);
  const probe = useTool((s) => s.probe);
  const profile = useTool((s) => s.profile);
  const cursor = useTool((s) => s.profileCursor);
  const gcpScene = useGcp((s) => s.sceneId);
  const gcps = useGcp((s) => s.points);

  const frame = useMemo<TerrainFrame | null>(
    () => (scene ? { grid: scene.heights, gsd: scene.gsd, base: sceneBase(scene), scaleY: mode === 'dsm3d' ? exaggeration : FLAT_SCALE } : null),
    [scene, mode, exaggeration],
  );
  const { size, stem } = scene ? pinScale(scene, mode) : { size: 1, stem: 0 };

  const profileLine = useMemo(() => (frame && profile.length > 1 ? drape(frame, profile, LIFT) : null), [frame, profile]);
  const cursorPt = useMemo(() => {
    if (!profileLine || cursor === null) return null;
    return profileLine[Math.min(profileLine.length - 1, Math.round(cursor * (profileLine.length - 1)))];
  }, [profileLine, cursor]);

  if (!frame) return null;
  return (
    <group name="markers">
      {gcpScene === scene?.id && gcps.map((p) => <Pin key={p.id} frame={frame} p={p} color="#12b886" size={size} stem={stem} />)}
      {probe && <Pin frame={frame} p={probe} color="#2d6cdf" size={size} stem={stem} />}
      {scene && <MeasureMarkers scene={scene} frame={frame} size={size} stem={stem} />}
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
