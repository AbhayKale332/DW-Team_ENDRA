import { useMemo } from 'react';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { useUseCases } from '@/store/usecases';
import { gridToWorld, type TerrainFrame } from '@/lib/pick';
import { FLAT_SCALE, sceneBase, sceneExtent } from '../terrainState';

/** Telecom towers (masts rooted on the surface) and the suggested additional sites, in scene space.
 *  In the 2D map views the vertical scale is ~0, so a marker sphere stands in for the mast. */
export function TowerMarkers() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const exaggeration = useView((s) => s.exaggeration);
  const open = useUseCases((s) => s.open);
  const active = useUseCases((s) => s.active);
  const towers = useUseCases((s) => s.towers);
  const selected = useUseCases((s) => s.selectedTower);
  const suggestions = useUseCases((s) => s.suggestions);

  const frame = useMemo<TerrainFrame | null>(
    () => (scene ? { grid: scene.heights, gsd: scene.gsd, base: sceneBase(scene), scaleY: mode === 'dsm3d' ? exaggeration : FLAT_SCALE } : null),
    [scene, mode, exaggeration],
  );
  if (!scene || !frame || !open || active !== 'telecom') return null;
  const e = sceneExtent(scene);
  const size = Math.max(e.x, e.z) / 200;
  const is3d = mode === 'dsm3d';

  return (
    <group name="towers">
      {towers.map((t) => {
        const [x, y, z] = gridToWorld(frame, t.col, t.row);
        const mast = is3d ? t.heightM * exaggeration : size * 4;
        const on = t.id === selected;
        const color = on ? '#ffd166' : '#ff5c8a';
        return (
          <group key={t.id} position={[x, y, z]}>
            <mesh position={[0, mast / 2, 0]} renderOrder={12}>
              <cylinderGeometry args={[size * 0.28, size * 0.5, mast, 8]} />
              <meshBasicMaterial color={color} depthTest={false} transparent opacity={0.95} />
            </mesh>
            <mesh position={[0, mast, 0]} renderOrder={13}>
              <sphereGeometry args={[size * (on ? 1.5 : 1.2), 18, 12]} />
              <meshBasicMaterial color={color} depthTest={false} />
            </mesh>
          </group>
        );
      })}
      {suggestions.map((s, i) => {
        const [x, y, z] = gridToWorld(frame, s.col, s.row);
        const mast = is3d ? 25 * exaggeration : size * 4;
        return (
          <group key={`s${i}`} position={[x, y, z]}>
            <mesh position={[0, mast, 0]} renderOrder={13}>
              <sphereGeometry args={[size * 1.4, 18, 12]} />
              <meshBasicMaterial color="#3bc9db" depthTest={false} transparent opacity={0.75} wireframe />
            </mesh>
            <mesh position={[0, mast / 2, 0]} renderOrder={12}>
              <cylinderGeometry args={[size * 0.2, size * 0.2, mast, 6]} />
              <meshBasicMaterial color="#3bc9db" depthTest={false} transparent opacity={0.6} />
            </mesh>
          </group>
        );
      })}
    </group>
  );
}
