import { Group, Loader, Text } from '@mantine/core';
import { ProductBadge } from '@/components/ProductBadge';
import { useAnchor } from '@/store/anchor';
import { useScene } from '@/store/scene';

/** Viewport chip: which product this is (rDSM / nDSM / absolute DSM) and, while it runs, the DEM anchoring step. */
export function ProductChip() {
  const scene = useScene((s) => s.scene);
  const anchor = useAnchor();
  if (!scene) return null;
  const mine = anchor.sceneId === scene.id;
  const running = mine && anchor.status === 'running';
  return (
    <Group gap={6} className="dw-float" px={8} py={4} wrap="nowrap" role="status" aria-label="Height product">
      <ProductBadge scene={scene} full />
      {running && (
        <>
          <Loader size={12} />
          <Text size="xs" c="dimmed">
            {anchor.message ?? 'Anchoring to DEM'}…
          </Text>
        </>
      )}
      {!running && scene.product === 'nDSM' && mine && anchor.status === 'error' && (
        <Text size="xs" c="dimmed">
          not anchored — see Info
        </Text>
      )}
      {!running && scene.product === 'DSM' && scene.anchoring && (
        <Text size="xs" c="dimmed">
          {scene.anchoring.sourceId === 'gcp' ? 'GCP-anchored' : scene.anchoring.sourceId === 'local-file' ? 'local DEM' : 'DEM-anchored'}
        </Text>
      )}
    </Group>
  );
}
