import { Badge, Tooltip } from '@mantine/core';
import type { Scene } from '@/domain/types';
import { productInfo, type ProductTone } from '@/lib/product';

const COLOR: Record<ProductTone, string> = { orange: 'orange', blue: 'blue', green: 'teal' };

/** The product a scene really is: rDSM (relative), nDSM (above ground) or absolute DSM. Colour follows the honesty level. */
export function ProductBadge({ scene, size = 'sm', full = false }: { scene: Pick<Scene, 'product' | 'anchoring'>; size?: 'xs' | 'sm' | 'md'; full?: boolean }) {
  const p = productInfo(scene);
  return (
    <Tooltip label={p.summary} multiline maw={280} withArrow>
      <Badge size={size} color={COLOR[p.tone]} variant="light" data-product={scene.product} style={{ cursor: 'help' }}>
        {full ? p.short : scene.product}
      </Badge>
    </Tooltip>
  );
}
