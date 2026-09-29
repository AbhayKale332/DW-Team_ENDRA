import { useMemo } from 'react';
import { IconBuilding, IconMountain, IconRipple, IconTree, IconMapPin } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useTool } from '@/store/tool';
import { useView } from '@/store/view';
import { describeAt, type HoverKind } from '@/lib/hoverInfo';
import { osmFeaturesFor, useOsm } from '@/features/osm/osmStore';
import classes from './overlays.module.css';

const ICONS: Record<HoverKind, typeof IconBuilding> = {
  building: IconBuilding,
  tree: IconTree,
  water: IconRipple,
  elevated: IconMountain,
  surface: IconMapPin,
};

/** Card size assumed when deciding which side of the pointer it goes, so it never runs off the canvas. */
const CARD_W = 280;
const CARD_H = 240;
const OFFSET = 16;

/** Small popup next to the pointer: what is under it and how tall it is (Tools → Hover details). */
export function HoverCard() {
  const scene = useScene((s) => s.scene);
  const hover = useTool((s) => s.hover);
  const at = useTool((s) => s.hoverScreen);
  const enabled = useView((s) => s.hoverInfo);
  const osmOn = useView((s) => s.osm);
  const osm = useOsm((s) => osmFeaturesFor(s, scene));

  const info = useMemo(
    () => (enabled && scene && hover ? describeAt(scene, hover.col, hover.row, osmOn ? osm : null) : null),
    [enabled, scene, hover, osmOn, osm],
  );
  if (!info || !at) return null;

  const viewport = document.getElementById('dw-viewport')?.getBoundingClientRect();
  const flipX = viewport ? at.x + OFFSET + CARD_W > viewport.width : false;
  const flipY = viewport ? at.y + OFFSET + CARD_H > viewport.height : false;
  const Icon = ICONS[info.kind];
  return (
    <div
      className={`dw-float ${classes.hoverCard}`}
      style={{
        left: flipX ? at.x - OFFSET : at.x + OFFSET,
        top: flipY ? at.y - OFFSET : at.y + OFFSET,
        transform: `translate(${flipX ? '-100%' : '0'}, ${flipY ? '-100%' : '0'})`,
      }}
      role="tooltip"
      aria-live="off"
    >
      <div className={classes.hoverHead}>
        <Icon size={16} stroke={1.8} aria-hidden />
        <span className={classes.hoverTitle}>{info.title}</span>
        <span className={classes.hoverHeight}>{info.height.toFixed(1)} m</span>
      </div>
      {info.subtitle && <div className={classes.hoverSub}>{info.subtitle}</div>}
      <dl className={classes.hoverRows}>
        {info.rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{v}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}
