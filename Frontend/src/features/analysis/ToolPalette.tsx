import { ActionIcon, Divider, Group, Tooltip } from '@mantine/core';
import { IconBuilding, IconBuildingHospital, IconMap2, IconRipple, IconTree, IconWorld } from '@tabler/icons-react';
import { canGeolocate } from '@/lib/osm';
import { useOsm } from '@/features/osm/osmStore';
import { usePoi } from '@/features/poi/poiStore';
import type { ObjectKind } from '@/domain/types';
import { useScene } from '@/store/scene';
import { OBJECT_KIND_LABELS, OBJECT_KINDS, toggleObjectKind, useView } from '@/store/view';

export const OBJECT_KIND_ICONS: Record<ObjectKind, typeof IconBuilding> = { buildings: IconBuilding, trees: IconTree, water: IconRipple };

/** One 3D-object class on/off. Disabled (with a reason) when the result has none of that class. */
function ObjectKindToggle({ kind }: { kind: ObjectKind }) {
  const count = useScene((s) => s.scene?.objects?.[kind].length ?? 0);
  const on = useView((s) => s.objectKinds[kind]) && count > 0;
  const label = OBJECT_KIND_LABELS[kind];
  const Icon = OBJECT_KIND_ICONS[kind];
  return (
    <Tooltip label={count === 0 ? `No ${label.toLowerCase()} detected in this result` : `3D ${label.toLowerCase()} (${count}): ${on ? 'on — click to show the raw mesh here' : 'off — click to show as 3D models'}`} position="bottom">
      {/* the wrapper keeps the tooltip working while the button is disabled */}
      <span>
        <ActionIcon
          size={34}
          variant={on ? 'filled' : 'subtle'}
          color={on ? 'teal' : 'gray'}
          onClick={() => toggleObjectKind(kind)}
          disabled={count === 0}
          aria-label={`3D ${label.toLowerCase()}`}
          aria-pressed={on}
        >
          <Icon size={18} stroke={1.7} />
        </ActionIcon>
      </span>
    </Tooltip>
  );
}

type GeoLayer = 'osm' | 'basemap' | 'poi';

/** A layer that needs a georeferenced scene, on/off. Disabled (with a reason) for images that are not. */
function GeoToggle({ layer, label, icon: Icon, offHint, loading }: { layer: GeoLayer; label: string; icon: typeof IconMap2; offHint: string; loading?: boolean }) {
  const georef = useScene((s) => s.scene?.georef);
  const on = useView((s) => s[layer]);
  const geo = canGeolocate(georef);
  return (
    <Tooltip
      label={!geo ? `${label} needs a georeferenced image (e.g. a GeoTIFF)` : on ? `${label}: on — click to hide` : `${label}: off — ${offHint}`}
      position="bottom"
      multiline
      w={260}
    >
      <span>
        <ActionIcon
          size={34}
          variant={on && geo ? 'filled' : 'subtle'}
          color={on && geo ? 'dwBlue' : 'gray'}
          onClick={() => useView.getState().set({ [layer]: !on })}
          disabled={!geo}
          loading={on && loading}
          aria-label={label}
          aria-pressed={on && geo}
        >
          <Icon size={18} stroke={1.7} />
        </ActionIcon>
      </span>
    </Tooltip>
  );
}

/** Compact toolbar under the view switcher: the per-class 3D object switches and the georeferenced layers. */
export function ToolPalette() {
  const hasObjects = useScene((s) => !!s.scene?.objects);
  const osmLoading = useOsm((s) => s.status === 'loading');
  const poiLoading = usePoi((s) => s.status === 'loading');
  return (
    <Group gap={0} className="dw-float" p={3} role="toolbar" aria-label="Layer toggles">
      {hasObjects && (
        <>
          {OBJECT_KINDS.map((k) => (
            <ObjectKindToggle key={k} kind={k} />
          ))}
          <Divider orientation="vertical" mx={3} my={4} />
        </>
      )}
      <GeoToggle layer="basemap" label="Surrounding basemap" icon={IconWorld} offHint="shows map tiles around the processed area (display only)" />
      <GeoToggle layer="poi" label="Facilities" icon={IconBuildingHospital} offHint="shows hospitals, fire stations, police, schools, shelters and stations from OpenStreetMap" loading={poiLoading} />
      <GeoToggle layer="osm" label="OpenStreetMap overlay" icon={IconMap2} offHint="loads roads, buildings and water for this area from OpenStreetMap" loading={osmLoading} />
    </Group>
  );
}
