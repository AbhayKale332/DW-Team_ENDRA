import { useEffect, useState } from 'react';
import { Anchor, Button, Group, Loader, Stack, Text } from '@mantine/core';
import { IconExternalLink, IconMapPin } from '@tabler/icons-react';
import type { Scene } from '@/domain/types';
import { KeyValueRows, PanelSection } from '@/components/panel';
import { formatLonLat } from '@/lib/georef';
import { fetchOverpass, sceneBBox } from '@/lib/osm';
import { parseTileLocation, tileLocationQuery } from '@/lib/tileLocation';
import { useUi } from '@/store/ui';

/** Location lookup is limited to the visible Info tab; map links never depend on that service. */
export function TileLocation({ scene, centre }: { scene: Scene; centre: [number, number] | null }) {
  const active = useUi((s) => s.inspectorTab === 'info');
  const [place, setPlace] = useState<{ name: string | null; failed: boolean } | null>(null);
  const [attempt, setAttempt] = useState(0);
  const lon = centre?.[0];
  const lat = centre?.[1];

  useEffect(() => {
    if (!active || lon === undefined || lat === undefined) return;
    const controller = new AbortController();
    let cancelled = false;
    const timeout = setTimeout(() => controller.abort(), 20_000);
    void fetchOverpass(tileLocationQuery([lon, lat]), controller.signal)
      .then((json) => { if (!cancelled) setPlace({ name: parseTileLocation(json), failed: false }); })
      .catch(() => { if (!cancelled) setPlace({ name: null, failed: true }); })
      .finally(() => clearTimeout(timeout));
    return () => {
      cancelled = true;
      clearTimeout(timeout);
      controller.abort();
    };
  }, [active, lon, lat, attempt]);

  const bounds = centre && scene.georef ? sceneBBox(scene.georef, scene.heights.width, scene.heights.height) : null;

  return (
    <PanelSection title="General location" gap="xs" hint="The general area is resolved from OpenStreetMap boundaries at the tile centre. The Google Maps pin marks the centre, not the whole tile.">
      {centre ? (
        <>
          <Group gap={6} wrap="nowrap" align="flex-start" role="status">
            <IconMapPin size={16} color="var(--dw-dim)" aria-hidden style={{ flexShrink: 0, marginTop: 2 }} />
            <Stack gap={3} style={{ minWidth: 0 }}>
              {place?.name ? (
                <Text size="sm" fw={600} style={{ overflowWrap: 'anywhere' }}>{place.name}</Text>
              ) : !place ? (
                <Group gap={6}><Loader size={12} /><Text size="xs" c="dimmed">Finding the general area…</Text></Group>
              ) : (
                <Text size="xs" c="dimmed">{place.failed ? 'Area lookup unavailable. The map pin is still available.' : 'No named administrative area found at this tile centre.'}</Text>
              )}
              {place?.name && (
                <Text size="xs" c="dimmed">Tile centre · <Anchor fz="xs" href="https://www.openstreetmap.org/copyright" target="_blank" rel="noopener noreferrer">© OpenStreetMap contributors</Anchor></Text>
              )}
            </Stack>
          </Group>
          <KeyValueRows rows={[
            ['Latitude', `${Math.abs(centre[1]).toFixed(6)}° ${centre[1] >= 0 ? 'N' : 'S'}`],
            ['Longitude', `${Math.abs(centre[0]).toFixed(6)}° ${centre[0] >= 0 ? 'E' : 'W'}`],
          ]} />
          <Button component="a" variant="default" size="xs" fullWidth href={`https://www.google.com/maps/search/?api=1&query=${encodeURIComponent(`${centre[1].toFixed(6)},${centre[0].toFixed(6)}`)}`} target="_blank" rel="noopener noreferrer" rightSection={<IconExternalLink size={14} />}>
            Open tile centre in Google Maps
          </Button>
          {place?.failed && <Anchor component="button" type="button" fz="xs" ta="left" onClick={() => { setPlace(null); setAttempt((n) => n + 1); }}>Retry area lookup</Anchor>}
          {bounds && (
            <Text size="xs" c="dimmed" style={{ overflowWrap: 'anywhere' }}>
              Tile bounds (WGS84)<br />
              SW {formatLonLat([bounds[1], bounds[0]])}<br />
              NE {formatLonLat([bounds[3], bounds[2]])}
            </Text>
          )}
        </>
      ) : (
        <Text size="xs" c="dimmed">
          {scene.georef ? 'The tile’s coordinate system could not be converted to a map location.' : 'This tile has no geographic coordinates. Load a georeferenced GeoTIFF or add ground control points to locate it on a map.'}
        </Text>
      )}
    </PanelSection>
  );
}
