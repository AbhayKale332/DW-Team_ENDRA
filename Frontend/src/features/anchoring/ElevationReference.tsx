import { useState } from 'react';
import { Alert, Button, FileButton, Group, List, Loader, SegmentedControl, Select, Slider, Stack, Switch, Table, Text } from '@mantine/core';
import { IconAlertTriangle, IconInfoCircle } from '@tabler/icons-react';
import type { Scene, VerticalDatum } from '@/domain/types';
import { ProductBadge } from '@/components/ProductBadge';
import { anchorBlocker, refuseScene, withHeightReference } from '@/lib/dem';
import { productInfo } from '@/lib/product';
import { PRODUCT_LABELS } from '@/lib/sceneBuilder';
import { useAnchor } from '@/store/anchor';
import { useScene } from '@/store/scene';
import { useSettings } from '@/store/settings';
import { runAnchoring } from './runAnchoring';

const DATUMS: Array<{ value: VerticalDatum; label: string }> = [
  { value: 'EGM96', label: 'EGM96 (SRTM, NASADEM)' },
  { value: 'EGM2008', label: 'EGM2008 (Copernicus GLO-30)' },
  { value: 'ellipsoid', label: 'WGS84 ellipsoid (CartoDEM)' },
  { value: 'unknown', label: 'Not stated' },
];

/** What anchoring assumes and where it stops being valid. Shown next to the result, not buried in docs. */
export const ANCHOR_LIMITS = [
  'The DEM is coarse (~30 m). It fixes the terrain and the mean level of every 30 m cell; everything finer comes from the model.',
  'SRTM, Copernicus and Terrain Tiles are surface models. Their cells already hold part of each building, so only the model’s deviation from the cell mean is added on top (no double counting).',
  'Agreement with the DEM at 30 m is true by construction. It is not an independent accuracy result.',
  'No geoid conversion is applied. The datum is reported as published; EGM96 and EGM2008 differ by up to ~5 m in India, and an ellipsoidal DEM is tens of metres away from both.',
  'DEM vertical error (SRTM roughly ±5–10 m over relief) and the model’s own height error pass straight into the result.',
  'The DEM may have been acquired years apart from the image.',
];

function Rows({ rows }: { rows: Array<[string, React.ReactNode]> }) {
  return (
    <Table withRowBorders={false} verticalSpacing={3} horizontalSpacing={0} fz="xs">
      <Table.Tbody>
        {rows.map(([k, v]) => (
          <Table.Tr key={k}>
            <Table.Td c="dimmed" w="42%" valign="top">
              {k}
            </Table.Td>
            <Table.Td className="dw-mono" ta="right">
              {v}
            </Table.Td>
          </Table.Tr>
        ))}
      </Table.Tbody>
    </Table>
  );
}

/** Product, DEM anchoring status and controls, and the anchoring report. */
export function ElevationReference({ scene }: { scene: Scene }) {
  const anchor = useAnchor();
  const auto = useSettings((s) => s.autoAnchor);
  const [datum, setDatum] = useState<VerticalDatum>('EGM96');
  const [shareDraft, setShare] = useState<number | null>(null);
  const blocker = anchorBlocker(scene);
  const a = scene.anchoring;
  const mine = anchor.sceneId === scene.id;
  const running = mine && anchor.status === 'running';
  const info = productInfo(scene);
  const share = shareDraft ?? a?.structureShare ?? 0;

  return (
    <Stack gap="sm">
      <Group justify="space-between">
        <span className="dw-section-title">Product</span>
        <ProductBadge scene={scene} full />
      </Group>
      <Text size="xs">{PRODUCT_LABELS[scene.product].long}.</Text>
      <Text size="xs" c="dimmed">
        {info.summary}
      </Text>

      {a && (
        <>
          <SegmentedControl
            size="xs"
            fullWidth
            value={scene.product === 'DSM' ? 'dsm' : 'ndsm'}
            onChange={(v) => useScene.getState().updateScene(withHeightReference(scene, v as 'dsm' | 'ndsm'))}
            data={[
              { value: 'dsm', label: 'Elevation (DSM)' },
              { value: 'ndsm', label: 'Above ground (nDSM)' },
            ]}
            aria-label="Height reference"
          />
          <Rows
            rows={[
              ['DEM', a.source],
              ['Vertical datum', a.datum === 'unknown' ? 'not stated' : a.datum],
              ['DEM range', `${a.demMinM.toFixed(1)} – ${a.demMaxM.toFixed(1)} m`],
              ['Anchor cell', `${a.cellM.toFixed(0)} m (${a.cellPx} px)`],
              ['DSM vs DEM cells', `${a.meanOffsetM >= 0 ? '+' : ''}${a.meanOffsetM.toFixed(2)} m mean · ${a.cellMeanRmseM.toFixed(2)} m RMS`],
              ['Fetched', new Date(a.fetchedAt).toLocaleString()],
            ]}
          />
          <div>
            <Group justify="space-between">
              <Text size="xs" fw={500}>
                Structure already in the DEM
              </Text>
              <Text size="xs" className="dw-mono">
                {Math.round(share * 100)} %
              </Text>
            </Group>
            <Slider size="sm" min={0} max={1} step={0.05} value={share} onChange={setShare} onChangeEnd={(v) => useScene.getState().updateScene(refuseScene(scene, v))} label={(v) => `${Math.round(v * 100)} %`} aria-label="Share of structure height already in the DEM" />
            <Text size="xs" c="dimmed" mt={4} lh={1.5}>
              0 %: the DEM is bare terrain and the model’s heights sit on top (smooth ground, nothing sinks). 100 %: the DEM is a full surface model, so every 30 m cell is matched to it, which digs the terrain under dense buildings when the DEM does not really see them. SRTM at 30 m is somewhere in between.
            </Text>
          </div>
          {a.notes.map((n) => (
            <Text key={n} size="xs" c="dimmed">
              {n}
            </Text>
          ))}
        </>
      )}

      {!a && blocker && (
        <Alert color="orange" variant="light" icon={<IconInfoCircle size={16} />} p="xs">
          <Text size="xs">
            <b>Not an absolute DSM.</b> {blocker}
          </Text>
        </Alert>
      )}
      {!a && !blocker && running && (
        <Group gap={6}>
          <Loader size={14} />
          <Text size="xs">{anchor.message ?? 'Anchoring'}…</Text>
        </Group>
      )}
      {!a && !blocker && !running && mine && anchor.status === 'error' && (
        <Alert color="orange" variant="light" icon={<IconAlertTriangle size={16} />} p="xs">
          <Text size="xs">
            <b>Not anchored.</b> {anchor.message} The result stays a height-above-ground model (nDSM).
          </Text>
        </Alert>
      )}

      {!blocker && (
        <Stack gap={6}>
          <Group gap={6}>
            <Button size="xs" variant={a ? 'default' : 'filled'} loading={running} onClick={() => void runAnchoring({ kind: 'terrain-tiles' })}>
              {a ? 'Re-anchor (Terrain Tiles)' : 'Anchor to DEM (Terrain Tiles)'}
            </Button>
            <FileButton accept=".tif,.tiff" onChange={(f) => f && void runAnchoring({ kind: 'local-file', file: f, datum })}>
              {(props) => (
                <Button size="xs" variant="default" {...props}>
                  Load DEM file…
                </Button>
              )}
            </FileButton>
          </Group>
          <Select size="xs" label="Datum of a loaded DEM file" data={DATUMS} value={datum} onChange={(v) => v && setDatum(v as VerticalDatum)} allowDeselect={false} />
          <Switch size="xs" label="Anchor georeferenced results automatically" description="Sends the scene’s bounding box to the elevation tile server." checked={auto} onChange={(e) => useSettings.getState().set({ autoAnchor: e.currentTarget.checked })} />
        </Stack>
      )}

      <div>
        <span className="dw-section-title">Assumptions and limits</span>
        <List size="xs" mt={4} spacing={3}>
          {(scene.product === 'rDSM' ? ['Heights scale with the stated ground resolution; a wrong resolution rescales every height.', 'Without a coordinate system there is no way to look up terrain, so no absolute elevation is claimed.'] : ANCHOR_LIMITS).map((t) => (
            <List.Item key={t}>{t}</List.Item>
          ))}
        </List>
      </div>
    </Stack>
  );
}
