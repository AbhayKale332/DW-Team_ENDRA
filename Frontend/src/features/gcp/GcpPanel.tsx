import { useMemo } from 'react';
import { ActionIcon, Alert, Badge, Button, Group, NumberInput, Table } from '@mantine/core';
import { IconCrosshair, IconX } from '@tabler/icons-react';
import type { Scene } from '@/domain/types';
import { KeyValueRows, PanelSection } from '@/components/panel';
import { fitGeoref, fitGround } from '@/lib/gcp';
import { useScene } from '@/store/scene';
import { runPrediction } from '@/features/processing/runPrediction';
import { applyGcps, clearGcps, gcpsReady, removeGcp, updateGcp, useGcp } from './gcpStore';

const HINT = 'Elevation (surface, m) on 1+ point makes the heights absolute: 1–2 points level the ground, 3+ fit a tilted plane. Lat/lon (WGS84) on 3+ points georeferences the image.';

function Num({ value, onChange, min, max, label }: { value: number | null; onChange: (v: number | null) => void; min?: number; max?: number; label: string }) {
  return (
    <NumberInput
      size="xs"
      hideControls
      decimalScale={6}
      min={min}
      max={max}
      value={value ?? ''}
      onChange={(v) => onChange(typeof v === 'number' && Number.isFinite(v) ? v : null)}
      aria-label={label}
      styles={{ input: { paddingInline: 6 } }}
    />
  );
}

/** Ground control points for scenes without their own georeferencing: elevations make the rDSM an absolute DSM,
 *  three lat/lon points georeference it. */
export function GcpPanel({ scene }: { scene: Scene }) {
  const points = useGcp((s) => s.points);
  const picking = useGcp((s) => s.picking);
  const hasInput = useScene((s) => !!s.input);
  const applied = useMemo(() => scene.gcps ?? [], [scene.gcps]);
  const nd = scene.ndsm ?? scene.heights;
  const geo = useMemo(() => {
    try {
      return fitGeoref(applied);
    } catch {
      return null;
    }
  }, [applied]);
  const ground = useMemo(() => fitGround(applied, nd), [applied, nd]);
  // the model's heights follow the resolution it ran at; GCPs that imply another one call for a re-run
  const runGsd = scene.meta.scene?.gsd_m;
  const inputGsd = geo ? (geo.gsd * nd.width) / scene.imageWidth : null;
  const off = geo && runGsd ? Math.abs(geo.gsd / runGsd - 1) > 0.1 : false;

  return (
    <PanelSection title="Ground control points" hint={HINT} right={applied.length > 0 && <Badge size="xs" variant="light" color="teal">applied</Badge>}>
      {points.length > 0 && (
        <Table fz="xs" verticalSpacing={2} horizontalSpacing={4} withRowBorders={false}>
          <Table.Thead>
            <Table.Tr>
              <Table.Th w={22}>#</Table.Th>
              <Table.Th>Lat °</Table.Th>
              <Table.Th>Lon °</Table.Th>
              <Table.Th>Elev m</Table.Th>
              <Table.Th w={26} />
            </Table.Tr>
          </Table.Thead>
          <Table.Tbody>
            {points.map((p, i) => (
              <Table.Tr key={p.id}>
                <Table.Td className="dw-mono" title={`Pixel ${Math.round(p.col)}, ${Math.round(p.row)}`}>
                  {i + 1}
                </Table.Td>
                <Table.Td>
                  <Num label={`Point ${i + 1} latitude`} value={p.lat} min={-90} max={90} onChange={(lat) => updateGcp(p.id, { lat })} />
                </Table.Td>
                <Table.Td>
                  <Num label={`Point ${i + 1} longitude`} value={p.lon} min={-180} max={180} onChange={(lon) => updateGcp(p.id, { lon })} />
                </Table.Td>
                <Table.Td>
                  <Num label={`Point ${i + 1} elevation`} value={p.elev} onChange={(elev) => updateGcp(p.id, { elev })} />
                </Table.Td>
                <Table.Td>
                  <ActionIcon size="sm" variant="subtle" color="gray" onClick={() => removeGcp(p.id)} aria-label={`Remove point ${i + 1}`}>
                    <IconX size={14} />
                  </ActionIcon>
                </Table.Td>
              </Table.Tr>
            ))}
          </Table.Tbody>
        </Table>
      )}

      <Group gap={6}>
        <Button size="xs" variant={picking ? 'filled' : 'default'} leftSection={<IconCrosshair size={14} />} onClick={() => useGcp.setState({ picking: !picking })}>
          {picking ? 'Click the scene…' : 'Add point'}
        </Button>
        <Button size="xs" disabled={!gcpsReady(points)} onClick={() => void applyGcps()}>
          Apply
        </Button>
        {(applied.length > 0 || points.length > 0) && (
          <Button size="xs" variant="subtle" color="gray" onClick={clearGcps}>
            Remove all
          </Button>
        )}
      </Group>

      {applied.length > 0 && (
        <KeyValueRows
          rows={[
            ['Heights', ground ? `absolute · ${ground.kind === 'plane' ? 'tilted' : 'level'} ground · ±${ground.rmsM.toFixed(2)} m` : 'relative'],
            ['Georeference', geo ? `EPSG:${geo.georef.epsg} · ±${geo.rmsM.toFixed(1)} m` : '—'],
            ...(geo ? ([['Resolution', `${geo.gsd.toFixed(3)} m/px`]] as Array<[string, string]>) : []),
          ]}
        />
      )}
      {off && inputGsd && (
        <Alert color="orange" variant="light" p="xs" fz="xs">
          <Group justify="space-between" gap={6} wrap="nowrap">
            <span>
              Model ran at {runGsd!.toFixed(2)} m/px, points give {geo!.gsd.toFixed(2)} m/px.
            </span>
            {hasInput && (
              <Button
                size="compact-xs"
                variant="light"
                color="orange"
                onClick={() => {
                  useScene.getState().setParams({ gsdMode: 'custom', gsd: Math.round(inputGsd * 1000) / 1000 });
                  void runPrediction();
                }}
              >
                Re-run
              </Button>
            )}
          </Group>
        </Alert>
      )}
    </PanelSection>
  );
}
