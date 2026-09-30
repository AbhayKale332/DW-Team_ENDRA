import { useMemo } from 'react';
import { ActionIcon, Alert, Badge, Button, Group, NumberInput, ScrollArea, Stack, Table, Text } from '@mantine/core';
import { IconCrosshair, IconX } from '@tabler/icons-react';
import { InfoHint, KeyValueRows } from '@/components/panel';
import { fitGeoref, fitGround } from '@/lib/gcp';
import { useScene } from '@/store/scene';
import { runPrediction } from '@/features/processing/runPrediction';
import { applyGcps, clearGcps, gcpEligible, gcpsReady, removeGcp, setGcpOpen, updateGcp, useGcp } from './gcpStore';
import classes from '@/features/viewport/overlays/overlays.module.css';

const HINT =
  'Pick a point on the image and enter its height in metres: 1–2 points level the ground, 3+ fit a tilted ground, and the scene becomes an absolute DSM. Lat/lon are optional; on 3+ points they also georeference the image.';

function Num({ value, onChange, label, placeholder, min, max, autoFocus }: { value: number | null; onChange: (v: number | null) => void; label: string; placeholder?: string; min?: number; max?: number; autoFocus?: boolean }) {
  return (
    <NumberInput
      size="xs"
      hideControls
      decimalScale={6}
      min={min}
      max={max}
      value={value ?? ''}
      placeholder={placeholder}
      autoFocus={autoFocus}
      onChange={(v) => onChange(typeof v === 'number' && Number.isFinite(v) ? v : null)}
      aria-label={label}
      styles={{ input: { paddingInline: 6 } }}
    />
  );
}

/** Ground control points card (header → GCP): heights at picked points turn the rDSM into an absolute DSM. */
export function GcpOverlay() {
  const scene = useScene((s) => s.scene);
  const open = useGcp((s) => s.open);
  const points = useGcp((s) => s.points);
  const picking = useGcp((s) => s.picking);
  const lastAdded = useGcp((s) => s.lastAdded);
  const hasInput = useScene((s) => !!s.input);
  const gcps = scene?.gcps;
  const applied = useMemo(() => gcps ?? [], [gcps]);
  const nd = scene ? (scene.ndsm ?? scene.heights) : null;
  const geo = useMemo(() => {
    try {
      return fitGeoref(applied);
    } catch {
      return null;
    }
  }, [applied]);
  const ground = useMemo(() => (nd ? fitGround(applied, nd) : null), [applied, nd]);
  if (!open || !scene || !nd || !gcpEligible(scene)) return null;

  // the model's heights follow the resolution it ran at; points that imply another one call for a re-run
  const runGsd = scene.meta.scene?.gsd_m;
  const inputGsd = geo ? (geo.gsd * nd.width) / scene.imageWidth : null;
  const off = geo && runGsd ? Math.abs(geo.gsd / runGsd - 1) > 0.1 : false;

  return (
    <div className={`dw-float ${classes.scenarioCard}`} role="dialog" aria-label="Ground control points">
      <Group justify="space-between" wrap="nowrap" px="sm" pt="sm" pb="xs" style={{ borderBottom: '1px solid var(--dw-line)' }}>
        <Group gap={6} wrap="nowrap">
          <Text size="sm" fw={600}>
            Ground control points
          </Text>
          <InfoHint>{HINT}</InfoHint>
          {applied.length > 0 && (
            <Badge size="xs" variant="light" color="teal">
              applied
            </Badge>
          )}
        </Group>
        <ActionIcon size="sm" variant="subtle" color="gray" onClick={() => setGcpOpen(false)} aria-label="Close ground control points">
          <IconX size={15} />
        </ActionIcon>
      </Group>
      <ScrollArea type="auto" offsetScrollbars style={{ flex: '1 1 auto', minHeight: 0 }}>
        <Stack gap="sm" p="sm">
          {points.length > 0 && (
            <Table fz="xs" verticalSpacing={2} horizontalSpacing={4} withRowBorders={false}>
              <Table.Thead>
                <Table.Tr>
                  <Table.Th w={22}>#</Table.Th>
                  <Table.Th>Height m</Table.Th>
                  <Table.Th>Lat °</Table.Th>
                  <Table.Th>Lon °</Table.Th>
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
                      <Num label={`Point ${i + 1} height`} value={p.elev} autoFocus={p.id === lastAdded} onChange={(elev) => updateGcp(p.id, { elev })} />
                    </Table.Td>
                    <Table.Td>
                      <Num label={`Point ${i + 1} latitude`} placeholder="optional" value={p.lat} min={-90} max={90} onChange={(lat) => updateGcp(p.id, { lat })} />
                    </Table.Td>
                    <Table.Td>
                      <Num label={`Point ${i + 1} longitude`} placeholder="optional" value={p.lon} min={-180} max={180} onChange={(lon) => updateGcp(p.id, { lon })} />
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
              {picking ? 'Click the image…' : 'Add point'}
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
            <Alert color="orange" variant="light" p="xs">
              <Group justify="space-between" gap={6} wrap="nowrap">
                <Text size="xs">
                  Model ran at {runGsd!.toFixed(2)} m/px, points give {geo!.gsd.toFixed(2)} m/px.
                </Text>
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
        </Stack>
      </ScrollArea>
    </div>
  );
}
