import { useEffect } from 'react';
import { ActionIcon, Alert, Badge, Button, Group, NumberInput, SegmentedControl, Select, Stack, Switch, Text } from '@mantine/core';
import { IconCrosshair, IconPlus, IconTrash } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useUseCases } from '@/store/usecases';
import { CLASS_LABELS, ENV_MODEL, FREQ_PRESETS, type Environment } from '@/lib/usecases/telecom';
import { addTower, refreshCoverage, removeTower, suggestTowers, updateTower } from './actions';

export const COVERAGE_COLORS = ['#dc2626', '#f59e0b', '#fbd33b', '#33b85a'] as const;

const pct = (v: number) => `${(v * 100).toFixed(v > 0 && v < 0.1 ? 1 : 0)} %`;
const area = (m2: number) => (m2 >= 1e6 ? `${(m2 / 1e6).toFixed(2)} km²` : m2 >= 1e4 ? `${(m2 / 1e4).toFixed(1)} ha` : `${m2.toFixed(0)} m²`);

/** Use case 1: telecom tower network coverage, from the DSM. */
export function TelecomPanel() {
  const scene = useScene((s) => s.scene);
  const u = useUseCases();
  const { towers, params, coverage, placing } = u;

  // any change to the radio parameters recomputes coverage
  useEffect(() => {
    if (useUseCases.getState().towers.length) refreshCoverage();
  }, [params]);

  if (!scene) return null;
  const s = coverage?.summary;
  return (
    <Stack gap="md">
      <Text size="xs" c="dimmed" lh={1.5}>
        Place a tower on the surface model. The signal is traced over the DSM, so buildings
        {scene.terrain ? ' and terrain' : ''} block it and the shadows show where extra infrastructure is needed.
      </Text>
      {!scene.terrain && (
        <Alert color="blue" variant="light" p="xs">
          <Text size="xs">
            {scene.product === 'rDSM' ? 'Relative heights: ' : 'No terrain elevation yet: '}ground is assumed flat, so only the model’s buildings and trees block the signal. Anchor to a DEM (Info tab) to add terrain shadowing.
          </Text>
        </Alert>
      )}

      <Group gap={6}>
        <Button size="xs" variant={placing ? 'filled' : 'default'} leftSection={<IconCrosshair size={14} />} onClick={() => u.set({ placing: !placing })}>
          {placing ? 'Click the terrain…' : 'Place tower'}
        </Button>
        <Button size="xs" variant="default" disabled={!towers.length} onClick={() => towers.forEach((t) => removeTower(t.id))}>
          Clear
        </Button>
        <Button
          size="xs"
          variant="default"
          onClick={() => addTower((scene.heights.width - 1) / 2, (scene.heights.height - 1) / 2)}
          disabled={!!towers.length}
          title="Add a tower at the centre of the scene"
        >
          Centre
        </Button>
      </Group>

      {towers.map((t, i) => (
        <Stack key={t.id} gap={6} p="xs" style={{ border: `1px solid ${t.id === u.selectedTower ? 'var(--mantine-color-yellow-6)' : 'var(--mantine-color-default-border)'}`, borderRadius: 2 }} onClick={() => u.set({ selectedTower: t.id })}>
          <Group justify="space-between">
            <Text size="sm" fw={600}>
              Tower {i + 1}
            </Text>
            <ActionIcon size="sm" variant="subtle" color="red" aria-label={`Remove tower ${i + 1}`} onClick={() => removeTower(t.id)}>
              <IconTrash size={14} />
            </ActionIcon>
          </Group>
          <Group grow gap={6}>
            <NumberInput size="xs" label="Height (m)" min={3} max={150} step={1} value={t.heightM} onChange={(v) => typeof v === 'number' && updateTower(t.id, { heightM: v })} />
            <NumberInput size="xs" label="EIRP (dBm)" min={20} max={70} step={1} value={t.eirpDbm} onChange={(v) => typeof v === 'number' && updateTower(t.id, { eirpDbm: v })} />
          </Group>
        </Stack>
      ))}

      <Stack gap={8}>
        <Text size="sm" fw={500}>
          Radio
        </Text>
        <SegmentedControl size="xs" fullWidth value={String(params.freqMHz)} onChange={(v) => u.set({ params: { ...params, freqMHz: Number(v) } })} data={FREQ_PRESETS.map((f) => ({ value: String(f), label: `${f}` }))} aria-label="Frequency, MHz" />
        <Text size="xs" c="dimmed">
          Frequency, MHz
        </Text>
        <Group grow gap={6} align="flex-end">
          <Select size="xs" label="Environment" allowDeselect={false} value={params.env} onChange={(v) => v && u.set({ params: { ...params, env: v as Environment } })} data={(Object.keys(ENV_MODEL) as Environment[]).map((e) => ({ value: e, label: ENV_MODEL[e].label }))} />
          <NumberInput size="xs" label="Receiver height (m)" min={0} max={30} step={0.5} value={params.rxHeightM} onChange={(v) => typeof v === 'number' && u.set({ params: { ...params, rxHeightM: v } })} />
        </Group>
        <Switch size="xs" label="Show coverage on the terrain" checked={u.telecomOverlay} onChange={(e) => u.set({ telecomOverlay: e.currentTarget.checked })} />
      </Stack>

      {u.coverageStatus === 'running' && (
        <Text size="xs" c="dimmed">
          Tracing radio paths…
        </Text>
      )}

      {s && (
        <Stack gap={8}>
          <Text size="sm" fw={500}>
            Coverage
          </Text>
          <div style={{ display: 'flex', height: 14, borderRadius: 2, overflow: 'hidden' }} role="img" aria-label={`Coverage: ${CLASS_LABELS.map((l, k) => `${l} ${pct(s.share[k])}`).join(', ')}`}>
            {[3, 2, 1, 0].map((k) => (
              <div key={k} style={{ width: `${s.share[k] * 100}%`, background: COVERAGE_COLORS[k] }} title={`${CLASS_LABELS[k]} ${pct(s.share[k])}`} />
            ))}
          </div>
          <Group gap={10} wrap="wrap">
            {[3, 2, 1, 0].map((k) => (
              <Group key={k} gap={4} wrap="nowrap">
                <span style={{ width: 9, height: 9, background: COVERAGE_COLORS[k], display: 'inline-block' }} />
                <Text size="xs">
                  {CLASS_LABELS[k]} {pct(s.share[k])}
                </Text>
              </Group>
            ))}
          </Group>
          <Text size="xs">
            <b>No line of sight: {pct(s.blockedStructures + s.blockedTerrain)}</b> of the area (hatched) — {pct(s.blockedStructures)} behind buildings/trees{scene.terrain ? `, ${pct(s.blockedTerrain)} behind terrain` : ''}. Poorly served but visible (distance-limited): {pct(s.distanceLimited)}.
          </Text>
          {s.deadZones.length > 0 && (
            <Text size="xs" c="dimmed">
              Largest no-service patches: {s.deadZones.map((z) => area(z.areaM2)).join(' · ')}
            </Text>
          )}
          <Group gap={6}>
            <Button size="xs" variant="default" leftSection={<IconPlus size={14} />} loading={u.suggesting} onClick={() => void suggestTowers()}>
              Suggest additional towers
            </Button>
          </Group>
        </Stack>
      )}

      {u.suggestions.length > 0 && (
        <Stack gap={4}>
          <Text size="xs" c="dimmed">
            Ghost markers on the map. Gain = extra share of the scene reaching at least “fair” service.
          </Text>
          {u.suggestions.map((sg, i) => (
            <Group key={i} justify="space-between" wrap="nowrap">
              <Group gap={6}>
                <Badge size="sm" color="cyan" variant="light">
                  Site {i + 1}
                </Badge>
                <Text size="xs">+{pct(sg.gain)} coverage</Text>
              </Group>
              <Button size="compact-xs" variant="default" onClick={() => addTower(sg.col, sg.row)}>
                Add tower
              </Button>
            </Group>
          ))}
        </Stack>
      )}
      {towers.length > 0 && u.coverageStatus === 'done' && u.suggestions.length === 0 && !u.suggesting && s && s.share[0] + s.share[1] < 0.005 && (
        <Text size="xs" c="dimmed">
          The scene is well covered.
        </Text>
      )}

      <Text size="xs" c="dimmed" lh={1.5}>
        Planning-grade estimate: free-space loss with an environment exponent, ITU-R P.526 knife-edge diffraction over the model’s surface, extra loss without line of sight and inside buildings. No antenna pattern, foliage or reflections. Not a substitute for a drive test.
      </Text>
    </Stack>
  );
}
