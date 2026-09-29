import { useCallback, useEffect, useMemo, useState } from 'react';
import { Alert, Button, Checkbox, FileButton, Group, List, SegmentedControl, Slider, Stack, Switch, Table, Text } from '@mantine/core';
import { IconFileReport, IconInfoCircle, IconUpload, IconX } from '@tabler/icons-react';
import { notifications } from '@mantine/notifications';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import type { ReferenceKind } from '@/domain/types';
import { loadReference } from '@/lib/reference';
import { validate } from '@/lib/metrics';
import { baseAxis, EChart, type ChartTheme } from '@/components/EChart';
import { reportError } from '@/features/files/openFile';
import { exportValidationReport } from './report';

const f2 = (v: number, unit = ' m') => (Number.isFinite(v) ? `${v.toFixed(2)}${unit}` : '—');
const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)} %` : '—');

function ReferenceLoader() {
  const scene = useScene((s) => s.scene)!;
  const [kind, setKind] = useState<ReferenceKind>(scene.product === 'DSM' ? 'DSM' : 'nDSM');
  const [busy, setBusy] = useState(false);
  return (
    <Stack gap="xs">
      <Text size="xs" c="dimmed">
        Compare the estimate with LiDAR, stereo or any reference height raster. GeoTIFFs are aligned by their georeferencing when the input was georeferenced.
      </Text>
      <SegmentedControl
        fullWidth
        value={kind}
        onChange={(k) => setKind(k as ReferenceKind)}
        data={[
          { value: 'nDSM', label: 'Height above ground (nDSM)' },
          { value: 'DSM', label: 'Elevation (DSM)' },
        ]}
        aria-label="Reference type"
      />
      <FileButton
        accept=".tif,.tiff,.npy"
        onChange={async (file) => {
          if (!file) return;
          setBusy(true);
          try {
            const ref = await loadReference(file, kind, scene);
            useScene.getState().setReference(ref);
            useScene.getState().set({ removeOffset: kind === 'DSM' && scene.product !== 'DSM' });
            notifications.show({ title: 'Reference aligned', message: ref.notes[0] ?? file.name, color: 'teal' });
          } catch (e) {
            reportError(e, 'Could not load reference');
          } finally {
            setBusy(false);
          }
        }}
      >
        {(props) => (
          <Button {...props} leftSection={<IconUpload size={16} />} loading={busy}>
            Load reference raster…
          </Button>
        )}
      </FileButton>
    </Stack>
  );
}

function Metrics() {
  const scene = useScene((s) => s.scene)!;
  const reference = useScene((s) => s.reference)!;
  const removeOffset = useScene((s) => s.removeOffset);
  const validation = useScene((s) => s.validation);

  useEffect(() => {
    const t = setTimeout(() => useScene.getState().setValidation(validate(scene.heights.data, reference.data, { removeOffset })), 0);
    return () => clearTimeout(t);
  }, [scene, reference, removeOffset]);

  const scatter = useCallback(
    (t: ChartTheme) => {
      if (!validation) return {};
      const { bins, lo, hi, counts } = validation.scatter;
      const w = (hi - lo) / bins;
      const data: Array<[number, number, number]> = [];
      let max = 1;
      for (let y = 0; y < bins; y++)
        for (let x = 0; x < bins; x++) {
          const c = counts[y * bins + x];
          if (c) {
            data.push([x, y, Math.log10(c + 1)]);
            max = Math.max(max, Math.log10(c + 1));
          }
        }
      const labels = Array.from({ length: bins }, (_, i) => (lo + (i + 0.5) * w).toFixed(1));
      return {
        grid: { left: 46, right: 12, top: 10, bottom: 40 },
        tooltip: { formatter: (p: { value: [number, number, number] }) => `ref ${labels[p.value[0]]} m · pred ${labels[p.value[1]]} m<br/>${Math.round(10 ** p.value[2] - 1)} px` },
        xAxis: { type: 'category', data: labels, ...baseAxis(t, 'Reference (m)'), splitLine: { show: false }, axisLabel: { ...baseAxis(t, '').axisLabel, interval: Math.floor(bins / 4) } },
        yAxis: { type: 'category', data: labels, ...baseAxis(t, 'Prediction (m)'), nameGap: 34, splitLine: { show: false }, axisLabel: { ...baseAxis(t, '').axisLabel, interval: Math.floor(bins / 4) } },
        visualMap: { show: false, min: 0, max, inRange: { color: ['#dbe7fb', '#2d6cdf', '#1b2f6b'] } },
        series: [
          {
            type: 'heatmap',
            data,
            progressive: 0,
            markLine: { silent: true, symbol: 'none', lineStyle: { color: '#e5793a', type: 'dashed' }, data: [[{ coord: [0, 0] }, { coord: [bins - 1, bins - 1] }]] },
          },
        ],
      };
    },
    [validation],
  );

  const errHist = useCallback(
    (t: ChartTheme) => {
      if (!validation) return {};
      const { lo, hi, counts } = validation.errorHist;
      const w = (hi - lo) / counts.length;
      return {
        grid: { left: 46, right: 12, top: 10, bottom: 40 },
        tooltip: { trigger: 'axis' },
        xAxis: { type: 'value', ...baseAxis(t, 'Prediction − reference (m)'), min: lo, max: hi },
        yAxis: { type: 'value', ...baseAxis(t, 'Pixels'), nameGap: 36 },
        series: [{ type: 'bar', barCategoryGap: '0%', data: Array.from(counts, (c, i) => [lo + (i + 0.5) * w, c]), itemStyle: { color: '#6a4cf0' } }],
      };
    },
    [validation],
  );

  const strata = useCallback(
    (t: ChartTheme) => {
      if (!validation) return {};
      const s = validation.strata.filter((x) => x.n > 0);
      return {
        grid: { left: 46, right: 12, top: 10, bottom: 40 },
        tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${Number(v).toFixed(2)} m` },
        xAxis: { type: 'category', data: s.map((x) => x.label), ...baseAxis(t, 'Reference height stratum') },
        yAxis: { type: 'value', ...baseAxis(t, 'RMSE (m)'), nameGap: 34 },
        series: [{ type: 'bar', data: s.map((x) => x.rmse), itemStyle: { color: '#2d6cdf' }, barWidth: '55%' }],
      };
    },
    [validation],
  );

  if (!validation) return <Text size="xs">Computing…</Text>;
  const a = validation.all;
  return (
    <Stack gap="sm">
      <Table withTableBorder withColumnBorders fz="xs" verticalSpacing={4} aria-label="Validation metrics">
        <Table.Tbody>
          {(
            [
              ['RMSE', f2(a.rmse)],
              ['MAE', f2(a.mae)],
              ['Bias (mean error)', `${a.bias >= 0 ? '+' : ''}${f2(a.bias)}`],
              ['Pearson r', Number.isFinite(a.r) ? a.r.toFixed(3) : '—'],
              ['Within ±1 m / ±2 m', `${pct(a.within1m)} / ${pct(a.within2m)}`],
              ['Balanced RMSE (strata mean)', f2(validation.balancedRmse)],
              ['Pixels compared', a.n.toLocaleString()],
            ] as Array<[string, string]>
          ).map(([k, v]) => (
            <Table.Tr key={k}>
              <Table.Td>{k}</Table.Td>
              <Table.Td className="dw-mono" ta="right" fw={600}>
                {v}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
      {validation.biasRemoved && (
        <Text size="xs" c="dimmed">
          Ground offset removed: {validation.offset.toFixed(2)} m (median reference − prediction where the prediction is below 1 m).
        </Text>
      )}
      <span className="dw-section-title">Prediction vs reference (density)</span>
      <EChart option={scatter} height={240} ariaLabel={`Density scatter of predicted against reference heights, Pearson r ${a.r.toFixed(3)}`} />
      <span className="dw-section-title">Error distribution</span>
      <EChart option={errHist} height={160} ariaLabel={`Histogram of height errors, bias ${a.bias.toFixed(2)} metres`} />
      <span className="dw-section-title">RMSE by height stratum</span>
      <EChart option={strata} height={160} ariaLabel="RMSE per reference height stratum" />
      <Table fz="xs" verticalSpacing={2} striped aria-label="Per-stratum metrics">
        <Table.Thead>
          <Table.Tr>
            <Table.Th>Stratum</Table.Th>
            <Table.Th ta="right">N</Table.Th>
            <Table.Th ta="right">RMSE</Table.Th>
            <Table.Th ta="right">Bias</Table.Th>
          </Table.Tr>
        </Table.Thead>
        <Table.Tbody>
          {validation.strata.map((s) => (
            <Table.Tr key={s.label}>
              <Table.Td>{s.label}</Table.Td>
              <Table.Td ta="right" className="dw-mono">
                {s.n.toLocaleString()}
              </Table.Td>
              <Table.Td ta="right" className="dw-mono">
                {f2(s.rmse, '')}
              </Table.Td>
              <Table.Td ta="right" className="dw-mono">
                {f2(s.bias, '')}
              </Table.Td>
            </Table.Tr>
          ))}
        </Table.Tbody>
      </Table>
    </Stack>
  );
}

/** Validate estimated heights against a reference dataset (problem-statement deliverable). */
export function ValidationTab() {
  const scene = useScene((s) => s.scene);
  const reference = useScene((s) => s.reference);
  const removeOffset = useScene((s) => s.removeOffset);
  const compare = useView((s) => s.compareSwipe);
  const swipe = useView((s) => s.swipe);
  const layerIsError = useView((s) => s.layer3d === 'error');
  const notes = useMemo(() => reference?.notes ?? [], [reference]);
  if (!scene) return <Text size="sm" c="dimmed" p="md">Open a result to validate it.</Text>;
  return (
    <Stack gap="md" p="md">
      {!reference ? (
        <ReferenceLoader />
      ) : (
        <>
          <Group justify="space-between" wrap="nowrap">
            <div style={{ minWidth: 0 }}>
              <Text size="sm" fw={600} truncate="end">
                {reference.name}
              </Text>
              <Text size="xs" c="dimmed">
                {reference.kind} · {reference.alignment === 'georeferenced' ? 'aligned by coordinates' : 'same-extent alignment'}
              </Text>
            </div>
            <Button size="compact-xs" variant="subtle" color="gray" leftSection={<IconX size={12} />} onClick={() => useScene.getState().setReference(null)}>
              Remove
            </Button>
          </Group>
          {notes.length > 0 && (
            <Alert variant="light" color="gray" icon={<IconInfoCircle size={16} />} p="xs">
              <List size="xs" spacing={2}>
                {notes.map((n) => (
                  <List.Item key={n}>{n}</List.Item>
                ))}
              </List>
            </Alert>
          )}
          {reference.kind === 'DSM' && scene.product !== 'DSM' && (
            <Checkbox
              size="xs"
              checked={removeOffset}
              onChange={(e) => useScene.getState().set({ removeOffset: e.currentTarget.checked })}
              label="Remove ground offset (absolute DSM reference vs height-above-ground estimate)"
            />
          )}
          <Group gap="xs">
            <Button size="xs" variant="default" onClick={() => useView.getState().set({ layer3d: layerIsError ? 'tint' : 'error', layer2d: layerIsError ? 'height' : 'error' })}>
              {layerIsError ? 'Hide error layer' : 'Show error layer'}
            </Button>
            <Button size="xs" variant="default" leftSection={<IconFileReport size={14} />} onClick={() => void exportValidationReport()}>
              Export report
            </Button>
          </Group>
          <Switch label="Compare swipe (prediction | reference)" checked={compare} onChange={(e) => useView.getState().set({ compareSwipe: e.currentTarget.checked })} />
          {compare && <Slider min={0.02} max={0.98} step={0.01} value={swipe} onChange={(x) => useView.getState().set({ swipe: x })} label={null} aria-label="Swipe position" />}
          <Metrics />
        </>
      )}
    </Stack>
  );
}
