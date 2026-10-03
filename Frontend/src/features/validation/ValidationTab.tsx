import { useCallback, useEffect, useMemo, useState } from 'react';
import { Accordion, Alert, Button, Checkbox, FileButton, Group, List, SegmentedControl, Slider, Stack, Switch, Table, Text } from '@mantine/core';
import { IconCheck, IconFileReport, IconInfoCircle, IconScale, IconUpload } from '@tabler/icons-react';
import { EmptyPanel, PanelSection } from '@/components/panel';
import { notifications } from '@mantine/notifications';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { useUi } from '@/store/ui';
import type { ReferenceKind } from '@/domain/types';
import { loadReference, referenceKinds } from '@/lib/reference';
import { withHeightReference } from '@/lib/dem';
import { validate } from '@/lib/metrics';
import { baseAxis, EChart, type ChartTheme } from '@/components/EChart';
import { reportError } from '@/features/files/openFile';
import { exportValidationReport } from './report';

const f2 = (v: number, unit = ' m') => (Number.isFinite(v) ? `${v.toFixed(2)}${unit}` : '—');
const pct = (v: number) => (Number.isFinite(v) ? `${(v * 100).toFixed(1)} %` : '—');

function ReferenceLoader() {
  const scene = useScene((s) => s.scene)!;
  const kinds = referenceKinds(scene);
  const [selectedKind, setKind] = useState<ReferenceKind>(kinds[0]);
  const kind = kinds.includes(selectedKind) ? selectedKind : kinds[0];
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  return (
    <Stack gap="lg">
      <PanelSection title="1 · Choose height type">
        {kinds.length > 1 ? (
          <SegmentedControl
            fullWidth
            value={kind}
            disabled={busy}
            onChange={(k) => {
              setKind(k as ReferenceKind);
              setError(null);
            }}
            data={kinds.map((value) => ({
              value,
              label: (
                <Stack gap={2} align="center">
                  <Text size="sm" fw={600}>
                    {value}
                  </Text>
                  <Text size="xs">{value === 'DSM' ? 'Surface elevation' : 'Above-ground height'}</Text>
                </Stack>
              ),
            }))}
            aria-label="Reference height type"
          />
        ) : (
          <Text size="sm" fw={600}>
            nDSM · Height above ground
          </Text>
        )}
        <Text size="xs" c="dimmed">
          {kind === 'DSM' ? 'Compare surface elevations, including the ground level, in metres.' : 'Compare building and tree heights above the ground, in metres.'}
          {kinds.length === 1 && ' PNG / JPG inputs use nDSM validation.'}
        </Text>
        {kind === 'DSM' && scene.product !== 'DSM' && !scene.anchoring && (
          <Alert variant="light" color="gray" icon={<IconInfoCircle size={15} />} p="sm">
            <Stack gap="xs">
              <Text size="xs">
                This result is above-ground height. Set a ground reference in Info to compare absolute elevations. Without it, only a constant ground offset can be removed.
              </Text>
              <Button size="compact-xs" variant="subtle" onClick={() => useUi.getState().set({ inspectorTab: 'info' })}>
                Set ground reference in Info
              </Button>
            </Stack>
          </Alert>
        )}
      </PanelSection>
      <PanelSection title="2 · Load ground truth">
        <Text size="xs" c="dimmed">
          Use a measured height file covering the same area as your input image.
        </Text>
        <FileButton
          accept=".tif,.tiff,.geotiff,.npy"
          onChange={async (file) => {
            if (!file) return;
            setBusy(true);
            setError(null);
            try {
              const ref = await loadReference(file, kind, scene);
              const current = useScene.getState().scene;
              if (!current || current.id !== scene.id) return;
              const alignedScene = withHeightReference(current, kind === 'DSM' ? 'dsm' : 'ndsm');
              if (alignedScene !== current) useScene.getState().updateScene(alignedScene);
              useScene.getState().setReference(ref);
              useScene.getState().set({
                removeOffset: kind === 'DSM' && alignedScene.product !== 'DSM',
              });
              notifications.show({
                title: 'Reference aligned',
                message: ref.notes[0] ?? file.name,
                color: 'teal',
              });
            } catch (e) {
              setError(e instanceof Error ? e.message : 'Could not read this reference. Try another height file.');
              reportError(e, 'Could not load reference');
            } finally {
              setBusy(false);
            }
          }}
        >
          {(props) => (
            <Button {...props} fullWidth leftSection={<IconUpload size={16} />} loading={busy}>
              {busy ? 'Aligning reference…' : 'Load reference file…'}
            </Button>
          )}
        </FileButton>
        <Text size="xs">
          <b>Supported files:</b> GeoTIFF (.tif, .tiff, .geotiff) or NumPy (.npy).
        </Text>
        <Text size="xs" c="dimmed">
          Values must be in metres. NumPy files must be 2D; match the image’s crop and orientation.
        </Text>
        {error && (
          <Alert color="red" variant="light" title="Reference could not be loaded" role="alert">
            <Text size="xs" style={{ overflowWrap: 'anywhere' }}>
              {error}
            </Text>
          </Alert>
        )}
      </PanelSection>
    </Stack>
  );
}

function Metrics() {
  const scene = useScene((s) => s.scene)!;
  const reference = useScene((s) => s.reference)!;
  const removeOffset = useScene((s) => s.removeOffset);
  const validation = useScene((s) => s.validation);

  useEffect(() => {
    const t = setTimeout(
      () =>
        useScene.getState().setValidation(
          validate(scene.heights.data, reference.data, {
            removeOffset,
            sigma: scene.uncertainty,
          }),
        ),
      0,
    );
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

  const sparsify = useCallback(
    (t: ChartTheme) => {
      const sp = validation?.uncertainty?.sparsification;
      if (!sp) return {};
      // the curves' forced 0 at 100 % removed is for the areas only: plot up to the last real step
      const pts = (ys: number[]) => sp.removed.slice(0, -1).map((x, i) => [x * 100, ys[i]]);
      return {
        grid: { left: 46, right: 12, top: 48, bottom: 40 },
        legend: { top: 0, textStyle: { color: t.dim, fontSize: 10 }, itemHeight: 8 },
        tooltip: { trigger: 'axis', valueFormatter: (v: number) => `${Number(v).toFixed(2)} m` },
        xAxis: { type: 'value', ...baseAxis(t, 'Pixels removed, most uncertain first (%)'), min: 0, max: 100 },
        yAxis: { type: 'value', ...baseAxis(t, 'RMSE of the rest (m)'), nameGap: 34, min: 0 },
        series: [
          { name: 'By model σ', type: 'line', showSymbol: false, data: pts(sp.bySigma), lineStyle: { width: 2, color: '#2d6cdf' }, itemStyle: { color: '#2d6cdf' } },
          { name: 'By true error (best)', type: 'line', showSymbol: false, data: pts(sp.oracle), lineStyle: { width: 1.5, type: 'dashed', color: '#12b886' }, itemStyle: { color: '#12b886' } },
          { name: 'Random', type: 'line', showSymbol: false, data: pts(sp.removed.map(() => sp.rmse)), lineStyle: { width: 1.5, type: 'dotted', color: t.dim }, itemStyle: { color: t.dim } },
        ],
      };
    },
    [validation],
  );

  if (!validation)
    return (
      <Text size="xs" role="status">
        Calculating accuracy…
      </Text>
    );
  const a = validation.all;
  if (!a.n)
    return (
      <Alert color="orange" title="No valid pixels to compare">
        Check that the reference overlaps this image and contains valid height values.
      </Alert>
    );
  const u = validation.uncertainty;
  const sp = u?.sparsification;
  return (
    <Stack gap="sm">
      <span className="dw-section-title">3 · Review accuracy</span>
      <Text size="xs" c="dimmed">
        Lower RMSE and MAE mean a closer match. Positive bias means the prediction is too high.
      </Text>
      <Table withTableBorder withColumnBorders fz="xs" verticalSpacing={4} aria-label="Validation metrics">
        <Table.Tbody>
          {(
            [
              ['RMSE · overall error', f2(a.rmse)],
              ['MAE · average error', f2(a.mae)],
              ['Bias (mean error)', `${a.bias >= 0 ? '+' : ''}${f2(a.bias)}`],
              ['Correlation (Pearson r)', Number.isFinite(a.r) ? a.r.toFixed(3) : '—'],
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
      <Accordion variant="default" keepMounted={false} transitionDuration={0}>
        <Accordion.Item value="details">
          <Accordion.Control>Charts & detailed metrics</Accordion.Control>
          <Accordion.Panel>
            <Stack gap="sm">
              {u ? (
                <>
                  <span className="dw-section-title">Model uncertainty vs error</span>
                  <Table withTableBorder withColumnBorders fz="xs" verticalSpacing={4} aria-label="Uncertainty metrics">
                    <Table.Tbody>
                      {(
                        [
                          [`RMSE on confident pixels (σ ≤ ${u.confidentM.toFixed(1)} m)`, f2(u.confident.rmse)],
                          ['Confident share of compared pixels', pct(u.coverage)],
                          ...(sp
                            ? ([
                                ['AUSE (0 = σ ranks errors perfectly)', f2(sp.ause)],
                                ['AURG (> 0 = better than random)', `${sp.aurg >= 0 ? '+' : ''}${f2(sp.aurg)}`],
                              ] as Array<[string, string]>)
                            : []),
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
                  {sp && (
                    <>
                      <EChart
                        option={sparsify}
                        height={200}
                        ariaLabel={`Sparsification curve: RMSE of the remaining pixels as the most uncertain are removed, AUSE ${sp.ause.toFixed(2)} metres`}
                      />
                      <Text size="xs" c="dimmed">
                        If σ tracks the error, removing the most uncertain pixels first lowers the RMSE of the rest, close to the best possible order (Poggi et al., CVPR 2020).
                        {sp.n < a.n ? ` Curves from ${sp.n.toLocaleString()} evenly spaced pixels.` : ''}
                      </Text>
                    </>
                  )}
                </>
              ) : (
                <Text size="xs" c="dimmed">
                  This result has no uncertainty map. Height accuracy metrics are still available above.
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
          </Accordion.Panel>
        </Accordion.Item>
      </Accordion>
    </Stack>
  );
}

/** Validate estimated heights against a reference dataset (problem-statement deliverable). */
export function ValidationTab() {
  const scene = useScene((s) => s.scene);
  const reference = useScene((s) => s.reference);
  const validation = useScene((s) => s.validation);
  const removeOffset = useScene((s) => s.removeOffset);
  const compare = useView((s) => s.compareSwipe);
  const swipe = useView((s) => s.swipe);
  const layerIsError = useView((s) => s.layer3d === 'error');
  const notes = useMemo(() => reference?.notes ?? [], [reference]);
  if (!scene) return <EmptyPanel icon={IconScale}>Run height estimation or open a result, then compare it with a ground-truth height file (.tif, .tiff, .geotiff or .npy).</EmptyPanel>;
  return (
    <Stack gap="md" p="md">
      <Stack gap={4}>
        <Text size="sm" fw={600}>
          Compare with ground truth
        </Text>
        <Text size="xs" c="dimmed">
          Check height accuracy, inspect errors, and export the results.
        </Text>
      </Stack>
      {!reference ? (
        <ReferenceLoader key={scene.id} />
      ) : (
        <>
          <Group justify="space-between" wrap="nowrap">
            <div style={{ minWidth: 0, flex: 1 }}>
              <Group gap={4} mb={4}>
                <IconCheck size={14} color="var(--mantine-color-teal-6)" aria-hidden />
                <Text size="xs" fw={600}>
                  Reference loaded
                </Text>
              </Group>
              <Text size="sm" fw={600} style={{ overflowWrap: 'anywhere' }}>
                {reference.name}
              </Text>
              <Text size="xs" c="dimmed">
                {reference.kind} · {reference.alignment === 'georeferenced' ? 'aligned by coordinates' : 'same-extent alignment'}
              </Text>
            </div>
            <Button
              size="compact-xs"
              variant="default"
              onClick={() => {
                useScene.getState().setReference(null);
                useView.getState().set({
                  compareSwipe: false,
                  ...(layerIsError ? { layer3d: 'optical', layer2d: 'height' } : {}),
                });
              }}
            >
              Change
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
            <Checkbox size="xs" checked={removeOffset} onChange={(e) => useScene.getState().set({ removeOffset: e.currentTarget.checked })} label="Remove ground offset" />
          )}
          <Metrics />
          <span className="dw-section-title">Inspect the comparison</span>
          <Group gap="xs">
            <Button
              size="xs"
              variant="default"
              onClick={() =>
                useView.getState().set({
                  layer3d: layerIsError ? 'optical' : 'error',
                  layer2d: layerIsError ? 'height' : 'error',
                })
              }
            >
              {layerIsError ? 'Hide error layer' : 'Show error layer'}
            </Button>
          </Group>
          <Switch label="Compare prediction and reference" checked={compare} onChange={(e) => useView.getState().set({ compareSwipe: e.currentTarget.checked })} />
          {compare && <Slider min={0.02} max={0.98} step={0.01} value={swipe} onChange={(x) => useView.getState().set({ swipe: x })} label={null} thumbLabel="Swipe position" />}
          <Button fullWidth leftSection={<IconFileReport size={16} />} disabled={!validation?.all.n} onClick={() => void exportValidationReport()}>
            Export report
          </Button>
        </>
      )}
    </Stack>
  );
}
