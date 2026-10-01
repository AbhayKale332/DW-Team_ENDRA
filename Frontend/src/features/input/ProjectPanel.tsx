import { useEffect, useState } from 'react';
import { ActionIcon, Badge, Button, Divider, Group, Input, Loader, NumberInput, ScrollArea, SegmentedControl, Stack, Switch, Text, Tooltip } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import { IconChevronLeft, IconInfoCircle, IconPhotoUp, IconPlayerPlay, IconWorld, IconX } from '@tabler/icons-react';
import { ProductBadge } from '@/components/ProductBadge';
import { KeyValueRows, PanelSection } from '@/components/panel';
import { useScene } from '@/store/scene';
import { useUi } from '@/store/ui';
import { effectiveModelGrid, MODEL_MAX_SIDE } from '@/lib/input';
import { stageImage } from '@/features/files/openFile';
import { cancelPrediction, resolveGsd, runPrediction } from '@/features/processing/runPrediction';
import { cloudOverlay } from '@/lib/cloudImage';
import classes from '@/features/shell/shell.module.css';

const PRESETS = [
  { value: '0.3', label: '0.3' },
  { value: '0.5', label: '0.5' },
  { value: '0.6', label: '0.6' },
  { value: '1', label: '1.0' },
];

/** Object URL of the cyan cloud overlay for the staged input, while the cloud mask is on. */
function useCloudOverlayUrl() {
  const mask = useScene((s) => (s.params.cloudMask ? s.inputCloud?.mask : null));
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    if (!mask) return setUrl(null);
    let cancelled = false;
    let made: string | null = null;
    void cloudOverlay(mask).then((b) => {
      if (cancelled) return;
      made = URL.createObjectURL(b);
      setUrl(made);
    });
    return () => {
      cancelled = true;
      if (made) URL.revokeObjectURL(made);
    };
  }, [mask]);
  return url;
}

function InputCard() {
  const input = useScene((s) => s.input);
  const running = useScene((s) => s.run.status === 'running');
  const overlay = useCloudOverlayUrl();
  if (!input) {
    return (
      <Dropzone
        onDrop={(files) => files[0] && void stageImage(files[0])}
        accept={{ 'image/png': ['.png'], 'image/jpeg': ['.jpg', '.jpeg'], 'image/tiff': ['.tif', '.tiff'] }}
        multiple={false}
        maxSize={512 * 1024 * 1024}
        radius="xs"
        p="lg"
        aria-label="Drop an image or click to choose one"
      >
        <Stack align="center" gap={6} style={{ pointerEvents: 'none' }}>
          <IconPhotoUp size={34} stroke={1.3} color="var(--dw-faint)" />
          <Text size="sm" fw={500} ta="center">
            Drop an image here or click to browse
          </Text>
          <Text size="xs" c="dimmed" ta="center">
            PNG · JPG · GeoTIFF
          </Text>
        </Stack>
      </Dropzone>
    );
  }
  return (
    <Stack gap={8}>
      <div style={{ position: 'relative', border: '1px solid var(--dw-line)', background: 'var(--dw-surface)' }}>
        <img src={input.previewUrl} alt={`Preview of ${input.name}`} style={{ display: 'block', width: '100%', maxHeight: 180, objectFit: 'contain' }} />
        {overlay && (
          <img
            src={overlay}
            alt=""
            aria-hidden
            style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', maxHeight: 180, objectFit: 'contain', pointerEvents: 'none' }}
          />
        )}
        <Tooltip label="Remove image">
          <ActionIcon
            variant="filled"
            color="dark"
            size="sm"
            style={{ position: 'absolute', top: 6, right: 6 }}
            onClick={() => useScene.getState().setInput(null)}
            disabled={running}
            aria-label="Remove image"
          >
            <IconX size={14} />
          </ActionIcon>
        </Tooltip>
      </div>
      <Text size="sm" fw={600} truncate="end" title={input.name}>
        {input.name}
      </Text>
      <KeyValueRows
        rows={[
          ['Size', `${input.width} × ${input.height} px`],
          ['Bands / depth', `${input.bands} × ${input.bitsPerSample}-bit`],
          [
            'Georeferenced',
            input.georef ? (
              <Badge size="xs" color="teal" leftSection={<IconWorld size={10} />}>
                {input.georef.epsg ? `EPSG:${input.georef.epsg}` : 'yes'}
              </Badge>
            ) : (
              'no'
            ),
          ],
          ...(input.fileGsd ? ([['GSD from file', `${input.fileGsd.toFixed(3)} m/px`]] as Array<[string, string]>) : []),
        ]}
      />
    </Stack>
  );
}

function Parameters() {
  const input = useScene((s) => s.input);
  const params = useScene((s) => s.params);
  const running = useScene((s) => s.run.status === 'running');
  const setParams = useScene((s) => s.setParams);
  const mode = params.gsdMode === 'auto' ? 'auto' : PRESETS.some((p) => Number(p.value) === params.gsd) && params.gsdMode === 'preset' ? String(params.gsd) : 'custom';
  const { gsd, source } = resolveGsd();
  const eff = input ? effectiveModelGrid(input.width, input.height, gsd ?? 0.5) : null;

  return (
    <Stack gap="sm">
      <Input.Wrapper label="Resolution (m / pixel)" labelProps={{ id: 'gsd-label', mb: 6 }}>
        <SegmentedControl
          fullWidth
          aria-labelledby="gsd-label"
          disabled={running}
          value={mode}
          onChange={(v) => {
            if (v === 'auto') setParams({ gsdMode: 'auto' });
            else if (v === 'custom') setParams({ gsdMode: 'custom' });
            else setParams({ gsdMode: 'preset', gsd: Number(v) });
          }}
          data={[{ value: 'auto', label: 'Auto' }, ...PRESETS, { value: 'custom', label: 'Custom' }]}
        />
        {mode === 'custom' && (
          <NumberInput
            mt={6}
            aria-label="Custom resolution"
            min={0.05}
            max={30}
            step={0.05}
            decimalScale={3}
            value={params.gsd}
            onChange={(v) => typeof v === 'number' && setParams({ gsd: v, gsdMode: 'custom' })}
            disabled={running}
            suffix=" m/px"
          />
        )}
        <Text size="xs" mt={6} c={source === 'assumed' ? 'dwOrange.7' : 'dimmed'}>
          {source === 'geotiff' && `Using ${gsd?.toFixed(3)} m/px from the GeoTIFF.`}
          {source === 'user' && `Using ${gsd?.toFixed(3)} m/px as declared.`}
          {source === 'assumed' && 'Unknown — assuming 0.5 m/px.'}
        </Text>
        {eff?.resampled && (
          <Tooltip label={`The long side exceeds ${MODEL_MAX_SIDE} px; the 3D drape still uses the full-resolution image.`} multiline maw={260}>
            <Text size="xs" c="dimmed" mt={4}>
              Processed at {eff.width} × {eff.height} px (≈{eff.gsd.toFixed(2)} m/px)
            </Text>
          </Tooltip>
        )}
      </Input.Wrapper>
      <Switch
        checked={params.tta}
        onChange={(e) => setParams({ tta: e.currentTarget.checked })}
        disabled={running}
        label="Higher quality (TTA, slower)"
      />
      <CloudSwitch />
    </Stack>
  );
}

function CloudSwitch() {
  const input = useScene((s) => s.input);
  const cloud = useScene((s) => s.inputCloud);
  const on = useScene((s) => s.params.cloudMask);
  const running = useScene((s) => s.run.status === 'running');
  const pct = cloud?.mask ? cloud.mask.coverage * 100 : 0;
  const description = !input
    ? 'Masks clouds out of the image before the model sees it.'
    : cloud?.status === 'detecting'
      ? 'Looking for clouds…'
      : cloud?.status === 'error'
        ? 'Cloud detection failed; the image is sent as is.'
        : cloud?.mask
          ? `${pct < 1 ? pct.toFixed(1) : pct.toFixed(0)} % of the image looks like cloud${on ? ' (tinted in the preview)' : ''}.`
          : 'No clouds detected.';
  return (
    <Switch
      checked={on}
      onChange={(e) => useScene.getState().setParams({ cloudMask: e.currentTarget.checked })}
      disabled={running}
      label={
        <Group gap={6} wrap="nowrap">
          Cloud mask
          {cloud?.status === 'detecting' && <Loader size={10} />}
        </Group>
      }
      description={description}
    />
  );
}

function RunControls() {
  const input = useScene((s) => s.input);
  const running = useScene((s) => s.run.status === 'running');
  return (
    <Group gap="xs" grow>
      <Button leftSection={<IconPlayerPlay size={16} />} disabled={!input || running} loading={running} onClick={() => void runPrediction()}>
        Estimate heights
      </Button>
      {running && (
        <Button variant="default" leftSection={<IconX size={16} />} onClick={cancelPrediction}>
          Cancel
        </Button>
      )}
    </Group>
  );
}

function CurrentResult() {
  const scene = useScene((s) => s.scene);
  if (!scene) return null;
  const st = scene.stats;
  return (
    <PanelSection title="Current result" right={<ProductBadge scene={scene} />} gap={6}>
      <Text size="sm" fw={600} truncate="end">
        {scene.name}
      </Text>
      <KeyValueRows
        rows={[
          [scene.product === 'DSM' ? 'Elevation range' : 'Height range', `${st.min.toFixed(2)} – ${st.max.toFixed(2)} m`],
          ['Mean / median', `${st.mean.toFixed(2)} / ${st.median.toFixed(2)} m`],
          ['Below 1 m', `${(st.fracBelow1m * 100).toFixed(1)} %`],
          ['Grid', `${scene.heights.width} × ${scene.heights.height} @ ${scene.gsd.toFixed(3)} m`],
          ['Source', scene.provenance.provider],
        ]}
      />
      <Button size="xs" variant="default" leftSection={<IconInfoCircle size={14} />} onClick={() => useUi.getState().openInspector('info')}>
        Details in inspector
      </Button>
    </PanelSection>
  );
}

/** Left dock: input, parameters, run. */
export function ProjectPanel() {
  const open = useUi((s) => s.projectOpen);
  if (!open) return null;
  return (
    <aside className={`${classes.panel} ${classes.panelLeft} dw-no-print`} aria-label="Project">
      <div className={classes.panelHeader}>
        <span>Project</span>
        <Tooltip label="Hide panel ([)">
          <ActionIcon onClick={() => useUi.getState().set({ projectOpen: false })} aria-label="Hide project panel">
            <IconChevronLeft size={16} />
          </ActionIcon>
        </Tooltip>
      </div>
      <ScrollArea style={{ flex: 1 }} type="auto">
        <Stack p="md" gap="lg">
          <PanelSection title="Input image">
            <InputCard />
          </PanelSection>
          <Divider />
          <PanelSection title="Model parameters">
            <Parameters />
            <RunControls />
          </PanelSection>
          <CurrentResult />
        </Stack>
      </ScrollArea>
    </aside>
  );
}
