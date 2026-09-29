import { ActionIcon, Alert, Badge, Button, Divider, Group, NumberInput, ScrollArea, SegmentedControl, Stack, Switch, Text, Tooltip } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import { IconAlertTriangle, IconChevronLeft, IconPhotoUp, IconPlayerPlay, IconRefresh, IconWorld, IconX } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useUi } from '@/store/ui';
import { effectiveModelGrid, MODEL_MAX_SIDE } from '@/lib/input';
import { stageImage } from '@/features/files/openFile';
import { cancelPrediction, resolveGsd, runPrediction } from '@/features/processing/runPrediction';
import { PRODUCT_LABELS } from '@/lib/sceneBuilder';
import classes from '@/features/shell/shell.module.css';

const PRESETS = [
  { value: '0.3', label: '0.3' },
  { value: '0.5', label: '0.5' },
  { value: '0.6', label: '0.6' },
  { value: '1', label: '1.0' },
];

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <Group justify="space-between" gap="xs" wrap="nowrap">
      <Text size="xs" c="dimmed">
        {k}
      </Text>
      <Text size="xs" className="dw-mono" ta="right" style={{ minWidth: 0, overflow: 'hidden', textOverflow: 'ellipsis' }}>
        {v}
      </Text>
    </Group>
  );
}

function InputCard() {
  const input = useScene((s) => s.input);
  const running = useScene((s) => s.run.status === 'running');
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
      <Stack gap={2}>
        <Row k="Size" v={`${input.width} × ${input.height} px`} />
        <Row k="Bands / depth" v={`${input.bands} × ${input.bitsPerSample}-bit`} />
        <Row
          k="Georeferenced"
          v={
            input.georef ? (
              <Badge size="xs" color="teal" leftSection={<IconWorld size={10} />}>
                {input.georef.epsg ? `EPSG:${input.georef.epsg}` : 'yes'}
              </Badge>
            ) : (
              'no'
            )
          }
        />
        {input.fileGsd && <Row k="GSD from file" v={`${input.fileGsd.toFixed(3)} m/px`} />}
      </Stack>
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
      <div>
        <Text size="sm" fw={500} id="gsd-label">
          Resolution (m / pixel)
        </Text>
        <Text size="xs" c="dimmed" mb={6}>
          Heights scale with this value.
        </Text>
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
            label="Custom resolution"
            description="Metres per pixel of the image you opened"
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
          {source === 'assumed' && 'No resolution known — the model will assume 0.5 m/px. Declare it if you know it.'}
        </Text>
        {eff?.resampled && (
          <Alert mt={8} p={8} color="gray" variant="light" icon={<IconAlertTriangle size={14} />} fz="xs">
            The long side exceeds {MODEL_MAX_SIDE} px, so the model works at {eff.width} × {eff.height} px (≈{eff.gsd.toFixed(2)} m/px). The 3D drape still uses your full-resolution image.
          </Alert>
        )}
      </div>
      <Switch
        checked={params.tta}
        onChange={(e) => setParams({ tta: e.currentTarget.checked })}
        disabled={running}
        label="Higher quality (test-time augmentation)"
        description="Averages 8 flipped/rotated passes. Takes about 1-2 min"
      />
    </Stack>
  );
}

function RunControls() {
  const input = useScene((s) => s.input);
  const running = useScene((s) => s.run.status === 'running');
  return (
    <Group gap="xs" grow>
      <Button leftSection={<IconPlayerPlay size={16} />} disabled={!input || running} loading={running} onClick={() => void runPrediction()} aria-describedby="run-hint">
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
    <Stack gap={6}>
      <Group justify="space-between">
        <span className="dw-section-title">Current result</span>
        <Tooltip label={PRODUCT_LABELS[scene.product].long} multiline maw={260}>
          <Badge size="sm">{scene.product}</Badge>
        </Tooltip>
      </Group>
      <Text size="sm" fw={600} truncate="end">
        {scene.name}
      </Text>
      <Stack gap={2}>
        <Row k="Height range" v={`${st.min.toFixed(2)} – ${st.max.toFixed(2)} m`} />
        <Row k="Mean / median" v={`${st.mean.toFixed(2)} / ${st.median.toFixed(2)} m`} />
        <Row k="Below 1 m" v={`${(st.fracBelow1m * 100).toFixed(1)} %`} />
        <Row k="Grid" v={`${scene.heights.width} × ${scene.heights.height} @ ${scene.gsd.toFixed(3)} m`} />
        <Row k="Source" v={scene.provenance.provider} />
      </Stack>
      <Button size="xs" variant="default" leftSection={<IconRefresh size={14} />} onClick={() => useUi.getState().openInspector('info')}>
        Details in inspector
      </Button>
    </Stack>
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
        <Stack p="md" gap="md">
          <span className="dw-section-title">Input image</span>
          <InputCard />
          <Divider />
          <span className="dw-section-title">Model parameters</span>
          <Parameters />
          <RunControls />
          {/* <Text size="xs" c="dimmed" id="run-hint">
            Runs on the DepthWizard model (Hugging Face Space). Output: height above ground in metres.
          </Text> */}
          {/* <Divider /> */}
          <CurrentResult />
        </Stack>
      </ScrollArea>
    </aside>
  );
}
