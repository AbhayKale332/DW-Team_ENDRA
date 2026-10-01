import { Button, CloseButton, Group, Loader, Progress, Stack, Text } from '@mantine/core';
import { IconCube, IconX } from '@tabler/icons-react';
import { cancelSampleLoad, useSampleLoad } from '@/lib/samples';
import classes from './overlays.module.css';

const mb = (bytes: number) => (bytes / 1024 / 1024).toFixed(1);

/** Progress card while a sample scene downloads and opens. */
export function SampleLoading() {
  const load = useSampleLoad((s) => s.load);
  if (!load || load.stage === 'extras' || load.preview) return null;
  const known = load.stage === 'download' && load.total > 0;
  const pct = known ? Math.min(100, (load.loaded / load.total) * 100) : 100;
  const label = load.stage === 'open' ? 'Opening scene' : 'Downloading';
  const amount = known ? `${pct.toFixed(0)} % · ${mb(load.loaded)} / ${mb(load.total)} MB` : load.stage === 'download' && load.loaded > 0 ? `${mb(load.loaded)} MB` : '';
  return (
    <div className={classes.empty}>
      <div className={`dw-float ${classes.emptyCard}`} role="status" aria-live="polite" aria-label={`Loading sample ${load.name}`}>
        <Stack gap="sm">
          <Text fw={600} truncate>
            {load.name}
          </Text>
          <Progress value={pct} animated={!known} striped={!known} size="md" aria-label={label} />
          <Group justify="space-between" wrap="nowrap">
            <Text size="sm" c="dimmed">
              {label}
            </Text>
            <Text size="xs" c="dimmed" className="dw-mono">
              {amount}
            </Text>
          </Group>
          {load.stage === 'download' && (
            <Group justify="flex-end">
              <Button size="xs" variant="default" leftSection={<IconX size={14} />} onClick={cancelSampleLoad}>
                Cancel
              </Button>
            </Group>
          )}
        </Stack>
      </div>
    </div>
  );
}

/** Slim status while the rest of an opened sample (source image, model outputs) downloads behind the scene. */
export function SampleExtras() {
  const load = useSampleLoad((s) => (s.load?.stage === 'extras' ? s.load : null));
  if (!load) return null;
  const pct = load.total > 0 ? Math.min(100, (load.loaded / load.total) * 100) : null;
  return (
    <div className="dw-float" style={{ alignSelf: 'center', padding: '4px 4px 4px 11px', fontSize: 11.5 }} role="status" aria-label={`Downloading the full sample ${load.name}`}>
      <Group gap={8} wrap="nowrap">
        <Loader size={12} />
        <span>Downloading full scene</span>
        <span className="dw-mono" style={{ color: 'var(--dw-dim)' }}>
          {pct === null ? `${mb(load.loaded)} MB` : `${pct.toFixed(0)} %`}
        </span>
        <CloseButton size="sm" onClick={cancelSampleLoad} aria-label="Stop downloading the full scene" />
      </Group>
    </div>
  );
}

/** One quick look: the image once loaded, a spinner before, a note if it could not be loaded. */
function Look({ url, label }: { url: string | null; label: string }) {
  return (
    <figure className={classes.sampleLook}>
      {url ? (
        <img src={url} alt={label} className={classes.sampleLookImage} />
      ) : url === null ? (
        <Loader size="md" aria-label={`Loading ${label.toLowerCase()}`} />
      ) : (
        <Text size="sm" c="dimmed">
          {label} unavailable
        </Text>
      )}
      <figcaption className={classes.sampleLookCaption}>{label}</figcaption>
    </figure>
  );
}

/** A sample's input image and height map over the viewport while its project downloads; then the way into 3D. */
export function SamplePreview() {
  const load = useSampleLoad((s) => (s.load?.preview ? s.load : null));
  if (!load?.preview) return null;
  const { image, height, ready } = load.preview;
  const pct = load.total > 0 ? Math.min(100, (load.loaded / load.total) * 100) : null;
  const amount = load.stage === 'open' ? '' : pct === null ? (load.loaded ? `${mb(load.loaded)} MB` : '') : `${pct.toFixed(0)} % · ${mb(load.loaded)} / ${mb(load.total)} MB`;
  return (
    <div className={classes.samplePreview}>
      <Text fw={600} size="lg" className={classes.sampleTitle} truncate>
        {load.name}
      </Text>
      <div className={classes.sampleLooks}>
        <Look url={image} label="Input image" />
        {image !== null && <Look url={height} label="Height map" />}
      </div>
      <div className={classes.sampleAction}>
        {ready ? (
          <Button size="xl" radius="md" leftSection={<IconCube size={26} />} onClick={cancelSampleLoad} autoFocus className={classes.sampleOpen}>
            Open 3D viewer
          </Button>
        ) : (
          image !== null &&
          height !== null && (
            <div className="dw-float" style={{ padding: '6px 6px 6px 12px', fontSize: 12.5 }} role="status" aria-live="polite">
              <Group gap={10} wrap="nowrap">
                <Loader size={14} />
                <span>{load.stage === 'open' ? 'Preparing 3D scene' : 'Downloading 3D scene'}</span>
                {amount && (
                  <span className="dw-mono" style={{ color: 'var(--dw-dim)' }}>
                    {amount}
                  </span>
                )}
                <CloseButton size="sm" onClick={cancelSampleLoad} aria-label="Cancel loading the sample" />
              </Group>
            </div>
          )
        )}
      </div>
    </div>
  );
}
