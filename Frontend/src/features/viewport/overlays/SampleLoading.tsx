import { Button, Group, Progress, Stack, Text } from '@mantine/core';
import { IconX } from '@tabler/icons-react';
import { cancelSampleLoad, useSampleLoad } from '@/lib/samples';
import classes from './overlays.module.css';

const mb = (bytes: number) => (bytes / 1024 / 1024).toFixed(1);

/** Progress card while a sample scene downloads and opens. */
export function SampleLoading() {
  const load = useSampleLoad((s) => s.load);
  if (!load) return null;
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
