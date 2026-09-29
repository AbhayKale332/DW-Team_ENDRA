import { useEffect, useState } from 'react';
import { Button, Group, Loader, Stack, Text, ThemeIcon } from '@mantine/core';
import { IconAlertTriangle, IconCheck, IconCircle, IconKey, IconRefresh, IconSparkles, IconX } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useUi } from '@/store/ui';
import { ERROR_COPY } from '@/api/errors';
import type { ProgressStage } from '@/api/provider';
import { cancelPrediction, runPrediction } from './runPrediction';
import { openSample } from '@/features/files/openFile';
import { SAMPLES } from '@/lib/samples';
import classes from '@/features/viewport/overlays/overlays.module.css';

const STAGES: Array<{ id: ProgressStage; label: string }> = [
  { id: 'connecting', label: 'Connecting to the model' },
  { id: 'uploading', label: 'Uploading image' },
  { id: 'queued', label: 'Waiting in queue' },
  { id: 'processing', label: 'Estimating heights (GPU)' },
  { id: 'fetching', label: 'Downloading results' },
  { id: 'building', label: 'Building 3D mesh' },
];
const ORDER = STAGES.map((s) => s.id);

function useElapsed(start: number | null, active: boolean) {
  const [now, setNow] = useState(() => performance.now());
  useEffect(() => {
    if (!active) return;
    const t = setInterval(() => setNow(performance.now()), 500);
    return () => clearInterval(t);
  }, [active]);
  return start ? Math.max(0, (now - start) / 1000) : 0;
}

/** Accessible progress stepper for a model run, plus the recovery UI when a run fails. */
export function ProcessingOverlay() {
  const run = useScene((s) => s.run);
  const tta = useScene((s) => s.params.tta);
  const hasPrevious = useScene((s) => !!s.scene);
  const elapsed = useElapsed(run.startedAt, run.status === 'running');

  if (run.status === 'error' && run.error) {
    const copy = ERROR_COPY[run.error.kind];
    return (
      <div className={`dw-float ${classes.processing}`} role="alert">
        <Stack gap="sm">
          <Group gap="xs" wrap="nowrap">
            <ThemeIcon color="red" variant="light" size="md">
              <IconAlertTriangle size={16} />
            </ThemeIcon>
            <Text fw={600}>{copy.title}</Text>
          </Group>
          <Text size="sm">{copy.hint}</Text>
          {run.error.detail && (
            <Text size="xs" c="dimmed" className="dw-mono" style={{ wordBreak: 'break-word' }}>
              {run.error.detail}
            </Text>
          )}
          <Group gap="xs">
            <Button size="xs" leftSection={<IconRefresh size={14} />} onClick={() => void runPrediction()}>
              Retry
            </Button>
            {run.error.kind === 'auth' && (
              <Button size="xs" variant="default" leftSection={<IconKey size={14} />} onClick={() => useUi.getState().set({ dialog: 'settings' })}>
                Connection settings
              </Button>
            )}
            <Button size="xs" variant="default" leftSection={<IconSparkles size={14} />} onClick={() => void openSample(SAMPLES[0].id)}>
              Open a sample
            </Button>
            <Button size="xs" variant="subtle" color="gray" onClick={() => useScene.getState().setRun({ status: 'idle', error: null })}>
              Dismiss
            </Button>
          </Group>
        </Stack>
      </div>
    );
  }

  if (run.status !== 'running') return null;
  const current = run.stage ?? 'connecting';
  const ci = ORDER.indexOf(current);
  const ev = run.event;
  return (
    <div className={`dw-float ${classes.processing}`} role="status" aria-live="polite" aria-label="Model run progress">
      <Stack gap={10}>
        <Group justify="space-between" wrap="nowrap">
          <Text fw={600}>Processing</Text>
          <Text size="xs" c="dimmed" className="dw-mono">
            {elapsed.toFixed(0)} s{tta ? ' · TTA' : ''}
          </Text>
        </Group>
        <Stack gap={6} component="ol" style={{ listStyle: 'none', margin: 0, padding: 0 }}>
          {STAGES.map((s, i) => {
            const done = i < ci;
            const active = i === ci;
            let detail = '';
            if (active && s.id === 'queued' && ev?.position !== undefined) detail = `position ${ev.position}${ev.eta ? ` · ~${Math.round(ev.eta)} s` : ''}`;
            if (active && s.id === 'processing' && ev?.eta) detail = `~${Math.round(ev.eta)} s remaining`;
            if (active && s.id === 'processing' && ev?.progress !== undefined) detail = `${Math.round((ev.progress ?? 0) * 100)} %`;
            return (
              <Group key={s.id} component="li" gap={10} wrap="nowrap" aria-current={active ? 'step' : undefined} style={{ opacity: i > ci ? 0.45 : 1 }}>
                {done ? (
                  <ThemeIcon size={18} radius="xl" color="teal" variant="filled">
                    <IconCheck size={12} />
                  </ThemeIcon>
                ) : active ? (
                  <Loader size={18} />
                ) : (
                  <IconCircle size={18} stroke={1.5} color="var(--dw-faint)" />
                )}
                <Text size="sm" fw={active ? 600 : 400} style={{ flex: 1 }}>
                  {s.label}
                </Text>
                {detail && (
                  <Text size="xs" c="dimmed" className="dw-mono">
                    {detail}
                  </Text>
                )}
              </Group>
            );
          })}
        </Stack>
        {hasPrevious && current !== 'building' && (
          <Text size="xs" c="dimmed">
            Showing the previous result.
          </Text>
        )}
        {current !== 'building' && (
          <Button size="xs" variant="default" leftSection={<IconX size={14} />} onClick={cancelPrediction}>
            Cancel run
          </Button>
        )}
      </Stack>
    </div>
  );
}
