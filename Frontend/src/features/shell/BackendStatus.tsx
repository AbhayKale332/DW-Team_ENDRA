import { useEffect, useState } from 'react';
import { Group, Text, Tooltip, UnstyledButton } from '@mantine/core';
import { getProvider } from '@/api/registry';
import type { BackendStatus as Status } from '@/api/provider';
import { useSettings } from '@/store/settings';
import { useUi } from '@/store/ui';

const COLORS: Record<Status['state'], string> = {
  unknown: 'var(--dw-faint)',
  connecting: '#e5b73a',
  running: '#2fb36f',
  sleeping: '#e5b73a',
  building: '#e5b73a',
  starting: '#e5b73a',
  error: '#e5484d',
  paused: '#e5484d',
};

const LABELS: Record<Status['state'], string> = {
  unknown: 'Not connected',
  connecting: 'Connecting…',
  running: 'Model online',
  sleeping: 'Model waking up',
  building: 'Model building',
  starting: 'Model starting',
  error: 'Model unreachable',
  paused: 'Model paused',
};

/** Live connection state of the inference backend, shown in the header. Connects lazily on first render. */
export function BackendStatus() {
  const configured = useSettings((s) => s.provider);
  const spaceId = useSettings((s) => s.spaceId);
  const [status, setStatus] = useState<Status>({ state: 'unknown' });

  useEffect(() => {
    let alive = true;
    const p = getProvider({ provider: configured, spaceId });
    const on = (s: Status) => alive && setStatus(s);
    void p.status(on).then(on);
    // Re-check every 2 minutes so a sleeping Space shows up as waking/online.
    const t = setInterval(() => void p.status().then(on), 120_000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [configured, spaceId]);

  const provider = getProvider({ provider: configured, spaceId }).id;

  const label = provider === 'mock' ? 'Offline demo' : LABELS[status.state];
  return (
    <Tooltip label={status.message ?? (provider === 'mock' ? 'Using the bundled sample instead of the live model' : spaceId)} multiline maw={320}>
      <UnstyledButton
        onClick={() => useUi.getState().set({ dialog: 'settings' })}
        aria-label={`Model service: ${label}. Open settings`}
        px={10}
        h={26}
        style={{ borderRadius: 999, border: '1px solid var(--dw-line)', display: 'inline-flex', alignItems: 'center' }}
      >
        <Group gap={7} wrap="nowrap">
          <span style={{ width: 7, height: 7, borderRadius: '50%', background: provider === 'mock' ? '#6a4cf0' : COLORS[status.state], flex: 'none' }} />
          <Text size="xs" fw={500} c="dimmed" visibleFrom="md">
            {label}
          </Text>
        </Group>
      </UnstyledButton>
    </Tooltip>
  );
}
