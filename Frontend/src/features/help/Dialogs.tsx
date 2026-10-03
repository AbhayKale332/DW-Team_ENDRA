import { useState } from 'react';
import { Badge, Button, Divider, Group, Kbd, Modal, Select, SimpleGrid, Stack, Table, Text, Title } from '@mantine/core';
import { useUi } from '@/store/ui';
import { useSettings } from '@/store/settings';
import { getProvider, PROVIDER_OPTIONS, type ProviderId } from '@/api/registry';
import { SHORTCUTS } from '@/features/shell/commands';
import { BrandMark } from '@/features/shell/Brand';

const close = () => useUi.getState().set({ dialog: null });

function SettingsDialog() {
  const s = useSettings();
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  return (
    <Stack gap="md">
      <Select label="Inference provider" data={PROVIDER_OPTIONS} value={s.provider} onChange={(v) => v && s.set({ provider: v as ProviderId })} />
      <Group justify="space-between">
        <Text size="xs" c="dimmed">
          {result}
        </Text>
        <Button
          variant="default"
          loading={checking}
          onClick={async () => {
            setChecking(true);
            const st = await getProvider({ provider: s.provider, spaceId: s.spaceId }).status();
            setResult(st.state === 'running' ? 'Connected — the model is ready.' : (st.message ?? st.state));
            setChecking(false);
          }}
        >
          Test connection
        </Button>
      </Group>
    </Stack>
  );
}

function ShortcutsDialog() {
  const groups = [...new Set(SHORTCUTS.map((s) => s.group))];
  return (
    <SimpleGrid cols={{ base: 1, sm: 2 }} spacing="lg">
      {groups.map((g) => (
        <div key={g}>
          <Text className="dw-section-title" mb={6}>
            {g}
          </Text>
          <Table fz="sm" verticalSpacing={3} withRowBorders={false}>
            <Table.Tbody>
              {SHORTCUTS.filter((s) => s.group === g).map((s) => (
                <Table.Tr key={s.keys + s.label}>
                  <Table.Td w={120}>
                    <Kbd size="xs">{s.keys}</Kbd>
                  </Table.Td>
                  <Table.Td>{s.label}</Table.Td>
                </Table.Tr>
              ))}
            </Table.Tbody>
          </Table>
        </div>
      ))}
    </SimpleGrid>
  );
}

function ModelDialog() {
  const provider = useSettings((s) => s.provider);
  const spaceId = useSettings((s) => s.spaceId);
  const p = getProvider({ provider, spaceId });
  const m = p.modelInfo();
  const caps = p.capabilities;
  return (
    <Stack gap="md">
      <div>
        <Title order={4}>{m.name}</Title>
        <Text size="sm" mt={4}>
          {m.summary}
        </Text>
      </div>
      <Table fz="sm" verticalSpacing={4} withTableBorder>
        <Table.Tbody>
          {m.details.map(([k, v]) => (
            <Table.Tr key={k}>
              <Table.Td c="dimmed" w="34%">
                {k}
              </Table.Td>
              <Table.Td>{v}</Table.Td>
            </Table.Tr>
          ))}
          <Table.Tr>
            <Table.Td c="dimmed">Endpoint</Table.Td>
            <Table.Td className="dw-mono">{m.endpoint}</Table.Td>
          </Table.Tr>
        </Table.Tbody>
      </Table>
      <Group gap="xs">
        <Badge color={caps.absoluteDsm ? 'teal' : 'gray'}>Absolute DSM {caps.absoluteDsm ? '✓' : '—'}</Badge>
        <Badge color={caps.uncertainty ? 'teal' : 'gray'}>Uncertainty {caps.uncertainty ? '✓' : '—'}</Badge>
        <Badge color={caps.cancel ? 'teal' : 'gray'}>Cancellable {caps.cancel ? '✓' : '—'}</Badge>
      </Group>
      <Text size="xs" c="dimmed">
        The model returns height above ground (nDSM). Georeferenced results can be anchored to a DEM in the Info tab for absolute elevations (DSM).
      </Text>
    </Stack>
  );
}

function AboutDialog() {
  return (
    <Stack gap="sm" align="flex-start">
      <Group>
        <BrandMark size={48} />
        <div>
          <Title order={3}>DepthWizard</Title>
          <Text size="sm" c="dimmed">
            Single-view height estimation and 3D flythrough · v{__APP_VERSION__}
          </Text>
        </div>
      </Group>
      <Text size="sm">
        Built for Smart India Hackathon 2026, problem statement 26175 (Indian Space Research Organisation): turn one optical RGB remote-sensing image into a surface model and explore it in an
        interactive 3D environment.
      </Text>
      <Divider w="100%" />
      <Text size="xs" c="dimmed">
        Open-source components: React (MIT), three.js (MIT), React Three Fiber & drei (MIT), Mantine (MIT), Apache ECharts (Apache-2.0), geotiff.js (MIT), proj4js (MIT), fflate (MIT), Tabler
        Icons (MIT). Model encoder: DINOv3 (Meta DINOv3 licence).
      </Text>
    </Stack>
  );
}

/** All application dialogs, driven by uiStore.dialog. */
export function Dialogs() {
  const dialog = useUi((s) => s.dialog);
  return (
    <>
      <Modal opened={dialog === 'settings'} onClose={close} title="Settings" size="lg">
        <SettingsDialog />
      </Modal>
      <Modal opened={dialog === 'shortcuts'} onClose={close} title="Keyboard shortcuts" size="xl">
        <ShortcutsDialog />
      </Modal>
      <Modal opened={dialog === 'model'} onClose={close} title="Model information" size="lg">
        <ModelDialog />
      </Modal>
      <Modal opened={dialog === 'about'} onClose={close} title="About" size="md">
        <AboutDialog />
      </Modal>
    </>
  );
}
