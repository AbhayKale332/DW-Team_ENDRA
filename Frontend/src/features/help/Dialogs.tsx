import { useState } from 'react';
import { Alert, Anchor, Badge, Button, Divider, Drawer, Group, Kbd, List, Modal, Select, SimpleGrid, Slider, Stack, Table, Text, TextInput, Title, Typography } from '@mantine/core';
import { IconBookmark, IconLock, IconTrash } from '@tabler/icons-react';
import { useUi } from '@/store/ui';
import { useSettings } from '@/store/settings';
import { getProvider, PROVIDER_OPTIONS, type ProviderId } from '@/api/registry';
import { useCamera, viewportApi, CAMERA_MODE_LABELS, type CameraMode } from '@/store/camera';
import { SHORTCUTS } from '@/features/shell/commands';
import { setCameraMode } from '@/features/viewport/overlays/NavigationHud';
import { BrandMark } from '@/features/shell/Brand';

const close = () => useUi.getState().set({ dialog: null });

function SettingsDialog() {
  const s = useSettings();
  const [checking, setChecking] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  return (
    <Stack gap="md">
      <Select
        label="Inference provider"
        data={PROVIDER_OPTIONS}
        value={s.provider}
        onChange={(v) => v && s.set({ provider: v as ProviderId })}
        description="The offline demo returns a bundled sample result — useful without network access."
      />
      <TextInput label="Model Space" value={s.spaceId} readOnly description="Configured on the server (VITE_SPACE_ID / HF_SPACE_URL in .env)." />
      <Alert variant="light" color="gray" p="sm" icon={<IconLock size={16} />}>
        <Text size="xs">
          The Space is private. Requests go through this app's own server at <code>/hf-space</code>, which adds the access token from the <code>HF_TOKEN</code> variable in{' '}
          <code>.env</code>. The token is never sent to or stored in the browser.
        </Text>
      </Alert>
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
            setResult(st.state === 'running' ? 'Connected — the model is online.' : (st.message ?? st.state));
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
      <Text size="xs" c="dimmed" style={{ gridColumn: '1 / -1' }}>
        Touch: one finger orbits (pans in map views), two fingers zoom and pan. Shortcuts are ignored while typing in a field.
      </Text>
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
        The deployed model returns height above local ground (nDSM). For GeoTIFF inputs DepthWizard restores the georeferencing in the browser, so exports are
        positioned correctly; absolute elevations (DSM = terrain + nDSM) require a backend with DEM calibration.
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

function DocsDrawer() {
  return (
    <Typography fz="sm">
      <h3>Workflow</h3>
      <ol>
        <li>
          <b>Open</b> a PNG, JPG or GeoTIFF (File → Load / Open, drag and drop, or the Project panel).
        </li>
        <li>
          <b>Declare the resolution.</b> GeoTIFFs provide it; for plain images pick the metres per pixel. Heights scale with this value.
        </li>
        <li>
          <b>Estimate heights.</b> The image is sent to the DepthWizard model; progress is shown over the viewport.
        </li>
        <li>
          <b>Explore</b> in the 3D DSM view, the Height Map or the Input Image (quick switcher, top left, or keys 1–3).
        </li>
        <li>
          <b>Analyse</b> heights and slopes with Probe, Measure and Profile; <b>validate</b> against a reference raster in the Inspector.
        </li>
        <li>
          <b>Export</b> the DSM as GeoTIFF, the terrain as GLB/OBJ/PLY/STL, or the heatmap as PNG/JPG; save the workspace as a project.
        </li>
      </ol>
      <h3>Products</h3>
      <ul>
        <li>
          <b>rDSM</b> — plain images: heights above local ground, not georeferenced. Absolute values depend on the declared resolution.
        </li>
        <li>
          <b>nDSM</b> — GeoTIFFs: heights above ground on the image's own coordinate grid.
        </li>
        <li>
          <b>DSM</b> — absolute elevations; produced when the backend performs DEM calibration.
        </li>
      </ul>
      <h3>Navigation</h3>
      <ul>
        <li>Orbit: drag to rotate, right-drag to pan, scroll to zoom. The compass faces north on click.</li>
        <li>First person (F): pointer-lock mouse look, WASD, Shift to run.</li>
        <li>Flight simulator (G): W/S throttle, arrows pitch and roll, Q/E yaw; the HUD shows height above ground.</li>
        <li>Drone tour (T): a cinematic loop for presentations and recordings.</li>
      </ul>
      <h3>Projection</h3>
      <p>
        Every grid pixel becomes a vertex at its pixel centre (x east, z south, y up, metres). The optical image is draped with pixel-centre exact texture coordinates, so a feature in the image
        sits exactly on its estimated height. Buildings are drawn with vertical walls where the height jumps by more than the wall threshold.
      </p>
      <h3>Limitations</h3>
      <ul>
        <li>Scenes up to 8192 px (long side) are processed at full resolution as 512 px tiles; larger ones are downsampled first. The drape keeps full resolution.</li>
        <li>Cloud, water glint and deep shadow can produce spurious heights; check the flat-ground warning.</li>
      </ul>
    </Typography>
  );
}

function CameraDialog() {
  const cam = useCamera();
  const [name, setName] = useState('');
  return (
    <Stack gap="md">
      <div>
        <Text size="sm" fw={500} mb={6}>
          Navigation mode
        </Text>
        <Group gap="xs">
          {(Object.keys(CAMERA_MODE_LABELS) as CameraMode[]).map((m) => (
            <Button key={m} size="xs" variant={cam.mode === m ? 'filled' : 'default'} onClick={() => setCameraMode(m)} aria-pressed={cam.mode === m}>
              {CAMERA_MODE_LABELS[m]}
            </Button>
          ))}
        </Group>
      </div>
      <div>
        <Text size="sm" fw={500} mb={6}>
          Preset views
        </Text>
        <Group gap="xs">
          {(['fit', 'top', 'north', 'east', 'south', 'west'] as const).map((p) => (
            <Button key={p} size="xs" variant="default" onClick={() => viewportApi.preset(p)} disabled={cam.mode !== 'orbit'}>
              {p === 'fit' ? 'Fit scene' : p === 'top' ? 'Top down' : `From ${p}`}
            </Button>
          ))}
        </Group>
      </div>
      <div>
        <Group justify="space-between" mb={4}>
          <Text size="sm" fw={500}>
            Field of view
          </Text>
          <Text size="xs" c="dimmed" className="dw-mono">
            {cam.fov}°
          </Text>
        </Group>
        <Slider min={25} max={90} step={1} value={cam.fov} onChange={(v) => cam.set({ fov: v })} label={null} aria-label="Field of view" />
      </div>
      <div>
        <Group justify="space-between" mb={4}>
          <Text size="sm" fw={500}>
            Movement speed (walk / flight)
          </Text>
          <Text size="xs" c="dimmed" className="dw-mono">
            {cam.moveSpeed.toFixed(1)}×
          </Text>
        </Group>
        <Slider min={0.2} max={5} step={0.1} value={cam.moveSpeed} onChange={(v) => cam.set({ moveSpeed: v })} label={null} aria-label="Movement speed" />
      </div>
      <Divider label="Bookmarks" labelPosition="left" />
      <Group gap="xs" align="flex-end">
        <TextInput style={{ flex: 1 }} label="Save current view" placeholder="e.g. Village from the east" value={name} onChange={(e) => setName(e.currentTarget.value)} disabled={cam.mode !== 'orbit'} />
        <Button
          leftSection={<IconBookmark size={14} />}
          disabled={cam.mode !== 'orbit'}
          onClick={() => {
            const pose = viewportApi.getPose();
            if (!pose) return;
            cam.set({ bookmarks: [...cam.bookmarks, { id: crypto.randomUUID(), name: name.trim() || `View ${cam.bookmarks.length + 1}`, ...pose }] });
            setName('');
          }}
        >
          Save
        </Button>
      </Group>
      {cam.bookmarks.length === 0 ? (
        <Text size="xs" c="dimmed">
          No saved views yet. Bookmarks are stored with the project.
        </Text>
      ) : (
        <List spacing={4} size="sm" listStyleType="none">
          {cam.bookmarks.map((b) => (
            <List.Item key={b.id}>
              <Group justify="space-between" wrap="nowrap">
                <Anchor component="button" onClick={() => { setCameraMode('orbit'); setTimeout(() => viewportApi.setPose(b), 50); }}>
                  {b.name}
                </Anchor>
                <Button size="compact-xs" variant="subtle" color="gray" onClick={() => cam.set({ bookmarks: cam.bookmarks.filter((x) => x.id !== b.id) })} aria-label={`Delete ${b.name}`}>
                  <IconTrash size={12} />
                </Button>
              </Group>
            </List.Item>
          ))}
        </List>
      )}
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
      <Modal opened={dialog === 'camera'} onClose={close} title="Camera control" size="lg" overlayProps={{ backgroundOpacity: 0.15, blur: 0 }}>
        <CameraDialog />
      </Modal>
      <Drawer opened={dialog === 'docs'} onClose={close} title="Documentation" position="right" size="lg">
        <DocsDrawer />
      </Drawer>
    </>
  );
}
