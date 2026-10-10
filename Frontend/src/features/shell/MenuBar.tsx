import { useState } from 'react';
import { Menu, Menubar, Text, useMantineColorScheme } from '@mantine/core';
import {
  IconBox,
  IconBrain,
  IconClock,
  IconCompass,
  IconCube,
  IconDeviceFloppy,
  IconEye,
  IconFile,
  IconFileExport,
  IconFilePlus,
  IconFolder,
  IconFolderOpen,
  IconHelpCircle,
  IconInfoCircle,
  IconKeyboard,
  IconMap2,
  IconPackage,
  IconPlaneTilt,
  IconRoute,
  IconRuler,
  IconSettings,
  IconSparkles,
  IconTable,
  IconTools,
  IconWalk,
} from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useUi } from '@/store/ui';
import { useView } from '@/store/view';
import { useSettings } from '@/store/settings';
import { newProject, openSample } from '@/features/files/openFile';
import { openRecent, saveProject } from '@/features/files/project';
import { runExport, type ExportKind } from '@/features/files/exports';
import { listRecent, type RecentEntry } from '@/lib/recent';
import { loadSamples, useSamples } from '@/lib/samples';
import { setViewMode } from '@/features/viewport/overlays/ViewSwitcher';
import { setCameraMode } from '@/features/viewport/overlays/NavigationHud';
import { openAny } from './commands';
import { toggleMeasure } from '@/features/analysis/MeasureMenu';
import { useTool } from '@/store/tool';
import { startTour } from '@/features/help/tour/Tour';
import classes from './shell.module.css';

const I = { size: 16, stroke: 1.6 };

function Shortcut({ k }: { k: string }) {
  return (
    <Text span size="xs" c="dimmed" className="dw-mono">
      {k}
    </Text>
  );
}

function TargetLabel({ icon: Icon, label }: { icon: typeof IconFolder; label: string }) {
  return (
    <span className={classes.menuTarget}>
      <Icon size={18} stroke={1.6} aria-hidden className={classes.menuIcon} />
      <span className={classes.menuLabel}>{label}</span>
    </span>
  );
}

function ExportItem({ kind, label, disabled }: { kind: ExportKind; label: string; disabled?: boolean }) {
  return (
    <Menu.Item onClick={() => void runExport(kind)} disabled={disabled}>
      {label}
    </Menu.Item>
  );
}

/** Desktop-style application menu bar: File · View · Tools · Help (Data/UI.md). */
export function MenuBar() {
  const hasScene = useScene((s) => !!s.scene);
  const hasServerMesh = useScene((s) => !!s.scene?.artefacts.some((a) => a.name === 'terrain.glb'));
  const hasAnchor = useScene((s) => !!s.scene?.terrain);
  const mode = useView((s) => s.mode);
  const hoverInfo = useView((s) => s.hoverInfo);
  const measuring = useTool((s) => s.tool === 'measure');
  const ui = useUi();
  const showStatusBar = useSettings((s) => s.showStatusBar);
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const [recent, setRecent] = useState<RecentEntry[]>([]);
  const samples = useSamples((s) => s.samples);

  return (
    <Menubar
      className={classes.menubar}
      data-tour="menubar"
      aria-label="Application menu"
      onOpenChange={(i) => {
        if (i !== 0) return;
        void listRecent().then(setRecent);
        void loadSamples();
      }}
    >
      <Menubar.Menu width={260}>
        <Menubar.Target className={classes.menuButton}>
          <TargetLabel icon={IconFolder} label="File" />
        </Menubar.Target>
        <Menubar.Dropdown>
          <Menu.Item leftSection={<IconFilePlus {...I} />} rightSection={<Shortcut k="Alt+N" />} onClick={newProject}>
            New Project
          </Menu.Item>
          <Menu.Item leftSection={<IconFolderOpen {...I} />} rightSection={<Shortcut k="Ctrl+O" />} onClick={() => void openAny()}>
            Open…
          </Menu.Item>
          <Menu.Sub>
            <Menu.Sub.Target>
              <Menu.Sub.Item leftSection={<IconClock {...I} />}>Recent</Menu.Sub.Item>
            </Menu.Sub.Target>
            <Menu.Sub.Dropdown miw={240}>
              {recent.length === 0 && <Menu.Item disabled>No Recent Projects</Menu.Item>}
              {recent.map((r) => (
                <Menu.Item key={r.id} onClick={() => void openRecent(r.id)} leftSection={r.thumbnail ? <img src={r.thumbnail} alt="" width={36} height={24} style={{ objectFit: 'cover' }} /> : <IconFile {...I} />}>
                  <Text size="sm">{r.name}</Text>
                  <Text size="xs" c="dimmed">
                    {new Date(r.savedAt).toLocaleString()}
                  </Text>
                </Menu.Item>
              ))}
            </Menu.Sub.Dropdown>
          </Menu.Sub>
          <Menu.Sub>
            <Menu.Sub.Target>
              <Menu.Sub.Item leftSection={<IconSparkles {...I} />}>Sample Scenes</Menu.Sub.Item>
            </Menu.Sub.Target>
            <Menu.Sub.Dropdown miw={240}>
              {samples.length === 0 && <Menu.Item disabled>No Sample Scenes</Menu.Item>}
              {samples.map((s) => (
                <Menu.Item key={s.id} leftSection={<IconBox {...I} />} onClick={() => void openSample(s)}>
                  {s.name}
                </Menu.Item>
              ))}
            </Menu.Sub.Dropdown>
          </Menu.Sub>
          <Menu.Item leftSection={<IconDeviceFloppy {...I} />} rightSection={<Shortcut k="Ctrl+S" />} disabled={!hasScene} onClick={() => void saveProject()}>
            Save Project
          </Menu.Item>
          <Menu.Divider />
          <Menu.Sub>
            <Menu.Sub.Target>
              <Menu.Sub.Item leftSection={<IconFileExport {...I} />} disabled={!hasScene}>
                Export
              </Menu.Sub.Item>
            </Menu.Sub.Target>
            <Menu.Sub.Dropdown miw={220}>
              <Menu.Item leftSection={<IconPackage {...I} />} onClick={() => void saveProject()}>
                Entire Project (.dwproj)
              </Menu.Item>
              <Menu.Divider />
              <Menu.Sub>
                <Menu.Sub.Target>
                  <Menu.Sub.Item leftSection={<IconCube {...I} />}>3D Object</Menu.Sub.Item>
                </Menu.Sub.Target>
                <Menu.Sub.Dropdown miw={260}>
                  <ExportItem kind="glb" label="GLB — glTF binary, textured" />
                  <ExportItem kind="obj" label="OBJ — with MTL + texture (.zip)" />
                  <ExportItem kind="ply" label="PLY — vertex-coloured mesh" />
                  <ExportItem kind="stl" label="STL — 3D printing" />
                  {hasServerMesh && (
                    <>
                      <Menu.Divider />
                      <ExportItem kind="server-glb" label="Model output mesh (terrain.glb)" />
                    </>
                  )}
                </Menu.Sub.Dropdown>
              </Menu.Sub>
              <Menu.Sub>
                <Menu.Sub.Target>
                  <Menu.Sub.Item leftSection={<IconMap2 {...I} />}>Heatmap</Menu.Sub.Item>
                </Menu.Sub.Target>
                <Menu.Sub.Dropdown miw={180}>
                  <ExportItem kind="png" label="PNG" />
                  <ExportItem kind="jpg" label="JPG" />
                </Menu.Sub.Dropdown>
              </Menu.Sub>
              <Menu.Sub>
                <Menu.Sub.Target>
                  <Menu.Sub.Item leftSection={<IconTable {...I} />}>Elevation Data</Menu.Sub.Item>
                </Menu.Sub.Target>
                <Menu.Sub.Dropdown miw={260}>
                  <Menu.Item onClick={() => void runExport('geotiff')} rightSection={<Shortcut k="Ctrl+E" />}>
                    GeoTIFF — Float32 metres
                  </Menu.Item>
                  <ExportItem kind="geotiff-ndsm" label="GeoTIFF — height above ground (nDSM)" disabled={!hasAnchor} />
                  <ExportItem kind="geotiff-dtm" label="GeoTIFF — terrain elevation (DTM)" disabled={!hasAnchor} />
                  <ExportItem kind="npy" label="NumPy .npy — Float32 metres" />
                  <ExportItem kind="png16" label="16-bit PNG — encoded heights" />
                </Menu.Sub.Dropdown>
              </Menu.Sub>
              <Menu.Divider />
              <ExportItem kind="screenshot" label="Viewport Screenshot (PNG)" />
            </Menu.Sub.Dropdown>
          </Menu.Sub>
          <Menu.Divider />
          <Menu.Item leftSection={<IconSettings {...I} />} onClick={() => ui.set({ dialog: 'settings' })}>
            Settings
          </Menu.Item>
        </Menubar.Dropdown>
      </Menubar.Menu>

      <Menubar.Menu width={260}>
        <Menubar.Target className={classes.menuButton}>
          <TargetLabel icon={IconEye} label="View" />
        </Menubar.Target>
        <Menubar.Dropdown>
          <Menu.Label>Visualisation</Menu.Label>
          <Menu.RadioGroup value={mode} onChange={(v) => setViewMode(v as never)}>
            <Menu.RadioItem value="dsm3d" rightSection={<Shortcut k="1" />} closeMenuOnClick>
              3D Height Map (DSM)
            </Menu.RadioItem>
            <Menu.RadioItem value="heightmap" rightSection={<Shortcut k="2" />} closeMenuOnClick>
              Height Map
            </Menu.RadioItem>
            <Menu.RadioItem value="image" rightSection={<Shortcut k="3" />} closeMenuOnClick>
              Input Image
            </Menu.RadioItem>
          </Menu.RadioGroup>
          <Menu.Divider />
          <Menu.Label>Workspace</Menu.Label>
          <Menu.CheckboxItem checked={ui.projectOpen} onChange={() => ui.set({ projectOpen: !ui.projectOpen })} rightSection={<Shortcut k="[" />}>
            Project Panel
          </Menu.CheckboxItem>
          <Menu.CheckboxItem checked={ui.inspectorOpen} onChange={() => ui.set({ inspectorOpen: !ui.inspectorOpen })} rightSection={<Shortcut k="]" />}>
            Inspector
          </Menu.CheckboxItem>
          <Menu.CheckboxItem checked={showStatusBar} onChange={() => useSettings.getState().set({ showStatusBar: !showStatusBar })}>
            Status Bar
          </Menu.CheckboxItem>
          <Menu.Divider />
          <Menu.Label>Theme</Menu.Label>
          <Menu.RadioGroup value={colorScheme} onChange={(v) => setColorScheme(v as never)}>
            <Menu.RadioItem value="light">Light</Menu.RadioItem>
            <Menu.RadioItem value="dark">Dark</Menu.RadioItem>
            <Menu.RadioItem value="auto">System</Menu.RadioItem>
          </Menu.RadioGroup>
        </Menubar.Dropdown>
      </Menubar.Menu>

      <Menubar.Menu width={260}>
        <Menubar.Target className={classes.menuButton}>
          <TargetLabel icon={IconTools} label="Tools" />
        </Menubar.Target>
        <Menubar.Dropdown>
          <Menu.Label>Navigation</Menu.Label>
          <Menu.Item leftSection={<IconPlaneTilt {...I} />} rightSection={<Shortcut k="G" />} disabled={!hasScene} onClick={() => setCameraMode('flight')}>
            Flight Simulator
          </Menu.Item>
          <Menu.Item leftSection={<IconWalk {...I} />} rightSection={<Shortcut k="F" />} disabled={!hasScene} onClick={() => setCameraMode('walk')}>
            First-Person Walk
          </Menu.Item>
          <Menu.Item leftSection={<IconRoute {...I} />} rightSection={<Shortcut k="T" />} disabled={!hasScene} onClick={() => setCameraMode('tour')}>
            Drone Tour
          </Menu.Item>
          <Menu.Divider />
          <Menu.Label>Analysis</Menu.Label>
          <Menu.Item leftSection={<IconRuler {...I} />} rightSection={<Shortcut k="M" />} disabled={!hasScene} onClick={toggleMeasure}>
            {measuring ? 'Stop Measuring' : 'Measure…'}
          </Menu.Item>
          <Menu.Divider />
          <Menu.Label>Display</Menu.Label>
          <Menu.CheckboxItem checked={hoverInfo} disabled={!hasScene} onChange={() => useView.getState().set({ hoverInfo: !hoverInfo })} closeMenuOnClick={false}>
            Hover Details
          </Menu.CheckboxItem>
        </Menubar.Dropdown>
      </Menubar.Menu>

      <Menubar.Menu width={240}>
        <Menubar.Target className={classes.menuButton} data-tour="help">
          <TargetLabel icon={IconHelpCircle} label="Help" />
        </Menubar.Target>
        <Menubar.Dropdown>
          <Menu.Item leftSection={<IconCompass {...I} />} onClick={startTour}>
            Take the Tour
          </Menu.Item>
          <Menu.Divider />
          <Menu.Item leftSection={<IconKeyboard {...I} />} rightSection={<Shortcut k="?" />} onClick={() => ui.set({ dialog: 'shortcuts' })}>
            Keyboard Shortcuts
          </Menu.Item>
          <Menu.Item leftSection={<IconBrain {...I} />} onClick={() => ui.set({ dialog: 'model' })}>
            Model Information
          </Menu.Item>
          <Menu.Divider />
          <Menu.Item leftSection={<IconInfoCircle {...I} />} onClick={() => ui.set({ dialog: 'about' })}>
            About DepthWizard
          </Menu.Item>
        </Menubar.Dropdown>
      </Menubar.Menu>
    </Menubar>
  );
}
