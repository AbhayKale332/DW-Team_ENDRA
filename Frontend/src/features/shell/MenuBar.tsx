import { useState } from 'react';
import { Menu, Menubar, Text, useMantineColorScheme } from '@mantine/core';
import {
  IconBox,
  IconClock,
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
  IconPlaneTilt,
  IconRoute,
  IconScale,
  IconSettings,
  IconSparkles,
  IconTools,
  IconWalk,
  IconBook,
  IconBrain,
  IconTable,
} from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useUseCases } from '@/store/usecases';
import { useUi } from '@/store/ui';
import { OBJECT_KIND_LABELS, OBJECT_KINDS, toggleAllObjects, toggleObjectKind, useView } from '@/store/view';
import { useSettings } from '@/store/settings';
import { newProject, openSample } from '@/features/files/openFile';
import { openRecent, saveProject } from '@/features/files/project';
import { runExport, type ExportKind } from '@/features/files/exports';
import { listRecent, type RecentEntry } from '@/lib/recent';
import { SAMPLES } from '@/lib/samples';
import { canGeolocate } from '@/lib/osm';
import { setViewMode } from '@/features/viewport/overlays/ViewSwitcher';
import { setCameraMode } from '@/features/viewport/overlays/NavigationHud';
import { openAny } from './commands';
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

/** Desktop-style application menu bar: File Â· View Â· Tools Â· Help (Data/UI.md). */
export function MenuBar() {
  const hasScene = useScene((s) => !!s.scene);
  const hasServerMesh = useScene((s) => !!s.scene?.artefacts.some((a) => a.name === 'terrain.glb'));
  const hasRef = useScene((s) => !!s.reference);
  const hasAnchor = useScene((s) => !!s.scene?.terrain);
  const mode = useView((s) => s.mode);
  const objectKinds = useView((s) => s.objectKinds);
  const hoverInfo = useView((s) => s.hoverInfo);
  const osmOn = useView((s) => s.osm);
  const basemapOn = useView((s) => s.basemap);
  const poiOn = useView((s) => s.poi);
  const canOsm = useScene((s) => canGeolocate(s.scene?.georef));
  const objects = useScene((s) => s.scene?.objects);
  const hasObjects = !!objects;
  const objectCounts = { buildings: objects?.buildings.length ?? 0, trees: objects?.trees.length ?? 0, water: objects?.water.length ?? 0 };
  const ui = useUi();
  const showStatusBar = useSettings((s) => s.showStatusBar);
  const { colorScheme, setColorScheme } = useMantineColorScheme();
  const [recent, setRecent] = useState<RecentEntry[]>([]);

  return (
    <Menubar className={classes.menubar} aria-label="Application menu" onOpenChange={(i) => i === 0 && void listRecent().then(setRecent)}>
      <Menubar.Menu width={260}>
        <Menubar.Target className={classes.menuButton}>
          <TargetLabel icon={IconFolder} label="File" />
        </Menubar.Target>
        <Menubar.Dropdown>
          <Menu.Item leftSection={<IconFilePlus {...I} />} rightSection={<Shortcut k="Alt+N" />} onClick={newProject}>
            New Project
          </Menu.Item>
          <Menu.Item leftSection={<IconFolderOpen {...I} />} rightSection={<Shortcut k="Ctrl+O" />} onClick={() => void openAny()}>
            Load / Open
          </Menu.Item>
          <Menu.Sub>
            <Menu.Sub.Target>
              <Menu.Sub.Item leftSection={<IconClock {...I} />}>Recent</Menu.Sub.Item>
            </Menu.Sub.Target>
            <Menu.Sub.Dropdown miw={240}>
              {recent.length === 0 && <Menu.Item disabled>No recent projects</Menu.Item>}
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
              <Menu.Sub.Item leftSection={<IconSparkles {...I} />}>Sample scenes</Menu.Sub.Item>
            </Menu.Sub.Target>
            <Menu.Sub.Dropdown miw={240}>
              {SAMPLES.map((s) => (
                <Menu.Item key={s.id} leftSection={<IconBox {...I} />} onClick={() => void openSample(s.id)}>
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
              <Menu.Sub>
                <Menu.Sub.Target>
                  <Menu.Sub.Item leftSection={<IconCube {...I} />}>3D Object</Menu.Sub.Item>
                </Menu.Sub.Target>
                <Menu.Sub.Dropdown miw={260}>
                  <ExportItem kind="glb" label="GLB ” glTF binary, textured" />
                  <ExportItem kind="obj" label="OBJ ” with MTL + texture (.zip)" />
                  <ExportItem kind="ply" label="PLY ” vertex-coloured mesh" />
                  <ExportItem kind="stl" label="STL ” 3D printing" />
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
                  <Menu.Sub.Item leftSection={<IconTable {...I} />}>Elevation data</Menu.Sub.Item>
                </Menu.Sub.Target>
                <Menu.Sub.Dropdown miw={260}>
                  <Menu.Item onClick={() => void runExport('geotiff')} rightSection={<Shortcut k="Ctrl+E" />}>
                    GeoTIFF ” Float32 metres
                  </Menu.Item>
                  <ExportItem kind="geotiff-ndsm" label="GeoTIFF: height above ground (nDSM)" disabled={!hasAnchor} />
                  <ExportItem kind="geotiff-dtm" label="GeoTIFF: terrain elevation (DTM)" disabled={!hasAnchor} />
                  <ExportItem kind="npy" label="NumPy .npy ” Float32 metres" />
                  <ExportItem kind="png16" label="16-bit PNG ” encoded heights" />
                </Menu.Sub.Dropdown>
              </Menu.Sub>
              <Menu.Divider />
              <ExportItem kind="screenshot" label="Viewport screenshot (PNG)" />
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
          <Menu.Label>Primary visualisation</Menu.Label>
          <Menu.RadioGroup value={mode} onChange={(v) => setViewMode(v as never)}>
            <Menu.RadioItem value="image" rightSection={<Shortcut k="3" />} closeMenuOnClick>
              Input Image
            </Menu.RadioItem>
            <Menu.RadioItem value="heightmap" rightSection={<Shortcut k="2" />} closeMenuOnClick>
              Heatmap (Height Map)
            </Menu.RadioItem>
            <Menu.RadioItem value="dsm3d" rightSection={<Shortcut k="1" />} closeMenuOnClick>
              3D Height Map (DSM)
            </Menu.RadioItem>
          </Menu.RadioGroup>
          <Menu.Divider />
          <Menu.Label>Workspace</Menu.Label>
          <Menu.CheckboxItem checked={ui.projectOpen} onChange={() => ui.set({ projectOpen: !ui.projectOpen })} rightSection={<Shortcut k="[" />}>
            Project panel
          </Menu.CheckboxItem>
          <Menu.CheckboxItem checked={ui.inspectorOpen} onChange={() => ui.set({ inspectorOpen: !ui.inspectorOpen })} rightSection={<Shortcut k="]" />}>
            Inspector
          </Menu.CheckboxItem>
          <Menu.CheckboxItem checked={showStatusBar} onChange={() => useSettings.getState().set({ showStatusBar: !showStatusBar })}>
            Status bar
          </Menu.CheckboxItem>
          <Menu.Divider />
          <Menu.Label>Theme</Menu.Label>
          <Menu.RadioGroup value={colorScheme} onChange={(v) => setColorScheme(v as never)}>
            <Menu.RadioItem value="light">
              Light
            </Menu.RadioItem>
            <Menu.RadioItem value="dark">
              Dark
            </Menu.RadioItem>
            <Menu.RadioItem value="auto">
              System
            </Menu.RadioItem>
          </Menu.RadioGroup>
        </Menubar.Dropdown>
      </Menubar.Menu>

      <Menubar.Menu width={280}>
        <Menubar.Target className={classes.menuButton}>
          <TargetLabel icon={IconTools} label="Tools" />
        </Menubar.Target>
        <Menubar.Dropdown>
          <Menu.Label>Navigation</Menu.Label>
          {/* remove this camera control */}
          {/* <Menu.Item leftSection={<IconCamera {...I} />} disabled={!hasScene} onClick={() => ui.set({ dialog: 'camera' })}>
            Camera Control
          </Menu.Item> */}
          <Menu.Item leftSection={<IconPlaneTilt {...I} />} rightSection={<Shortcut k="G" />} disabled={!hasScene} onClick={() => setCameraMode('flight')}>
            Flight Simulator
          </Menu.Item>
          <Menu.Item leftSection={<IconWalk {...I} />} rightSection={<Shortcut k="F" />} disabled={!hasScene} onClick={() => setCameraMode('walk')}>
            First-person walk
          </Menu.Item>
          <Menu.Item leftSection={<IconRoute {...I} />} rightSection={<Shortcut k="T" />} disabled={!hasScene} onClick={() => setCameraMode('tour')}>
            Drone tour
          </Menu.Item>
          <Menu.Divider />
          <Menu.Label>3D objects</Menu.Label>
          {OBJECT_KINDS.map((k) => (
            <Menu.CheckboxItem key={k} checked={objectCounts[k] > 0 && objectKinds[k]} disabled={!objectCounts[k]} onChange={() => toggleObjectKind(k)} closeMenuOnClick={false}>
              {OBJECT_KIND_LABELS[k]}
              {objectCounts[k] ? ` (${objectCounts[k]})` : ''}
            </Menu.CheckboxItem>
          ))}
          <Menu.Item disabled={!hasObjects} onClick={toggleAllObjects} rightSection={<Shortcut k="O" />}>
            All 3D objects on / off
          </Menu.Item>
          <Menu.Divider />
          <Menu.Label>Overlays</Menu.Label>
          <Menu.CheckboxItem checked={hoverInfo} disabled={!hasScene} onChange={() => useView.getState().set({ hoverInfo: !hoverInfo })} closeMenuOnClick={false}>
            Hover details
          </Menu.CheckboxItem>
          <Menu.CheckboxItem checked={osmOn && canOsm} disabled={!canOsm} onChange={() => useView.getState().set({ osm: !osmOn })} closeMenuOnClick={false}>
            OpenStreetMap overlay{hasScene && !canOsm ? ' (needs a georeferenced image)' : ''}
          </Menu.CheckboxItem>
          <Menu.CheckboxItem checked={basemapOn && canOsm} disabled={!canOsm} onChange={() => useView.getState().set({ basemap: !basemapOn })} closeMenuOnClick={false}>
            Surrounding basemap{hasScene && !canOsm ? ' (needs a georeferenced image)' : ''}
          </Menu.CheckboxItem>
          <Menu.CheckboxItem checked={poiOn && canOsm} disabled={!canOsm} onChange={() => useView.getState().set({ poi: !poiOn })} closeMenuOnClick={false}>
            Facilities (OSM){hasScene && !canOsm ? ' (needs a georeferenced image)' : ''}
          </Menu.CheckboxItem>
          <Menu.Divider />
          <Menu.Label>Analysis</Menu.Label>
          <Menu.Item leftSection={<IconScale {...I} />} disabled={!hasScene} onClick={() => ui.openInspector('validation')}>
            {hasRef ? 'Validation results' : 'Validate against reference'}
          </Menu.Item>
          <Menu.Item leftSection={<IconScale {...I} />} disabled={!hasScene} onClick={() => useUseCases.getState().set({ open: true })}>
            Scenarios (telecom · flood)
          </Menu.Item>
        </Menubar.Dropdown>
      </Menubar.Menu>

      <Menubar.Menu width={240}>
        <Menubar.Target className={classes.menuButton}>
          <TargetLabel icon={IconHelpCircle} label="Help" />
        </Menubar.Target>
        <Menubar.Dropdown>
          <Menu.Item leftSection={<IconBook {...I} />} onClick={() => ui.set({ dialog: 'docs' })}>
            Documentation
          </Menu.Item>
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
