import { ActionIcon, Button, Menu, Switch, Tooltip } from '@mantine/core';
import { IconAntenna, IconBook, IconChevronDown, IconDroplet, IconLayoutSidebar, IconLayoutSidebarRight, IconMapPin } from '@tabler/icons-react';
import { gcpEligible, setGcpOpen, useGcp } from '@/features/gcp/gcpStore';
import { useUi } from '@/store/ui';
import { useScene } from '@/store/scene';
import { useUseCases, type UseCase } from '@/store/usecases';
import { setRawOutput, useView } from '@/store/view';
import { MeasureMenu } from '@/features/analysis/MeasureMenu';
import { BrandMark } from './Brand';
import { MenuBar } from './MenuBar';
import { BackendStatus } from './BackendStatus';
import classes from './shell.module.css';

const DOCS_URL = 'https://docs.depthwizard.teamendra.tech/';

/** Header toggle; a pressed one stays tinted so the open panels read at a glance. */
function HeaderToggle({ label, pressed, onClick, children }: { label: string; pressed?: boolean; onClick: () => void; children: React.ReactNode }) {
  return (
    <Tooltip label={label}>
      <ActionIcon size="lg" variant={pressed ? 'light' : 'subtle'} color={pressed ? 'dwBlue' : 'gray'} onClick={onClick} aria-label={label} aria-pressed={pressed}>
        {children}
      </ActionIcon>
    </Tooltip>
  );
}

/** Scenarios: opens the use-case overlay (telecom coverage, flood response) over the viewport. */
function ScenariosMenu() {
  const hasScene = useScene((s) => !!s.scene);
  const open = useUseCases((s) => s.open);
  const active = useUseCases((s) => s.active);
  const show = (u: UseCase) => {
    setGcpOpen(false);
    useUseCases.getState().set({ open: true, active: u });
  };
  return (
    <Menu position="bottom-start" withinPortal>
      <Menu.Target>
        <Button size="compact-md" variant={open ? 'light' : 'subtle'} color={open ? 'dwBlue' : 'gray'} disabled={!hasScene} rightSection={<IconChevronDown size={14} />} aria-label="Scenarios">
          Scenarios
        </Button>
      </Menu.Target>
      <Menu.Dropdown>
        <Menu.Item leftSection={<IconAntenna size={16} />} onClick={() => show('telecom')} color={open && active === 'telecom' ? 'dwBlue' : undefined}>
          Telecom tower coverage
        </Menu.Item>
        <Menu.Item leftSection={<IconDroplet size={16} />} onClick={() => show('flood')} color={open && active === 'flood' ? 'dwBlue' : undefined}>
          Flood response planning
        </Menu.Item>
        {open && (
          <>
            <Menu.Divider />
            <Menu.Item onClick={() => useUseCases.getState().set({ open: false, placing: false, pickingSource: false, playing: false })}>Close scenarios</Menu.Item>
          </>
        )}
      </Menu.Dropdown>
    </Menu>
  );
}

/** Ground control points: heights at picked points make an image without georeferencing an absolute DSM. */
function GcpButton() {
  const eligible = useScene((s) => gcpEligible(s.scene));
  const open = useGcp((s) => s.open);
  const toggle = () => {
    if (!open) useUseCases.getState().set({ open: false, placing: false, pickingSource: false, playing: false });
    setGcpOpen(!open);
  };
  return (
    <Tooltip label={eligible ? 'Ground control points' : 'Ground control points (images without georeferencing)'}>
      {/* the wrapper keeps the tooltip working while the button is disabled */}
      <span style={{ display: 'inline-flex' }}>
        <Button size="compact-md" variant={open ? 'light' : 'subtle'} color={open ? 'dwBlue' : 'gray'} disabled={!eligible} leftSection={<IconMapPin size={16} stroke={1.6} />} onClick={toggle} aria-pressed={open}>
          GCP
        </Button>
      </span>
    </Tooltip>
  );
}

/** Switch beside Scenarios: raw model output, without 3D objects or post-processing. */
function RawToggle() {
  const hasScene = useScene((s) => !!s.scene);
  const raw = useView((s) => s.raw);
  return (
    <Tooltip label="Raw model output, without 3D objects or post-processing" multiline maw={240}>
      {/* the wrapper keeps the tooltip working while the switch is disabled */}
      <div style={{ display: 'flex', alignItems: 'center', paddingInline: 6 }}>
        <Switch
          size="sm"
          color="dwOrange"
          label="Raw"
          labelPosition="left"
          checked={raw}
          disabled={!hasScene}
          onChange={(e) => setRawOutput(e.currentTarget.checked)}
          aria-label="Raw model output"
          styles={{ label: { fontWeight: 500, paddingInlineEnd: 8, cursor: hasScene ? 'pointer' : undefined } }}
        />
      </div>
    </Tooltip>
  );
}

/** Header in three groups: brand + menus · scenario controls · model status + panel toggles. Theme lives in View → Theme. */
export function AppHeader() {
  const projectOpen = useUi((s) => s.projectOpen);
  const inspectorOpen = useUi((s) => s.inspectorOpen);
  return (
    <header className={classes.header}>
      <div className={classes.brand}>
        <BrandMark size={30} />
        <span className={classes.wordmark}>DepthWizard</span>
      </div>
      <span className={classes.divider} aria-hidden />
      <MenuBar />
      <span className={classes.divider} aria-hidden />
      <ScenariosMenu />
      <GcpButton />
      <MeasureMenu />
      <RawToggle />
      <div className={`${classes.headerActions} dw-no-print`}>
        <Button component="a" href={DOCS_URL} target="_blank" rel="noopener" size="compact-md" variant="subtle" color="gray" leftSection={<IconBook size={16} stroke={1.6} />}>
          Docs
        </Button>
        <span className={classes.divider} style={{ marginInline: 8 }} aria-hidden />
        <BackendStatus />
        <span className={classes.divider} style={{ marginInline: 8 }} aria-hidden />
        <HeaderToggle label="Project panel ([)" pressed={projectOpen} onClick={() => useUi.getState().set({ projectOpen: !projectOpen })}>
          <IconLayoutSidebar size={18} stroke={1.6} />
        </HeaderToggle>
        <HeaderToggle label="Inspector (])" pressed={inspectorOpen} onClick={() => useUi.getState().set({ inspectorOpen: !inspectorOpen })}>
          <IconLayoutSidebarRight size={18} stroke={1.6} />
        </HeaderToggle>
      </div>
    </header>
  );
}
