import { ActionIcon, Tooltip, useComputedColorScheme, useMantineColorScheme } from '@mantine/core';
import { IconLayoutSidebar, IconLayoutSidebarRight, IconMoon, IconSun } from '@tabler/icons-react';
import { useUi } from '@/store/ui';
import { BrandMark } from './Brand';
import { MenuBar } from './MenuBar';
import { BackendStatus } from './BackendStatus';
import classes from './shell.module.css';

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

export function AppHeader() {
  const { setColorScheme } = useMantineColorScheme();
  const scheme = useComputedColorScheme('light');
  const projectOpen = useUi((s) => s.projectOpen);
  const inspectorOpen = useUi((s) => s.inspectorOpen);
  const nextScheme = scheme === 'dark' ? 'light' : 'dark';
  return (
    <header className={classes.header}>
      <div className={classes.brand}>
        <BrandMark size={30} />
        <span className={classes.wordmark}>DepthWizard</span>
      </div>
      <span className={classes.divider} aria-hidden />
      <MenuBar />
      <div className={`${classes.headerActions} dw-no-print`}>
        <BackendStatus />
        <span className={classes.divider} style={{ marginInline: 8 }} aria-hidden />
        <HeaderToggle label="Project panel ([)" pressed={projectOpen} onClick={() => useUi.getState().set({ projectOpen: !projectOpen })}>
          <IconLayoutSidebar size={18} stroke={1.6} />
        </HeaderToggle>
        <HeaderToggle label="Inspector (])" pressed={inspectorOpen} onClick={() => useUi.getState().set({ inspectorOpen: !inspectorOpen })}>
          <IconLayoutSidebarRight size={18} stroke={1.6} />
        </HeaderToggle>
        <Tooltip label={`${nextScheme === 'dark' ? 'Dark' : 'Light'} theme`}>
          <ActionIcon size="lg" onClick={() => setColorScheme(nextScheme)} aria-label={`Switch to ${nextScheme} theme`}>
            {scheme === 'dark' ? <IconSun size={18} stroke={1.6} /> : <IconMoon size={18} stroke={1.6} />}
          </ActionIcon>
        </Tooltip>
      </div>
    </header>
  );
}
