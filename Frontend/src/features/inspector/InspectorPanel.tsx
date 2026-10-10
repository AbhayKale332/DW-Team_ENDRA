import { ActionIcon, ScrollArea, Tabs, Tooltip } from '@mantine/core';
import { IconChevronRight, IconInfoCircle, IconScale, IconStack2 } from '@tabler/icons-react';
import { ErrorBoundary } from 'react-error-boundary';
import { useUi, type InspectorTab } from '@/store/ui';
import { LayersTab } from './LayersTab';
import { InfoTab } from './InfoTab';
import { ValidationTab } from '@/features/validation/ValidationTab';
import classes from '@/features/shell/shell.module.css';

function PanelError({ error }: { error: unknown }) {
  return (
    <div style={{ padding: 'var(--dw-gap-4)', fontSize: 12 }} role="alert">
      This panel failed to render: {String((error as Error)?.message ?? error)}
    </div>
  );
}

/** Right dock: Layers · Validation · Info */
export function InspectorPanel() {
  const open = useUi((s) => s.inspectorOpen);
  const tab = useUi((s) => s.inspectorTab);
  if (!open) return null;
  return (
    <aside className={`${classes.panel} ${classes.panelRight} dw-no-print`} aria-label="Inspector" data-tour="inspector">
      <Tabs value={tab} onChange={(t) => t && useUi.getState().set({ inspectorTab: t as InspectorTab })} style={{ display: 'flex', flexDirection: 'column', minHeight: 0, flex: 1 }}>
        <div className={classes.panelHeader} style={{ paddingLeft: 4 }}>
          <Tabs.List style={{ flexWrap: 'nowrap', borderBottom: 0 }}>
            <Tabs.Tab value="layers" leftSection={<IconStack2 size={14} />}>
              Layers
            </Tabs.Tab>
            <Tabs.Tab value="validation" leftSection={<IconScale size={14} />}>
              Validation
            </Tabs.Tab>
            <Tabs.Tab value="info" leftSection={<IconInfoCircle size={14} />}>
              Info
            </Tabs.Tab>
          </Tabs.List>
          <Tooltip label="Hide inspector (])">
            <ActionIcon onClick={() => useUi.getState().set({ inspectorOpen: false })} aria-label="Hide inspector">
              <IconChevronRight size={16} />
            </ActionIcon>
          </Tooltip>
        </div>
        <ScrollArea style={{ flex: 1 }} type="auto">
          <ErrorBoundary FallbackComponent={PanelError} resetKeys={[tab]}>
            <Tabs.Panel value="layers">
              <LayersTab />
            </Tabs.Panel>
            <Tabs.Panel value="validation">
              <ValidationTab />
            </Tabs.Panel>
            <Tabs.Panel value="info">
              <InfoTab />
            </Tabs.Panel>
          </ErrorBoundary>
        </ScrollArea>
      </Tabs>
    </aside>
  );
}
