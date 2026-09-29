import { Kbd, Menu } from '@mantine/core';
import { IconBox, IconChevronDown, IconMap2, IconPhoto, IconCheck } from '@tabler/icons-react';
import { useView, VIEW_LABELS, type ViewMode } from '@/store/view';
import { useCamera } from '@/store/camera';
import classes from './overlays.module.css';

export const VIEW_ICONS: Record<ViewMode, typeof IconBox> = { dsm3d: IconBox, heightmap: IconMap2, image: IconPhoto };
const ORDER: ViewMode[] = ['dsm3d', 'heightmap', 'image'];
const MENU_LABELS: Record<ViewMode, string> = { dsm3d: '3D Height Map (DSM)', heightmap: 'Height Map', image: 'Input Image' };

export function setViewMode(mode: ViewMode) {
  useView.getState().set({ mode });
  if (mode !== 'dsm3d') useCamera.getState().set({ mode: 'orbit' });
}

/** Quick-access view switcher floating over the viewport (UI.md "3D DSM View ▾"). */
export function ViewSwitcher() {
  const mode = useView((s) => s.mode);
  const Icon = VIEW_ICONS[mode];
  return (
    <Menu position="bottom-start" offset={6} width={260}>
      <Menu.Target>
        <button type="button" className={`dw-float ${classes.switcher}`} aria-label={`Primary view: ${VIEW_LABELS[mode]}. Change view`}>
          <Icon size={22} stroke={1.6} aria-hidden />
          <span style={{ flex: 1, textAlign: 'left' }}>{VIEW_LABELS[mode]}</span>
          <IconChevronDown size={18} stroke={1.8} aria-hidden />
        </button>
      </Menu.Target>
      <Menu.Dropdown>
        <Menu.Label>Primary visualisation</Menu.Label>
        {ORDER.map((m, i) => {
          const I = VIEW_ICONS[m];
          return (
            <Menu.Item
              key={m}
              leftSection={<I size={16} stroke={1.7} />}
              rightSection={m === mode ? <IconCheck size={14} /> : <Kbd size="xs">{i + 1}</Kbd>}
              onClick={() => setViewMode(m)}
              aria-current={m === mode}
            >
              {MENU_LABELS[m]}
            </Menu.Item>
          );
        })}
      </Menu.Dropdown>
    </Menu>
  );
}
