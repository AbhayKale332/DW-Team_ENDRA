import { useHotkeys } from '@mantine/hooks';
import { useScene } from '@/store/scene';
import { useUi } from '@/store/ui';
import { toggleAllObjects } from '@/store/view';
import { useCamera, viewportApi } from '@/store/camera';
import { OPEN_ACCEPT, newProject, openFiles, pickFiles } from '@/features/files/openFile';
import { saveProject } from '@/features/files/project';
import { runExport } from '@/features/files/exports';
import { setViewMode } from '@/features/viewport/overlays/ViewSwitcher';
import { setCameraMode } from '@/features/viewport/overlays/NavigationHud';

export interface Shortcut {
  keys: string;
  label: string;
  group: 'File' | 'View' | 'Navigation' | 'Tools' | 'Help';
}

/** Single source of truth for the Keyboard Shortcuts dialog. */
export const SHORTCUTS: Shortcut[] = [
  { keys: 'Ctrl + O', label: 'Open image, project or result bundle', group: 'File' },
  { keys: 'Ctrl + S', label: 'Save project', group: 'File' },
  { keys: 'Ctrl + E', label: 'Export GeoTIFF', group: 'File' },
  { keys: 'Alt + N', label: 'New project', group: 'File' },
  { keys: '1', label: '3D DSM view', group: 'View' },
  { keys: '2', label: 'Height map', group: 'View' },
  { keys: '3', label: 'Input image', group: 'View' },
  { keys: '[', label: 'Toggle project panel', group: 'View' },
  { keys: ']', label: 'Toggle inspector', group: 'View' },
  { keys: 'Ctrl + K', label: 'Command palette', group: 'View' },
  { keys: 'R', label: 'Reset camera', group: 'Navigation' },
  { keys: 'N', label: 'Face north', group: 'Navigation' },
  { keys: '+ / −', label: 'Zoom in / out', group: 'Navigation' },
  { keys: 'Arrow keys', label: 'Orbit (canvas focused)', group: 'Navigation' },
  { keys: 'F', label: 'First-person walk', group: 'Navigation' },
  { keys: 'G', label: 'Flight simulator', group: 'Navigation' },
  { keys: 'T', label: 'Drone tour', group: 'Navigation' },
  { keys: 'Esc', label: 'Leave navigation mode', group: 'Navigation' },
  { keys: 'O', label: 'All 3D objects off / back to your selection', group: 'Tools' },
  { keys: '?', label: 'Keyboard shortcuts', group: 'Help' },
];

export async function openAny() {
  const files = await pickFiles(OPEN_ACCEPT, true);
  if (files.length) await openFiles(files);
}

const hasScene = () => !!useScene.getState().scene;
const orbitOnly = () => useCamera.getState().mode === 'orbit';

/** Global keyboard shortcuts (inactive while typing in inputs). */
export function useGlobalHotkeys() {
  useHotkeys(
    [
      ['mod+O', () => void openAny()],
      ['mod+S', () => hasScene() && void saveProject()],
      ['mod+E', () => hasScene() && void runExport('geotiff')],
      ['alt+N', newProject],
      ['1', () => setViewMode('dsm3d')],
      ['2', () => setViewMode('heightmap')],
      ['3', () => setViewMode('image')],
      ['[', () => useUi.getState().set({ projectOpen: !useUi.getState().projectOpen })],
      [']', () => useUi.getState().set({ inspectorOpen: !useUi.getState().inspectorOpen })],
      ['R', () => orbitOnly() && viewportApi.reset()],
      ['N', () => viewportApi.faceNorth()],
      ['equal', () => viewportApi.zoom(1)],
      ['shift+equal', () => viewportApi.zoom(1)],
      ['minus', () => viewportApi.zoom(-1)],
      ['F', () => hasScene() && setCameraMode(useCamera.getState().mode === 'walk' ? 'orbit' : 'walk')],
      ['G', () => hasScene() && setCameraMode(useCamera.getState().mode === 'flight' ? 'orbit' : 'flight')],
      ['T', () => hasScene() && setCameraMode(useCamera.getState().mode === 'tour' ? 'orbit' : 'tour')],
      ['Escape', () => useCamera.getState().mode !== 'orbit' && !document.pointerLockElement && setCameraMode('orbit')],
      ['O', () => !!useScene.getState().scene?.objects && toggleAllObjects()],
      ['shift+slash', () => useUi.getState().set({ dialog: 'shortcuts' })],
    ],
    ['INPUT', 'TEXTAREA', 'SELECT'],
  );
}

/** Arrow-key orbit when the canvas itself has focus (keyboard-only navigation). */
export function onCanvasKeyDown(e: React.KeyboardEvent) {
  if (useCamera.getState().mode !== 'orbit') return;
  const map: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
  const d = map[e.key];
  if (!d) return;
  e.preventDefault();
  viewportApi.orbitKey(d[0], d[1]);
}
