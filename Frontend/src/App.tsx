import { Text } from '@mantine/core';
import { Dropzone } from '@mantine/dropzone';
import { IconUpload } from '@tabler/icons-react';
import { AppHeader } from '@/features/shell/AppHeader';
import { StatusBar } from '@/features/shell/StatusBar';
import { onCanvasKeyDown, useGlobalHotkeys } from '@/features/shell/commands';
import { ProjectPanel } from '@/features/input/ProjectPanel';
import { InspectorPanel } from '@/features/inspector/InspectorPanel';
import { Viewport } from '@/features/viewport/Viewport';
import { Dialogs } from '@/features/help/Dialogs';
import { openFiles } from '@/features/files/openFile';
import classes from '@/features/shell/shell.module.css';

export default function App() {
  useGlobalHotkeys();
  return (
    <div className={classes.app}>
      <a href="#dw-canvas" className="dw-sr-only">
        Skip to 3D viewport
      </a>
      <AppHeader />
      <main className={classes.body} onKeyDown={onCanvasKeyDown}>
        <ProjectPanel />
        <Viewport />
        <InspectorPanel />
      </main>
      <StatusBar />
      <Dialogs />
      <Dropzone.FullScreen onDrop={(files) => void openFiles(files)} multiple accept={undefined}>
        <div style={{ display: 'grid', placeItems: 'center', height: '100%', gap: 8, pointerEvents: 'none' }}>
          <div style={{ textAlign: 'center' }}>
            <IconUpload size={48} stroke={1.3} />
            <Text size="lg" fw={600}>
              Drop to open
            </Text>
            <Text size="sm" c="dimmed">
              Image (PNG, JPG, GeoTIFF) · project (.dwproj) · result bundle (.zip / .npy)
            </Text>
          </div>
        </div>
      </Dropzone.FullScreen>
    </div>
  );
}
