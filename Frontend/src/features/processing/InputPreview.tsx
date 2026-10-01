import { Button, Group, Text, ThemeIcon } from '@mantine/core';
import { IconCheck, IconCube } from '@tabler/icons-react';
import { isPreviewing, useScene } from '@/store/scene';
import classes from '@/features/viewport/overlays/overlays.module.css';

/** The input image over the viewport while a run is in progress; once the 3D result is built, a prompt to explore it. */
export function InputPreview() {
  const previewing = useScene(isPreviewing);
  const url = useScene((s) => s.input?.previewUrl);
  const name = useScene((s) => s.input?.name);
  const ready = useScene((s) => s.run.status === 'done');
  if (!previewing || !url) return null;
  return (
    <div className={classes.preview}>
      <img src={url} alt={name ? `Input image ${name}` : 'Input image'} className={classes.previewImage} />
      {ready && (
        <div className={`dw-float ${classes.previewReady}`} role="status" aria-live="polite">
          <Group gap="sm" wrap="nowrap">
            <ThemeIcon size={26} radius="xl" color="teal" variant="filled">
              <IconCheck size={16} />
            </ThemeIcon>
            <Text fw={600}>Output ready</Text>
            <Button size="sm" leftSection={<IconCube size={16} />} onClick={() => useScene.getState().set({ inputPreview: false })} autoFocus>
              Explore output
            </Button>
          </Group>
        </div>
      )}
    </div>
  );
}
