import { Button, Stack, Text, Title } from '@mantine/core';
import { IconAlertTriangle } from '@tabler/icons-react';
import { useScene } from '@/store/scene';

/** Shown when WebGL is unavailable or the context is lost; the 2D input image stays viewable. */
export function WebGLFallback({ message }: { message: string }) {
  const imageUrl = useScene((s) => s.imageUrl);
  return (
    <div style={{ position: 'absolute', inset: 0, display: 'grid', placeItems: 'center', padding: 24 }}>
      {imageUrl && <img src={imageUrl} alt="Input image" style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', objectFit: 'contain', opacity: 0.35 }} />}
      <div className="dw-float" style={{ padding: 24, maxWidth: 460, position: 'relative' }} role="alert">
        <Stack gap="sm">
          <IconAlertTriangle color="var(--mantine-color-dwOrange-6)" />
          <Title order={3} fz={18}>
            3D view unavailable
          </Title>
          <Text size="sm">
            DepthWizard needs WebGL 2 for the 3D terrain. Enable hardware acceleration in your browser settings or try a current Chrome, Edge or Firefox.
          </Text>
          <Text size="xs" c="dimmed" className="dw-mono">
            {message}
          </Text>
          <Button variant="default" onClick={() => window.location.reload()}>
            Reload
          </Button>
        </Stack>
      </div>
    </div>
  );
}
