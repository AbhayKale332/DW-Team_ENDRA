import { Alert } from '@mantine/core';
import { IconAlertTriangle } from '@tabler/icons-react';
import { useScene } from '@/store/scene';

/** Prominent, dismissible correctness warnings (e.g. "check the declared GSD"). Info-level notes live in the Info tab. */
export function SceneWarnings() {
  const scene = useScene((s) => s.scene);
  const dismissed = useScene((s) => s.dismissed);
  const running = useScene((s) => s.run.status === 'running');
  if (!scene || running) return null;
  const items = scene.warnings.filter((w) => w.level === 'warning' && !dismissed.includes(w.id));
  if (!items.length) return null;
  return (
    <>
      {items.map((w) => (
        <Alert
          key={w.id}
          color="dwOrange"
          variant="filled"
          icon={<IconAlertTriangle size={18} />}
          title={w.title}
          withCloseButton
          closeButtonLabel="Dismiss warning"
          onClose={() => useScene.getState().dismiss(w.id)}
          styles={{ root: { boxShadow: 'var(--dw-float-shadow)' } }}
        >
          {w.message}
        </Alert>
      ))}
    </>
  );
}
