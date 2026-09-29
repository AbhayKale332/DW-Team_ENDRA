import { Button, Group, Menu, Stack, Text, Title } from '@mantine/core';
import { IconChevronDown, IconFolderOpen, IconSparkles } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { SAMPLES } from '@/lib/samples';
import { IMAGE_ACCEPT, openSample, pickFiles, stageImage } from '@/features/files/openFile';
import { BrandMark } from '@/features/shell/Brand';
import classes from './overlays.module.css';

/** First-run guidance shown in the viewport until a result exists. */
export function EmptyState() {
  const hasScene = useScene((s) => !!s.scene);
  const hasInput = useScene((s) => !!s.input);
  const running = useScene((s) => s.run.status === 'running');
  if (hasScene || running) return null;
  return (
    <div className={classes.empty}>
      <div className={`dw-float ${classes.emptyCard}`}>
        <Stack gap="md">
          <Group gap="sm" wrap="nowrap">
            <BrandMark size={40} />
            <div>
              <Title order={2} fz={20}>
                {hasInput ? 'Ready to estimate heights' : 'DepthWizard'}
              </Title>
            </div>
          </Group>
          <Text size="sm">
            {hasInput
              ? 'Check the resolution in the Project panel, then run the model.'
              : 'Open a PNG, JPG or GeoTIFF image.'}
          </Text>
          <Group gap="sm">
            {!hasInput && (
              <Button leftSection={<IconFolderOpen size={16} />} onClick={async () => (await pickFiles(IMAGE_ACCEPT)).slice(0, 1).forEach((f) => void stageImage(f))}>
                Open image…
              </Button>
            )}
            <Menu position="bottom-start" width={320}>
              <Menu.Target>
                <Button variant="default" leftSection={<IconSparkles size={16} />} rightSection={<IconChevronDown size={14} />}>
                  Try a sample scene
                </Button>
              </Menu.Target>
              <Menu.Dropdown>
                {SAMPLES.map((s) => (
                  <Menu.Item key={s.id} onClick={() => void openSample(s.id)}>
                    {s.name}
                  </Menu.Item>
                ))}
              </Menu.Dropdown>
            </Menu>
          </Group>
        </Stack>
      </div>
    </div>
  );
}
