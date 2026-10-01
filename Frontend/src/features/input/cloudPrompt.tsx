import { Stack, Text } from '@mantine/core';
import { modals } from '@mantine/modals';
import { MIN_COVERAGE } from '@/lib/cloud';
import type { PreparedInput } from '@/lib/input';
import { useScene } from '@/store/scene';
import { analysisWorker } from '@/workers/clients';

/** Look for clouds in a freshly staged input. With `ask`, a likely cloud opens a prompt whose answer sets the
 *  "Cloud mask" switch; without it (a reopened project) the switch keeps its saved state. */
export async function detectInputClouds(input: PreparedInput, { ask }: { ask: boolean }) {
  const current = () => useScene.getState().input === input;
  useScene.getState().set({ inputCloud: { status: 'detecting', mask: null } });
  try {
    const found = await analysisWorker().cloudDetect(input.upload);
    if (!current()) return;
    const mask = found.coverage >= MIN_COVERAGE ? found : null;
    useScene.getState().set({ inputCloud: { status: 'done', mask } });
    if (!mask || !ask) return;
    const preview = await analysisWorker().cloudPreview(input.upload, mask);
    if (!current()) return;
    promptCloudMask(input, URL.createObjectURL(preview), mask.coverage);
  } catch (e) {
    console.warn('Cloud detection failed', e);
    if (current()) useScene.getState().set({ inputCloud: { status: 'error', mask: null } });
  }
}

function promptCloudMask(input: PreparedInput, previewUrl: string, coverage: number) {
  const pct = coverage * 100;
  modals.openConfirmModal({
    title: 'Clouds in this image?',
    centered: true,
    children: (
      <Stack gap="sm">
        <img
          src={previewUrl}
          alt="The image with the areas that look like cloud tinted cyan"
          style={{ display: 'block', width: '100%', maxHeight: 320, objectFit: 'contain', background: 'var(--dw-surface)' }}
        />
        <Text size="sm">
          About <b>{pct < 1 ? pct.toFixed(1) : pct.toFixed(0)} %</b> of {input.name} looks like cloud (tinted cyan). Clouds hide the ground, and the model
          would read them as tall structures.
        </Text>
        <Text size="sm" c="dimmed">
          Masking hides them from the model and fills those areas from the surrounding ground, hatched so they are not mistaken for measurements. You can
          change this later with the <i>Cloud mask</i> switch.
        </Text>
      </Stack>
    ),
    labels: { confirm: 'Mask clouds', cancel: 'Keep as is' },
    onConfirm: () => useScene.getState().setParams({ cloudMask: true }),
    onCancel: () => useScene.getState().setParams({ cloudMask: false }),
    onClose: () => URL.revokeObjectURL(previewUrl),
  });
}
