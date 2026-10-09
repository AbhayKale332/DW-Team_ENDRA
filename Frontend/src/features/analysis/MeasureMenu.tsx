import { ActionIcon, Menu, Text, Tooltip } from '@mantine/core';
import { IconRuler } from '@tabler/icons-react';
import { useScene } from '@/store/scene';
import { useCamera } from '@/store/camera';
import { useUseCases } from '@/store/usecases';
import { useGcp } from '@/features/gcp/gcpStore';
import { setCameraMode } from '@/features/viewport/overlays/NavigationHud';
import { MEASURE_MODES, MEASURE_MODE_LABELS, startMeasure, stopMeasure, useTool, type MeasureMode } from '@/store/tool';
import { MEASURE_MODE_ICONS } from './MeasurePanel';

const MODE_HELP: Record<MeasureMode, string> = {
  length: 'Distance along a line of points',
  area: 'Area and perimeter of a polygon',
  height: 'Height at a point, or Δh between two',
};

/** Measuring needs the orbit camera (clicks pick points there), so a fly/walk/tour mode is left first. A waiting
 *  GCP / tower / flood-source pick would take the clicks, so it is disarmed. */
export function openMeasure(mode: MeasureMode) {
  if (useCamera.getState().mode !== 'orbit') setCameraMode('orbit');
  if (useGcp.getState().picking) useGcp.setState({ picking: false });
  const uc = useUseCases.getState();
  if (uc.placing || uc.pickingSource) uc.set({ placing: false, pickingSource: false });
  startMeasure(mode);
}

/** M: measure in the last mode, or stop. */
export function toggleMeasure() {
  const st = useTool.getState();
  if (st.tool === 'measure') stopMeasure();
  else openMeasure(st.measureMode);
}

/** Header ruler: picks Length / Area / Height. Tinted while measuring; disabled (with the reason) without a scene. */
export function MeasureMenu() {
  const hasScene = useScene((s) => !!s.scene);
  const active = useTool((s) => s.tool === 'measure');
  const mode = useTool((s) => s.measureMode);
  const tip = !hasScene ? 'Measure · open an image or sample first' : active ? `Measuring ${MEASURE_MODE_LABELS[mode].toLowerCase()} (M)` : 'Measure length, area or height (M)';
  return (
    <Menu position="bottom" withinPortal width={250}>
      <Tooltip label={tip}>
        {/* the wrapper keeps the tooltip working while the button is disabled */}
        <span style={{ display: 'inline-flex' }}>
          <Menu.Target>
            <ActionIcon size="lg" variant={active ? 'light' : 'subtle'} color={active ? 'dwBlue' : 'gray'} disabled={!hasScene} data-tour="measure" aria-label="Measure" aria-pressed={active}>
              <IconRuler size={18} stroke={1.6} />
            </ActionIcon>
          </Menu.Target>
        </span>
      </Tooltip>
      <Menu.Dropdown>
        <Menu.Label>Measure</Menu.Label>
        {MEASURE_MODES.map((m) => {
          const Icon = MEASURE_MODE_ICONS[m];
          const on = active && mode === m;
          return (
            <Menu.Item key={m} leftSection={<Icon size={16} stroke={1.7} />} color={on ? 'dwBlue' : undefined} onClick={() => openMeasure(m)} aria-current={on || undefined}>
              <Text size="sm" fw={on ? 600 : 500} c={on ? 'dwBlue' : undefined}>
                {MEASURE_MODE_LABELS[m]}
              </Text>
              <Text size="xs" c="dimmed">
                {MODE_HELP[m]}
              </Text>
            </Menu.Item>
          );
        })}
        {active && (
          <>
            <Menu.Divider />
            <Menu.Item onClick={stopMeasure}>Stop measuring</Menu.Item>
          </>
        )}
      </Menu.Dropdown>
    </Menu>
  );
}
