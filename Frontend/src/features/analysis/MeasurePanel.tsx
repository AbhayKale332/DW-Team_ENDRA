import { useEffect, useMemo } from 'react';
import { ActionIcon, Button, Center, CopyButton, Group, SegmentedControl, Text, Tooltip } from '@mantine/core';
import { IconAlertTriangle, IconArrowBackUp, IconArrowsVertical, IconCheck, IconCopy, IconLine, IconPolygon, IconRuler, IconTrash, IconX } from '@tabler/icons-react';
import { KeyValueRows } from '@/components/panel';
import { measureText } from '@/lib/measure';
import { useScene } from '@/store/scene';
import { useCamera } from '@/store/camera';
import { MEASURE_MIN_POINTS, MEASURE_MODE_LABELS, clearMeasure, finishMeasure, startMeasure, stopMeasure, undoMeasurePoint, useTool, type MeasureMode } from '@/store/tool';
import { measureHint, measureReadout } from './measureReadout';
import classes from '@/features/viewport/overlays/overlays.module.css';

export const MEASURE_MODE_ICONS: Record<MeasureMode, typeof IconLine> = { length: IconLine, area: IconPolygon, height: IconArrowsVertical };

// the mode switch is a radio group: focus on it is not typing
const isTyping = (t: HTMLElement | null) =>
  !!t && (t.isContentEditable || /^(TEXTAREA|SELECT)$/.test(t.tagName) || (t.tagName === 'INPUT' && !/^(radio|checkbox|button)$/.test((t as HTMLInputElement).type)));

/** Enter finishes, Backspace undoes, Esc clears (and on an empty measurement, leaves the tool). Only while measuring
 *  in orbit, never while typing or inside a menu or dialog; a key that did nothing is left to the page. Runs after the
 *  global hotkeys, which mark Esc as handled even when they ignore it, so `defaultPrevented` is not a signal here. */
function useMeasureKeys(active: boolean) {
  useEffect(() => {
    if (!active) return;
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (e.ctrlKey || e.metaKey || e.altKey || isTyping(t)) return;
      if (t?.closest?.('[role="menu"], [role="dialog"], [role="listbox"]') || useCamera.getState().mode !== 'orbit') return;
      const st = useTool.getState();
      let handled = false;
      if (e.key === 'Escape') {
        if (st.measure.length) clearMeasure();
        else stopMeasure();
        handled = true;
      } else if (e.key === 'Backspace' && st.measure.length) {
        undoMeasurePoint();
        handled = true;
      } else if (e.key === 'Enter' && !t?.closest?.('[data-measure-panel]')) {
        // the panel's own buttons keep Enter; anywhere else it finishes the shape
        handled = finishMeasure();
      }
      if (handled) e.preventDefault();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [active]);
}

/** Floating read-out of the measure tool (header → ruler): mode switch, what to do next, the result, and actions. */
export function MeasurePanel() {
  const scene = useScene((s) => s.scene);
  const active = useTool((s) => s.tool === 'measure');
  const mode = useTool((s) => s.measureMode);
  const pts = useTool((s) => s.measure);
  const done = useTool((s) => s.measureDone);
  useMeasureKeys(active && !!scene);
  const readout = useMemo(() => (active && scene ? measureReadout(scene, mode, pts) : null), [active, scene, mode, pts]);
  if (!active || !scene || !readout) return null;

  const label = MEASURE_MODE_LABELS[mode];
  const rows = readout.primary ? [readout.primary, ...readout.rows] : [];
  const copy = measureText(`DepthWizard · ${label} · ${scene.name}`, rows, readout.notes);
  const canFinish = mode !== 'height' && !done && pts.length >= MEASURE_MIN_POINTS[mode];
  return (
    <div className={`dw-float ${classes.measureCard}`} role="region" aria-label="Measure" data-measure-panel>
      <Group justify="space-between" wrap="nowrap" gap={6}>
        <Group gap={6} wrap="nowrap">
          <IconRuler size={16} stroke={1.7} aria-hidden />
          <Text size="sm" fw={600}>
            Measure
          </Text>
        </Group>
        <Tooltip label="Stop measuring (Esc twice)">
          <ActionIcon size="sm" variant="subtle" color="gray" onClick={stopMeasure} aria-label="Stop measuring">
            <IconX size={15} />
          </ActionIcon>
        </Tooltip>
      </Group>
      <SegmentedControl
        fullWidth
        size="xs"
        color="dwBlue"
        value={mode}
        onChange={(v) => startMeasure(v as MeasureMode)}
        aria-label="Measure mode"
        data={(['length', 'area', 'height'] as MeasureMode[]).map((m) => {
          const Icon = MEASURE_MODE_ICONS[m];
          return {
            value: m,
            label: (
              <Center style={{ gap: 5 }}>
                <Icon size={14} stroke={1.8} aria-hidden />
                <span>{MEASURE_MODE_LABELS[m]}</span>
              </Center>
            ),
          };
        })}
      />
      <div className={classes.measureHint} aria-live="polite">
        {measureHint(mode, pts.length, done)}
      </div>
      {readout.primary && (
        <div aria-live="polite">
          <div className={classes.measurePrimary}>
            <Text size="xs" c="dimmed">
              {readout.primary[0]}
              {mode === 'area' && !done ? ' · open' : ''}
            </Text>
            <span className={classes.measureValue}>{readout.primary[1]}</span>
          </div>
          <KeyValueRows rows={readout.rows} />
        </div>
      )}
      {readout.notes.map((n) => (
        <Group key={n} gap={6} wrap="nowrap" align="flex-start">
          <IconAlertTriangle size={14} stroke={1.7} color="var(--mantine-color-dwOrange-6)" style={{ flex: 'none', marginTop: 1 }} aria-hidden />
          <Text size="xs" c="dimmed" lh={1.35}>
            {n}
          </Text>
        </Group>
      ))}
      <Group gap={4} justify="space-between" wrap="nowrap">
        <Group gap={2} wrap="nowrap">
          <Tooltip label="Undo last point (Backspace)">
            <ActionIcon variant="subtle" color="gray" onClick={undoMeasurePoint} disabled={!pts.length} aria-label="Undo last point">
              <IconArrowBackUp size={16} stroke={1.7} />
            </ActionIcon>
          </Tooltip>
          <Tooltip label="Clear (Esc)">
            <ActionIcon variant="subtle" color="gray" onClick={clearMeasure} disabled={!pts.length} aria-label="Clear measurement">
              <IconTrash size={16} stroke={1.7} />
            </ActionIcon>
          </Tooltip>
        </Group>
        <Group gap={6} wrap="nowrap">
          {canFinish && (
            <Button size="compact-xs" variant="light" color="dwBlue" onClick={finishMeasure}>
              {mode === 'area' ? 'Close polygon' : 'Finish'}
            </Button>
          )}
          <CopyButton value={copy} timeout={1500}>
            {({ copied, copy: doCopy }) => (
              <Button
                size="compact-xs"
                variant={copied ? 'light' : 'subtle'}
                color={copied ? 'teal' : 'gray'}
                leftSection={copied ? <IconCheck size={13} /> : <IconCopy size={13} />}
                onClick={doCopy}
                disabled={!readout.primary}
                aria-label="Copy result to clipboard"
              >
                {copied ? 'Copied' : 'Copy'}
              </Button>
            )}
          </CopyButton>
        </Group>
      </Group>
    </div>
  );
}
