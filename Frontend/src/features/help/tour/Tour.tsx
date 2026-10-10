import { useEffect, useLayoutEffect, useRef, useState } from 'react';
import { ActionIcon, Button, FocusTrap, Group, Text } from '@mantine/core';
import { useViewportSize } from '@mantine/hooks';
import { IconX } from '@tabler/icons-react';
import { useUi } from '@/store/ui';
import { TOUR_STEPS, type TourStep } from './steps';
import classes from './tour.module.css';

const GAP = 14; // card ↔ highlight
const MARGIN = 12; // card ↔ window edge
const PAD = 6; // highlight ↔ target

type Box = { top: number; left: number; width: number; height: number };

/** Below this width the side panels are drawers over the viewport (shell.module.css). */
const narrow = () => window.matchMedia('(max-width: 72em)').matches;

const findTarget = (name: string) => document.querySelector<HTMLElement>(`[data-tour="${name}"]`);

export function startTour() {
  useUi.getState().set({ tourOpen: true });
}

const endTour = () => useUi.getState().set({ tourOpen: false });

/** Card position: the preferred side, then its opposite, then the rest; inside the target when none has room. */
function placeCard(r: Box | null, w: number, h: number, side: TourStep['side'], vw: number, vh: number) {
  const clamp = (top: number, left: number) => ({
    top: Math.min(Math.max(top, MARGIN), vh - h - MARGIN),
    left: Math.min(Math.max(left, MARGIN), vw - w - MARGIN),
  });
  if (!r) return clamp((vh - h) / 2, (vw - w) / 2);
  const cx = r.left + r.width / 2 - w / 2;
  const cy = r.top + r.height / 2 - h / 2;
  const fits = {
    bottom: r.top + r.height + GAP + h <= vh - MARGIN,
    top: r.top - GAP - h >= MARGIN,
    right: r.left + r.width + GAP + w <= vw - MARGIN,
    left: r.left - GAP - w >= MARGIN,
  };
  const opposite = {
    top: 'bottom',
    bottom: 'top',
    left: 'right',
    right: 'left',
  } as const;
  const first = side ?? 'bottom';
  const order = [first, opposite[first], ...(['bottom', 'right', 'left', 'top'] as const)];
  for (const s of order) {
    if (!fits[s]) continue;
    if (s === 'bottom') return clamp(r.top + r.height + GAP, cx);
    if (s === 'top') return clamp(r.top - GAP - h, cx);
    if (s === 'right') return clamp(cy, r.left + r.width + GAP);
    return clamp(cy, r.left - GAP - w);
  }
  return clamp(r.top + r.height - h - GAP * 2, cx);
}

function TourRunner() {
  const [steps, setSteps] = useState<TourStep[] | null>(null);
  const [index, setIndex] = useState(0);
  const [rect, setRect] = useState<Box | null>(null);
  const [card, setCard] = useState({ w: 340, h: 200 });
  const cardRef = useRef<HTMLDivElement>(null);
  const nextRef = useRef<HTMLButtonElement>(null);
  const { width: vw, height: vh } = useViewportSize();

  // Panels come back as they were when the tour ends. Steps whose target is hidden (and no panel will reveal it) are dropped.
  useEffect(() => {
    const { projectOpen, inspectorOpen } = useUi.getState();
    useUi.getState().set({ dialog: null, ...(narrow() ? {} : { projectOpen: true, inspectorOpen: true }) });
    const id = requestAnimationFrame(() =>
      setSteps(
        TOUR_STEPS.filter((s) => {
          if (!s.target || s.panel) return true;
          const r = findTarget(s.target)?.getBoundingClientRect();
          return !!r && r.width > 0 && r.height > 0;
        }),
      ),
    );
    return () => {
      cancelAnimationFrame(id);
      useUi.getState().set({ projectOpen, inspectorOpen });
    };
  }, []);

  const step = steps?.[index];
  const last = !!steps && index === steps.length - 1;
  const next = () => (last ? endTour() : setIndex((i) => i + 1));
  const back = () => setIndex((i) => Math.max(0, i - 1));

  // Follow the target every frame: panels open, scroll, resize and animate while the tour is open.
  useLayoutEffect(() => {
    // Narrow screens float the panels over the map, so only the one this step needs stays open.
    if (step && narrow()) useUi.getState().set({ projectOpen: step.panel === 'project', inspectorOpen: step.panel === 'inspector' });
    let raf = 0;
    let key = '';
    let scrolled = false;
    const tick = () => {
      const el = step?.target ? findTarget(step.target) : null;
      if (el && !scrolled) {
        scrolled = true;
        el.scrollIntoView({ block: 'nearest', inline: 'nearest' });
      }
      const r = el?.getBoundingClientRect();
      const k = r ? `${r.top},${r.left},${r.width},${r.height}` : '';
      if (k !== key) {
        key = k;
        setRect(r && r.width > 0 ? { top: r.top, left: r.left, width: r.width, height: r.height } : null);
      }
      raf = requestAnimationFrame(tick);
    };
    tick();
    return () => cancelAnimationFrame(raf);
  }, [step]);

  useLayoutEffect(() => {
    const el = cardRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => setCard({ w: el.offsetWidth, h: el.offsetHeight }));
    ro.observe(el);
    return () => ro.disconnect();
  }, [steps]);

  useEffect(() => nextRef.current?.focus(), [index, steps]);

  // Captured so the app's own hotkeys (Escape, arrows) don't also fire.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const act = { ArrowRight: next, ArrowLeft: back, Escape: endTour }[e.key];
      if (!act) return;
      e.preventDefault();
      e.stopPropagation();
      act();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  });

  if (!steps || !step) return null;
  const pos = placeCard(rect, card.w, card.h, step.side, vw || window.innerWidth, vh || window.innerHeight);
  const hole = rect && {
    top: rect.top - PAD,
    left: rect.left - PAD,
    width: rect.width + PAD * 2,
    height: rect.height + PAD * 2,
  };

  return (
    <div className={classes.root}>
      {hole ? <div className={classes.spotlight} style={hole} aria-hidden /> : <div className={classes.backdrop} aria-hidden />}
      <FocusTrap active>
        <div ref={cardRef} className={classes.card} style={pos} role="dialog" aria-modal="true" aria-labelledby="dw-tour-title" aria-describedby="dw-tour-body">
          <div className={classes.progress} aria-hidden>
            <span style={{ width: `${((index + 1) / steps.length) * 100}%` }} />
          </div>
          <Group justify="space-between" wrap="nowrap" gap="xs" mb={6}>
            <Text size="xs" c="dimmed" fw={500}>
              {index + 1} of {steps.length}
            </Text>
            <ActionIcon size="sm" onClick={endTour} aria-label="End tour">
              <IconX size={14} />
            </ActionIcon>
          </Group>
          <Text id="dw-tour-title" fw={600} fz={16} mb={6} c="var(--dw-ink)">
            {step.title}
          </Text>
          <Text id="dw-tour-body" size="sm" c="var(--dw-dim)" lh={1.55} aria-live="polite">
            {step.body}
          </Text>
          <Group justify="space-between" wrap="nowrap" mt="md" gap="xs">
            <Button variant="subtle" color="gray" size="compact-sm" onClick={endTour}>
              End tour
            </Button>
            <Group gap={6} wrap="nowrap">
              {index > 0 && (
                <Button variant="default" size="compact-sm" onClick={back}>
                  Previous
                </Button>
              )}
              <Button ref={nextRef} size="compact-sm" onClick={next}>
                {index === 0 ? 'Start tour' : last ? 'Finish' : 'Next'}
              </Button>
            </Group>
          </Group>
        </div>
      </FocusTrap>
    </div>
  );
}

/** Guided tour: dims the app, highlights one section at a time and explains it in a card. Started from Help. */
export function Tour() {
  const open = useUi((s) => s.tourOpen);
  return open ? <TourRunner /> : null;
}
