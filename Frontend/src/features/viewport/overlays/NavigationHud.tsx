import { useEffect, useMemo, useState } from 'react';
import { ActionIcon, Button, Group, Kbd, Slider, Text, Tooltip } from '@mantine/core';
import { IconPlayerPause, IconPlayerPlay, IconX } from '@tabler/icons-react';
import { CAMERA_MODE_LABELS, useCamera, type CameraMode } from '@/store/camera';
import { useScene } from '@/store/scene';
import { useView } from '@/store/view';
import { sceneExtent } from '../terrainState';
import classes from './overlays.module.css';

export function setCameraMode(mode: CameraMode) {
  if (mode !== 'orbit') useView.getState().set({ mode: 'dsm3d' });
  if (document.pointerLockElement) document.exitPointerLock();
  useCamera.getState().set({ mode, tourPlaying: true });
}

const HINTS: Partial<Record<CameraMode, Array<[string, string]>>> = {
  walk: [
    ['W A S D', 'move'],
    ['Shift', 'run'],
    ['Space / C', 'rise / lower'],
    ['Mouse', 'look'],
    ['Esc', 'release mouse'],
  ],
  flight: [
    ['W / S', 'throttle'],
    ['↑ / ↓', 'pitch'],
    ['← / →', 'roll & bank'],
    ['Q / E', 'yaw'],
    ['Shift', 'boost'],
    ['R', 'level'],
  ],
};

function ModeBar() {
  const mode = useCamera((s) => s.mode);
  const playing = useCamera((s) => s.tourPlaying);
  const tourSpeed = useCamera((s) => s.tourSpeed);
  if (mode === 'orbit') return null;
  return (
    <div className={`dw-float ${classes.modeBar}`} role="toolbar" aria-label={`${CAMERA_MODE_LABELS[mode]} controls`}>
      <Text size="sm" fw={600}>
        {CAMERA_MODE_LABELS[mode]}
      </Text>
      {mode === 'tour' && (
        <>
          <Tooltip label={playing ? 'Pause' : 'Play'}>
            <ActionIcon variant="default" onClick={() => useCamera.getState().set({ tourPlaying: !playing })} aria-label={playing ? 'Pause tour' : 'Play tour'}>
              {playing ? <IconPlayerPause size={16} /> : <IconPlayerPlay size={16} />}
            </ActionIcon>
          </Tooltip>
          <Slider
            w={110}
            min={0.2}
            max={4}
            step={0.1}
            value={tourSpeed}
            onChange={(v) => useCamera.getState().set({ tourSpeed: v })}
            label={(v) => `${v.toFixed(1)}×`}
            aria-label="Tour speed"
          />
        </>
      )}
      <Button size="compact-sm" variant="default" leftSection={<IconX size={14} />} onClick={() => setCameraMode('orbit')}>
        Exit
      </Button>
    </div>
  );
}

function KeyHints() {
  const mode = useCamera((s) => s.mode);
  const hints = HINTS[mode];
  if (!hints) return null;
  return (
    <div className={`dw-float`} style={{ padding: '8px 10px', display: 'flex', flexWrap: 'wrap', gap: '6px 14px', maxWidth: 520 }} aria-label="Keyboard controls">
      {hints.map(([k, v]) => (
        <Group key={k} gap={6} wrap="nowrap">
          <Kbd size="xs">{k}</Kbd>
          <Text size="xs" c="dimmed">
            {v}
          </Text>
        </Group>
      ))}
    </div>
  );
}

/** Click-to-capture prompt for pointer-lock (first person). */
function PointerLockHint() {
  const mode = useCamera((s) => s.mode);
  const [locked, setLocked] = useState(false);
  useEffect(() => {
    const on = () => setLocked(!!document.pointerLockElement);
    document.addEventListener('pointerlockchange', on);
    return () => document.removeEventListener('pointerlockchange', on);
  }, []);
  if (mode !== 'walk' || locked) return null;
  return (
    <button type="button" id="dw-lock-target" className={classes.lockHint} aria-label="Click to look around in first person">
      <span className="dw-float" style={{ padding: '14px 18px', color: 'var(--dw-ink)' }}>
        <Text fw={600}>Click to look around</Text>
      </span>
    </button>
  );
}

function HeadingTape({ heading }: { heading: number }) {
  const ticks = useMemo(() => {
    const out: Array<{ deg: number; label: string | null }> = [];
    for (let d = -60; d <= 60; d += 5) {
      const a = Math.round((heading + d) / 5) * 5;
      const norm = ((a % 360) + 360) % 360;
      const label = norm % 90 === 0 ? ['N', 'E', 'S', 'W'][norm / 90] : norm % 30 === 0 ? String(norm) : null;
      out.push({ deg: a - heading, label });
    }
    return out;
  }, [heading]);
  return (
    <div className={classes.hudTape} aria-hidden>
      <svg width="320" height="36" viewBox="-160 0 320 36">
        {ticks.map((t, i) => (
          <g key={i} transform={`translate(${t.deg * 2.6},0)`}>
            <line y1="0" y2={t.label ? 10 : 6} stroke="#fff" strokeWidth="1" opacity="0.8" />
            {t.label && (
              <text y="24" textAnchor="middle" fontSize="11" fill="#fff">
                {t.label}
              </text>
            )}
          </g>
        ))}
        <path d="M-6,0 L6,0 L0,8 Z" fill="#ffd166" />
      </svg>
    </div>
  );
}

function FlightHud() {
  const mode = useCamera((s) => s.mode);
  const heading = useCamera((s) => s.heading);
  const f = useCamera((s) => s.flight);
  const scene = useScene((s) => s.scene);
  if (mode !== 'flight' && mode !== 'walk') return null;
  const datum = scene?.product === 'DSM' ? 'elev.' : scene?.product === 'rDSM' ? 'rel. to scene base' : 'above scene base';
  return (
    <div className={classes.hud}>
      <HeadingTape heading={heading} />
      {mode === 'flight' && (
        <>
          <svg className={classes.hudCross} viewBox="-32 -32 64 64" aria-hidden>
            <g transform={`rotate(${-f.roll})`}>
              <line x1="-28" x2="-10" y1="0" y2="0" stroke="#ffd166" strokeWidth="2" />
              <line x1="10" x2="28" y1="0" y2="0" stroke="#ffd166" strokeWidth="2" />
              <circle r="2.5" fill="#ffd166" />
            </g>
          </svg>
          <div className={classes.hudBox} style={{ left: 16, top: '50%', transform: 'translateY(-50%)' }}>
            <div>SPD {(f.speed * 3.6).toFixed(0)} km/h</div>
            <div>PIT {f.pitch.toFixed(0)}°</div>
            <div>ROL {f.roll.toFixed(0)}°</div>
          </div>
        </>
      )}
      <div className={classes.hudBox} style={{ right: 16, top: '50%', transform: 'translateY(-50%)', textAlign: 'right' }}>
        <div>AGL {f.agl.toFixed(1)} m</div>
        <div>
          ALT {f.altitude.toFixed(1)} m <span style={{ opacity: 0.7 }}>({datum})</span>
        </div>
        <div>HDG {Math.round(heading).toString().padStart(3, '0')}°</div>
      </div>
    </div>
  );
}

/** Inset map showing where the flying/walking camera is (echoes the v5 2D camera marker). */
function MiniMap() {
  const mode = useCamera((s) => s.mode);
  const imageUrl = useScene((s) => s.imageUrl);
  const scene = useScene((s) => s.scene);
  const heading = useCamera((s) => s.heading);
  const [x, z] = useCamera((s) => s.camXZ);
  if (mode === 'orbit' || !imageUrl || !scene) return null;
  const e = sceneExtent(scene);
  const u = Math.min(1, Math.max(0, 0.5 + x / e.x));
  const v = Math.min(1, Math.max(0, 0.5 + z / e.z));
  return (
    <div className={`dw-float ${classes.miniMap}`} aria-label="Position map" role="img">
      <img src={imageUrl} alt="" style={{ width: '100%', height: '100%', objectFit: 'fill', display: 'block', opacity: 0.9 }} />
      <svg
        width="22"
        height="22"
        viewBox="-11 -11 22 22"
        style={{ position: 'absolute', left: `calc(${u * 100}% - 11px)`, top: `calc(${v * 100}% - 11px)`, transform: `rotate(${heading}deg)` }}
        aria-hidden
      >
        <path d="M0,-9 L6,7 L0,3 L-6,7 Z" fill="#ffd166" stroke="#111" strokeWidth="1" />
      </svg>
    </div>
  );
}

export function NavigationHud() {
  return (
    <>
      <FlightHud />
      <PointerLockHint />
      <MiniMap />
    </>
  );
}

export { ModeBar, KeyHints };
