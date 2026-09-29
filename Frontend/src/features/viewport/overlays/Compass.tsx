import { Tooltip } from '@mantine/core';
import { useCamera, viewportApi } from '@/store/camera';
import classes from './overlays.module.css';

const CARDINALS: Array<[string, number]> = [
  ['N', 0],
  ['E', 90],
  ['S', 180],
  ['W', 270],
];

/** Compass rose that rotates with the camera heading. Click to face north. */
export function Compass() {
  const heading = useCamera((s) => s.heading);
  const rot = -heading;
  const r = 36;
  return (
    <Tooltip label="Face north" position="right">
      <button type="button" className={`dw-float ${classes.compass}`} onClick={() => viewportApi.faceNorth()} aria-label={`Compass: heading ${Math.round(heading)} degrees. Face north`}>
        <svg viewBox="-50 -50 100 100" width="100%" height="100%" aria-hidden>
          <defs>
            <linearGradient id="dw-needle" x1="0" y1="1" x2="0" y2="0">
              <stop offset="0" stopColor="#2d6cdf" />
              <stop offset="1" stopColor="#6a4cf0" />
            </linearGradient>
          </defs>
          <g transform={`rotate(${rot})`}>
            {Array.from({ length: 24 }, (_, i) => (
              <line key={i} x1="0" y1={-44} x2="0" y2={i % 6 === 0 ? -40 : -42} stroke="var(--dw-faint)" strokeWidth={i % 6 === 0 ? 1.4 : 0.8} transform={`rotate(${i * 15})`} />
            ))}
            {CARDINALS.map(([l, a]) => {
              const rad = (a * Math.PI) / 180;
              return (
                <text
                  key={l}
                  x={Math.sin(rad) * r * 0.86}
                  y={-Math.cos(rad) * r * 0.86}
                  transform={`rotate(${-rot} ${Math.sin(rad) * r * 0.86} ${-Math.cos(rad) * r * 0.86})`}
                  textAnchor="middle"
                  dominantBaseline="central"
                  fontSize="11"
                  fontWeight={l === 'N' ? 700 : 600}
                  fill={l === 'N' ? 'var(--dw-ink)' : 'var(--dw-dim)'}
                  fontFamily="Inter, sans-serif"
                >
                  {l}
                </text>
              );
            })}
            <path d="M0,-20 L9,10 L0,5 L-9,10 Z" fill="url(#dw-needle)" />
            <path d="M0,5 L9,10 L0,-20 Z" fill="#000" opacity="0.12" />
          </g>
        </svg>
      </button>
    </Tooltip>
  );
}
