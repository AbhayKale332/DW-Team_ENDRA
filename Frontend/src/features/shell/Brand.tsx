import { useId } from 'react';

/** Twin-peak gradient mark (from the product mockups). */
export function BrandMark({ size = 32 }: { size?: number }) {
  const id = useId().replace(/:/g, '');
  return (
    <svg width={size} height={size * 0.62} viewBox="0 0 64 40" role="img" aria-label="DepthWizard logo">
      <defs>
        <linearGradient id={`g1${id}`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#2d6cdf" />
          <stop offset="1" stopColor="#4f8df5" />
        </linearGradient>
        <linearGradient id={`g2${id}`} x1="0" y1="1" x2="1" y2="0">
          <stop offset="0" stopColor="#5b4ff0" />
          <stop offset="1" stopColor="#8a5cf6" />
        </linearGradient>
      </defs>
      <path d="M2 38 L20 6 L31 25 L24 38 Z" fill={`url(#g1${id})`} />
      <path d="M18 38 L40 4 L62 38 Z" fill={`url(#g2${id})`} opacity="0.95" />
      <path d="M18 38 L31 18 L38 38 Z" fill="#2d6cdf" opacity="0.55" />
    </svg>
  );
}
