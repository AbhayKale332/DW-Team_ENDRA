/** Colormap LUTs (256 × RGB). Stops are piecewise-linear approximations of the published maps. */

export type ColormapId = 'turbo' | 'terrain' | 'viridis' | 'cividis' | 'grey' | 'diverging' | 'slope' | 'confidence';

type Stop = [number, number, number];

const STOPS: Record<ColormapId, Stop[]> = {
  // Turbo, 9-stop (same stops as the v5 viewer / backend figures).
  turbo: [
    [48, 18, 59], [70, 107, 227], [40, 187, 226], [61, 231, 154], [163, 244, 78],
    [231, 215, 53], [253, 152, 39], [232, 79, 13], [122, 4, 3],
  ],
  // Green → yellow → orange → red: the look of the product mockups.
  terrain: [
    [22, 110, 58], [46, 160, 67], [120, 198, 61], [214, 222, 64], [250, 190, 50],
    [244, 122, 36], [220, 50, 32], [160, 20, 30],
  ],
  viridis: [
    [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142], [38, 130, 142],
    [31, 158, 137], [53, 183, 121], [109, 205, 89], [180, 222, 44], [253, 231, 37],
  ],
  cividis: [
    [0, 34, 78], [18, 53, 112], [59, 73, 108], [87, 92, 109], [112, 113, 115],
    [138, 134, 120], [165, 156, 116], [195, 179, 105], [225, 204, 85], [254, 232, 56],
  ],
  grey: [[20, 20, 20], [235, 235, 235]],
  // Blue – white – red for signed error.
  diverging: [[33, 102, 172], [103, 169, 207], [209, 229, 240], [247, 247, 247], [253, 219, 199], [239, 138, 98], [178, 24, 43]],
  // Flat (pale) → steep (deep magenta) for slope in degrees.
  slope: [[255, 255, 229], [254, 227, 145], [254, 153, 41], [217, 95, 14], [153, 52, 4], [102, 37, 6]],
  // Green (confident) → yellow → red (uncertain), as the v5 viewer.
  confidence: [[60, 190, 90], [255, 190, 40], [255, 40, 40]],
};

export const COLORMAPS: Array<{ id: ColormapId; label: string; cvdSafe?: boolean }> = [
  { id: 'turbo', label: 'Turbo' },
  { id: 'terrain', label: 'Terrain' },
  { id: 'viridis', label: 'Viridis', cvdSafe: true },
  { id: 'cividis', label: 'Cividis', cvdSafe: true },
  { id: 'grey', label: 'Greyscale', cvdSafe: true },
];

const cache = new Map<ColormapId, Uint8Array>();

/** 256×4 RGBA LUT. */
export function lut(id: ColormapId): Uint8Array {
  const hit = cache.get(id);
  if (hit) return hit;
  const stops = STOPS[id];
  const out = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (stops.length - 1);
    const k = Math.min(stops.length - 2, Math.floor(t));
    const f = t - k;
    const a = stops[k];
    const b = stops[k + 1];
    out[i * 4] = Math.round(a[0] + (b[0] - a[0]) * f);
    out[i * 4 + 1] = Math.round(a[1] + (b[1] - a[1]) * f);
    out[i * 4 + 2] = Math.round(a[2] + (b[2] - a[2]) * f);
    out[i * 4 + 3] = 255;
  }
  cache.set(id, out);
  return out;
}

export function colorAt(id: ColormapId, t: number): [number, number, number] {
  const l = lut(id);
  const i = Math.max(0, Math.min(255, Math.round(t * 255)));
  return [l[i * 4], l[i * 4 + 1], l[i * 4 + 2]];
}

export function cssGradient(id: ColormapId, direction = 'to top'): string {
  const n = 12;
  const parts: string[] = [];
  for (let i = 0; i <= n; i++) {
    const [r, g, b] = colorAt(id, i / n);
    parts.push(`rgb(${r} ${g} ${b}) ${((i / n) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(${direction}, ${parts.join(', ')})`;
}
