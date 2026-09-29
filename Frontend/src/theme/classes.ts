import type { ClassMap } from '@/domain/types';

/** Object-class palette, keyed by class *name* so backend id order can change.
 *  Matches the Space's `viz/classes.py` previews, so both read the same. */
export const CLASS_COLORS: Record<string, string> = {
  other: '#9aa0a6',
  ground: '#c8b58a',
  low_veg: '#9ccc65',
  building: '#e05a47',
  water: '#3a8fd9',
  road: '#5f6368',
  tree: '#2e7d32',
};
const UNKNOWN = '#9aa0a6';

export const CLASS_LABELS: Record<string, string> = {
  other: 'Other',
  ground: 'Ground',
  low_veg: 'Low vegetation',
  building: 'Building',
  water: 'Water',
  road: 'Road',
  tree: 'Tree',
};

export function classColor(name: string | undefined): string {
  return (name && CLASS_COLORS[name]) || UNKNOWN;
}

export function classLabel(name: string | undefined, id: number): string {
  return (name && (CLASS_LABELS[name] ?? name)) || `Class ${id}`;
}

/** 256-entry RGBA palette indexed by class id, for the terrain shader. */
export function classPaletteRGBA(names: string[]): Uint8Array {
  const out = new Uint8Array(256 * 4);
  for (let id = 0; id < 256; id++) {
    const hex = classColor(names[id]);
    out[id * 4] = parseInt(hex.slice(1, 3), 16);
    out[id * 4 + 1] = parseInt(hex.slice(3, 5), 16);
    out[id * 4 + 2] = parseInt(hex.slice(5, 7), 16);
    out[id * 4 + 3] = 255;
  }
  return out;
}

/** meta.json `classes` ({"0": "other", ...}) -> names indexed by id. */
export function classNamesFromMeta(classes: Record<string, string> | undefined): string[] {
  const names: string[] = [];
  for (const [k, v] of Object.entries(classes ?? {})) {
    const id = Number(k);
    if (Number.isInteger(id) && id >= 0 && id < 256) names[id] = String(v);
  }
  return names;
}

/** Share of pixels per class id present in the map, largest first. */
export function classShares(map: ClassMap): Array<{ id: number; share: number }> {
  const counts = new Uint32Array(256);
  for (let i = 0; i < map.data.length; i++) counts[map.data[i]]++;
  const out: Array<{ id: number; share: number }> = [];
  counts.forEach((n, id) => n > 0 && out.push({ id, share: n / map.data.length }));
  return out.sort((a, b) => b.share - a.share);
}
