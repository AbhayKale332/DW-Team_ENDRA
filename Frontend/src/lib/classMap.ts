import type { ClassMap, SceneMeta } from '@/domain/types';
import { decodeGray8Png } from './png';
import { classNamesFromMeta } from '@/theme/classes';

/** `seg.png` bytes + meta.json legend -> ClassMap on the height grid, or null if absent, unreadable or misaligned.
 *  The class layer is an extra: any problem here disables it rather than failing the scene. */
export function classMapFromPng(buf: ArrayBuffer | null, meta: SceneMeta, width: number, height: number): ClassMap | null {
  if (!buf || !meta.classes) return null;
  try {
    const seg = decodeGray8Png(buf);
    if (seg.width !== width || seg.height !== height) {
      console.warn(`seg.png is ${seg.width}×${seg.height}, heights are ${width}×${height}; class layer disabled`);
      return null;
    }
    return { ...seg, names: classNamesFromMeta(meta.classes) };
  } catch (e) {
    console.warn('Could not decode seg.png; class layer disabled', e);
    return null;
  }
}

/** Fetch an optional file; null on any failure (older backends and samples do not publish it). */
export async function fetchOptional(url: string, init?: RequestInit): Promise<ArrayBuffer | null> {
  try {
    const r = await fetch(url, init);
    return r.ok ? await r.arrayBuffer() : null;
  } catch {
    return null;
  }
}
