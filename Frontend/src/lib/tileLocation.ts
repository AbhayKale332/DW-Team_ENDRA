import type { Scene } from '@/domain/types';
import { lonLatAt } from './georef';

/** Only publish map links for a centre that converts to valid WGS84 coordinates. */
export function tileCentre(scene: Pick<Scene, 'georef' | 'heights'>): [number, number] | null {
  if (!scene.georef) return null;
  const ll = lonLatAt(scene.georef, (scene.heights.width - 1) / 2, (scene.heights.height - 1) / 2);
  return ll && ll.every(Number.isFinite) && Math.abs(ll[0]) <= 180 && Math.abs(ll[1]) <= 90 ? ll : null;
}

/** Enclosing administrative areas give a general location, rather than a nearby building's address. */
export function tileLocationQuery([lon, lat]: [number, number]): string {
  return `[out:json][timeout:15];is_in(${lat.toFixed(6)},${lon.toFixed(6)})->.areas;rel(pivot.areas)["boundary"="administrative"];out tags;`;
}

export function parseTileLocation(json: unknown): string | null {
  const elements = (json as { elements?: unknown } | null)?.elements;
  if (!Array.isArray(elements)) return null;
  const areas = elements.flatMap((e: { tags?: Record<string, unknown> } | null) => {
    const t = e?.tags;
    const name = t?.['name:en'] ?? t?.name;
    const level = Number(t?.admin_level);
    return t?.boundary === 'administrative' && typeof name === 'string' && name.trim() && Number.isInteger(level) && level >= 2 && level <= 10
      ? [{ name: name.trim(), level }]
      : [];
  }).sort((a, b) => b.level - a.level || a.name.localeCompare(b.name));
  const local = areas.find((a) => a.level >= 6);
  const region = areas.find((a) => a.level >= 3 && a.level <= 5);
  const country = areas.find((a) => a.level === 2);
  const names = [local, region, country].flatMap((a) => a ? [a.name] : []);
  return [...new Set(names)].join(', ') || null;
}
