import { create } from 'zustand';

/** A sample scene: a .dwproj in a folder under public/samples, listed by samples/index.json (server/samples.mjs). */
export interface SampleDef {
  id: string;
  name: string;
  /** Path of the .dwproj relative to samples/. */
  file: string;
}

export const sampleUrl = (s: SampleDef) => `./samples/${s.file.split('/').map(encodeURIComponent).join('/')}`;

export const useSamples = create<{ samples: SampleDef[] }>()(() => ({ samples: [] }));

let pending: Promise<SampleDef[]> | null = null;

/** Fetch the sample list (again): folders added since the last call show up. */
export function loadSamples(): Promise<SampleDef[]> {
  pending ??= fetch('./samples/index.json', { cache: 'no-store' })
    .then(async (r) => (r.ok ? ((await r.json()) as { samples?: unknown }) : null))
    .then((j) => (Array.isArray(j?.samples) ? (j.samples as SampleDef[]).filter((s) => s && typeof s.id === 'string' && typeof s.file === 'string') : []))
    .catch(() => useSamples.getState().samples)
    .then((samples) => {
      useSamples.setState({ samples });
      return samples;
    })
    .finally(() => {
      pending = null;
    });
  return pending;
}
