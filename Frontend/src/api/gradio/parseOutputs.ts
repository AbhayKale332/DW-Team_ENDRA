import type { Artefact } from '@/domain/types';

/** Gradio FileData as serialised by the client. */
export interface GradioFile {
  path?: string;
  url?: string | null;
  size?: number | null;
  orig_name?: string | null;
  mime_type?: string | null;
}

export function fileBaseName(f: GradioFile): string {
  const raw = f.orig_name || f.path || f.url || '';
  const clean = raw.split('?')[0];
  return decodeURIComponent(clean.split(/[\\/]/).pop() || '');
}

/** The /predict tuple: [gallery, height16, viewerHtml, downloads, status, warning]. */
export function parsePredictTuple(data: unknown[]) {
  const [, , , downloads, status, warning] = data as [unknown, unknown, unknown, GradioFile[] | GradioFile | null, string | null, string | null];
  const files: GradioFile[] = Array.isArray(downloads) ? downloads : downloads ? [downloads] : [];
  const artefacts: Artefact[] = files
    .filter((f) => f && (f.url || f.path))
    .map((f) => ({ name: fileBaseName(f), url: f.url ?? undefined, size: f.size ?? undefined }));
  return {
    artefacts,
    status: typeof status === 'string' ? status : '',
    warning: typeof warning === 'string' && warning.trim() ? warning : null,
  };
}

export function findArtefact(artefacts: Artefact[], test: string | RegExp): Artefact | undefined {
  return artefacts.find((a) => (typeof test === 'string' ? a.name === test : test.test(a.name)));
}
