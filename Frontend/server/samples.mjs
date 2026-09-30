// Sample scenes are DepthWizard projects: every *.dwproj in a folder under samples/ is listed, so adding a sample is
// just dropping a folder with a .dwproj in it. Served as samples/index.json by the dev server, the build and
// server/serve.mjs (which rescans, so a folder added to a deployed dist/samples shows up without a rebuild).
import { readdirSync } from 'node:fs';
import { join } from 'node:path';

const isProject = (f) => /\.dwproj$/i.test(f);
const stem = (f) => f.replace(/\.dwproj$/i, '');
/** "buildings_large_campus" -> "Buildings large campus" */
const pretty = (s) => {
  const t = s.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim();
  return t ? t[0].toUpperCase() + t.slice(1) : s;
};

function entries(dir) {
  try {
    return readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  } catch {
    return [];
  }
}

/** @returns {{ id: string; name: string; file: string }[]} `file` is relative to `dir`, with forward slashes. */
export function listSamples(dir) {
  const out = [];
  for (const e of entries(dir)) {
    if (e.isFile() && isProject(e.name)) out.push({ id: stem(e.name), name: pretty(stem(e.name)), file: e.name });
    if (!e.isDirectory()) continue;
    const files = entries(join(dir, e.name))
      .filter((f) => f.isFile() && isProject(f.name))
      .map((f) => f.name);
    for (const f of files) {
      // one project per folder is the norm: it takes the folder's name
      const one = files.length === 1;
      out.push({ id: one ? e.name : `${e.name}/${stem(f)}`, name: pretty(one ? e.name : stem(f)), file: `${e.name}/${f}` });
    }
  }
  return out;
}

export function samplesIndex(dir) {
  return JSON.stringify({ samples: listSamples(dir) });
}
