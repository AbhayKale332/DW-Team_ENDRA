// Pack a loose result folder (ndsm_m.npy + meta.json + rgb.png [+ seg.png, objects.json, dem_cells.json, the input
// image]) into a .dwproj sample project — the format the app lists under File → Sample Scenes.
//
//   node scripts/pack-sample.mjs public/samples/<folder> [--clean]
//
// Writes <folder>/<folder>.dwproj. --clean then deletes the loose files that went into it.
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { strToU8, zipSync } from 'fflate';

const dir = resolve(process.argv[2] ?? 'public/samples/buildings_large_campus');
const clean = process.argv.includes('--clean');
const name = basename(dir);
const read = (f) => new Uint8Array(readFileSync(join(dir, f)));
const has = (f) => existsSync(join(dir, f));
for (const f of ['ndsm_m.npy', 'meta.json', 'rgb.png']) if (!has(f)) throw new Error(`${join(dir, f)} is missing`);

const meta = JSON.parse(readFileSync(join(dir, 'meta.json'), 'utf8'));
const used = ['ndsm_m.npy', 'meta.json', 'rgb.png'];
const files = { 'input.png': [read('rgb.png'), { level: 0 }], 'ndsm_m.npy': read('ndsm_m.npy') };
const names = [];
for (const [k, v] of Object.entries(meta.classes ?? {})) if (Number.isInteger(Number(k))) names[Number(k)] = String(v);
if (has('seg.png') && meta.classes) {
  files['seg.png'] = [read('seg.png'), { level: 0 }];
  used.push('seg.png');
}
for (const f of ['objects.json', 'dem_cells.json']) {
  if (!has(f)) continue;
  files[f] = read(f);
  used.push(f);
}
// the input image the result was made from, so the model can be run again from the sample
const srcName = meta.scene?.path ? basename(meta.scene.path) : null;
let source = null;
if (srcName && has(srcName)) {
  source = { name: srcName, file: `source/${srcName}` };
  files[source.file] = [read(srcName), { level: 0 }];
  used.push(srcName);
}

const manifest = {
  format: 'dwproj',
  version: 2,
  name,
  imageName: 'input.png',
  gsd: meta.scene?.gsd_m ?? 0.5,
  gsdSource: meta.scene?.gsd_source ?? 'assumed',
  // null: the app derives it from meta.scene (transform + EPSG), exactly as for a fresh model result
  georef: null,
  meta,
  statusLines: meta.note ? [meta.note] : [],
  provenance: { provider: 'bundled sample', source: 'sample', createdAt: new Date().toISOString() },
  view: {},
  bookmarks: [],
  tools: { probe: null, measure: [], profile: [] },
  reference: null,
  classes: files['seg.png'] ? { names: Array.from(names, (n) => n ?? '') } : null,
  anchoring: null,
  source,
  outputs: [],
};
files['manifest.json'] = strToU8(JSON.stringify(manifest, null, 2));
const out = join(dir, `${name}.dwproj`);
writeFileSync(out, zipSync(files, { level: 6 }));
console.log(`${out}  (${Object.keys(files).join(', ')})`);
if (clean) for (const f of used) rmSync(join(dir, f));
