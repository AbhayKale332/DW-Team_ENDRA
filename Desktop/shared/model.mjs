import { copyFile, mkdir, readFile, readdir, rename, rm, stat, writeFile } from 'node:fs/promises';
import { basename, dirname, join, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';

export async function findModel(directory) {
  const names = await readdir(directory).catch(() => []);
  const graph = names.find((name) => name === 'depthwizard.onnx') ?? names.find((name) => /\.onnx$/i.test(name));
  return graph ? join(directory, graph) : null;
}

export async function modelFiles(graph) {
  const metadata = JSON.parse(await readFile(`${graph}.json`, 'utf8'));
  if (!metadata.preproc || !Array.isArray(metadata.preproc.mean) || !Array.isArray(metadata.preproc.std) || !Number.isInteger(metadata.preproc.tile_size) || metadata.preproc.tile_size <= 0 || !(metadata.preproc.canonical_gsd_m > 0)) {
    throw new Error('The model needs its exported .onnx.json preprocessing metadata.');
  }
  const external = metadata.external_data ?? [];
  if (!Array.isArray(external) || external.some((name) => typeof name !== 'string' || !name || name !== basename(name) || /[\\/]/.test(name) || name === '.' || name === '..')) {
    throw new Error('Model weight sidecars must be files beside the ONNX graph.');
  }
  const files = [...new Set([basename(graph), `${basename(graph)}.json`, ...external])];
  for (const file of files) {
    if (!(await stat(join(dirname(graph), file))).isFile()) throw new Error(`Missing model file: ${file}`);
  }
  return files;
}

/** Copy first, then publish the selection; a failed import leaves the old model intact. */
export async function installModel(graph, userData) {
  graph = resolve(graph);
  const files = await modelFiles(graph);
  const directory = join(userData, 'models', randomUUID());
  await mkdir(directory, { recursive: true });
  try {
    for (const file of files) await copyFile(join(dirname(graph), file), join(directory, file));
    const installed = join(directory, basename(graph));
    await writeFile(join(userData, 'model.json.tmp'), JSON.stringify({ graph: installed }));
    await rename(join(userData, 'model.json.tmp'), join(userData, 'model.json'));
    return installed;
  } catch (error) {
    await rm(directory, { recursive: true, force: true });
    throw error;
  }
}
