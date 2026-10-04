import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createServer, get } from 'node:http';
import { mkdtemp, mkdir, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { once } from 'node:events';
import { installModel, modelFiles } from './model.mjs';
import { desktopHandler } from './local-api.mjs';

test('model import copies sidecars and preserves selection when another import fails', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'dw-model-test-'));
  try {
    const data = join(directory, 'data');
    await mkdir(data);
    const graph = join(directory, 'test.onnx');
    const metadata = { preproc: { mean: [0, 0, 0], std: [1, 1, 1], tile_size: 512, canonical_gsd_m: 0.5 }, external_data: ['weights.data'] };
    await writeFile(graph, 'graph');
    await writeFile(`${graph}.json`, JSON.stringify(metadata));
    await writeFile(join(directory, 'weights.data'), 'weights');
    const installed = await installModel(graph, data);
    assert.equal(await readFile(join(installed, '..', 'weights.data'), 'utf8'), 'weights');
    const selection = await readFile(join(data, 'model.json'), 'utf8');
    await rm(join(directory, 'weights.data'));
    await assert.rejects(installModel(graph, data));
    assert.equal(await readFile(join(data, 'model.json'), 'utf8'), selection);
    await writeFile(`${graph}.json`, JSON.stringify({ ...metadata, external_data: ['../secret'] }));
    await assert.rejects(modelFiles(graph), /beside/);
    await writeFile(`${graph}.json`, '{}');
    await assert.rejects(modelFiles(graph), /preprocessing/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('local relay authenticates upstream, blocks cross-origin and never falls back to HF', async () => {
  const upstream = createServer((req, res) => {
    assert.equal(req.headers.authorization, 'Bearer test-token');
    res.writeHead(200, { 'content-type': 'application/json' }).end('{"ok":true}');
  }).listen(0, '127.0.0.1');
  await once(upstream, 'listening');
  let backend = { port: upstream.address().port, token: 'test-token' };
  const handle = desktopHandler(() => backend);
  const server = createServer((req, res) => { if (!handle(req, res)) res.writeHead(404).end(); }).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const origin = `http://127.0.0.1:${server.address().port}`;
  try {
    assert.deepEqual(await (await fetch(`${origin}/local-api/api/health`)).json(), { ok: true });
    assert.equal((await fetch(`${origin}/local-api/api/predict`, { method: 'POST', headers: { origin } })).status, 200);
    assert.equal((await fetch(`${origin}/local-api/api/predict`, { method: 'POST', headers: { origin: 'https://evil.test' } })).status, 403);
    assert.equal((await fetch(`${origin}/local-api/api/predict`, { method: 'POST' })).status, 404);
    const badHostStatus = await new Promise((resolve, reject) => {
      get(`${origin}/local-api/api/health`, { headers: { host: 'evil.test' } }, (res) => { res.resume(); resolve(res.statusCode); }).on('error', reject);
    });
    assert.equal(badHostStatus, 403);
    assert.equal((await fetch(`${origin}/local-api/docs`)).status, 404);
    assert.equal((await fetch(`${origin}/hf-space/config`)).status, 404);
    backend = { message: 'Install an ONNX model' };
    assert.match((await (await fetch(`${origin}/local-api/api/health`)).json()).detail, /Install/);
  } finally { server.closeAllConnections(); upstream.closeAllConnections(); await Promise.all([new Promise((r) => server.close(r)), new Promise((r) => upstream.close(r))]); }
});

test('model-free desktop relays hosted inference without exposing credentials', async () => {
  const hosted = await import('../../Frontend/server/guard.mjs');
  const calls = [];
  const remote = createServer((req, res) => {
    calls.push({ path: req.url, authorization: req.headers.authorization, cookie: req.headers.cookie });
    res.writeHead(200, { 'content-type': 'application/json', 'x-dw-spare-tokens': '2' }).end('{"version":"test"}');
  }).listen(0, '127.0.0.1');
  await once(remote, 'listening');
  let graph = null;
  const handle = desktopHandler(() => ({ message: 'No local model' }), { hosted, getModel: () => graph, hostedBase: `http://127.0.0.1:${remote.address().port}` });
  const server = createServer((req, res) => { if (!handle(req, res)) res.writeHead(404).end(); }).listen(0, '127.0.0.1');
  await once(server, 'listening');
  const origin = `http://127.0.0.1:${server.address().port}`;
  try {
    assert.deepEqual(await (await fetch(`${origin}/desktop-api/config`)).json(), { modelKey: null });
    graph = '/models/custom.onnx';
    assert.deepEqual(await (await fetch(`${origin}/desktop-api/config`)).json(), { modelKey: graph });
    const response = await fetch(`${origin}/hf-space/config`, { headers: { authorization: 'Bearer browser-secret', cookie: 'private=1' } });
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('x-dw-spare-tokens'), '2');
    assert.deepEqual(calls[0], { path: '/hf-space/config', authorization: undefined, cookie: undefined });
    assert.equal((await fetch(`${origin}/hf-space/gradio_api/call/predict`, { method: 'POST', headers: { origin } })).status, 200);
    assert.equal((await fetch(`${origin}/hf-space/gradio_api/call/predict`, { method: 'POST' })).status, 404);
    assert.equal((await fetch(`${origin}/hf-space/config`, { headers: { origin: 'https://evil.test' } })).status, 403);
    assert.equal((await fetch(`${origin}/hf-space/admin`)).status, 404);
    assert.equal((await fetch(`${origin}/overpass/test`)).status, 503);
    assert.equal((await fetch(`${origin}/local-api/api/health`)).status, 503, 'Local failures must never fall back to hosted inference');
    assert.equal(calls.length, 2);
  } finally { server.closeAllConnections(); remote.closeAllConnections(); await Promise.all([new Promise((r) => server.close(r)), new Promise((r) => remote.close(r))]); }
});
