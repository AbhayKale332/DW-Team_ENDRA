import test, { mock } from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { runInNewContext } from 'node:vm';

const writes = [];
mock.module('@vercel/blob', { namedExports: {
  put: async (pathname, body) => writes.push({ pathname, ...JSON.parse(body) }),
  list: async ({ cursor }) => ({
    blobs: (cursor ? writes.slice(2) : writes.slice(0, 2)).map(({ pathname, timestamp }) => ({ pathname, uploadedAt: new Date(timestamp) })),
    hasMore: !cursor,
    cursor: cursor ? undefined : 'next',
  }),
} });
const { default: track } = await import('../api/track-visit.mjs');
const { default: counts } = await import('../api/visitor-counts.mjs');
const response = () => ({ statusCode: 200, setHeader() {}, end(body) { this.body = body; } });

test('page loads stay separate and aggregate across paginated storage', async () => {
  for (const body of [{ ppt: true }, { page: 'judges', ppt: true }, { page: 'downloadPpt', ppt: true }]) {
    const res = response();
    await track({ method: 'POST', body }, res);
    assert.equal(res.statusCode, 204);
  }
  assert.deepEqual(writes.map(({ type }) => type), ['homepage', 'ppt', 'judges', 'downloadPpt']);
  const invalid = response();
  await track({ method: 'POST', body: { page: '../invalid' } }, invalid);
  assert.equal(invalid.statusCode, 400);
  assert.equal(writes.length, 4);

  const res = response();
  await counts({ method: 'GET' }, res);
  const data = JSON.parse(res.body);
  assert.deepEqual([data.homepage, data.ppt, data.judges, data.downloadPpt], [1, 1, 1, 1]);
  assert.equal(data.daily.length, 30);
  assert.deepEqual(data.daily.at(-1), { date: writes[0].timestamp.slice(0, 10), homepage: 1, ppt: 1, judges: 1, downloadPpt: 1 });
  assert.deepEqual(Object.values(data.daily[0]).slice(1), [0, 0, 0, 0]);
});

test('browser records judges and only ppt-tagged downloads, including trailing slash URLs', async () => {
  const source = await readFile(new URL('../src/components/HomepageTracker.astro', import.meta.url), 'utf8');
  const script = source.match(/<script>([\s\S]*?)<\/script>/)[1];
  for (const [url, expected] of [
    ['https://docs.depthwizard.teamendra.tech/judges', 'judges'],
    ['https://docs.depthwizard.teamendra.tech/judges/', 'judges'],
    ['https://depthwizard-docs.vercel.app/download/?ppt', 'downloadPpt'],
    ['https://depthwizard-docs.vercel.app/download?ppt=', 'downloadPpt'],
    ['https://depthwizard-docs.vercel.app/download/', null],
    ['https://depthwizard-docs.vercel.app/?ppt', 'homepage'],
    ['https://depthwizard-docs.vercel.app/model/overview/', null],
  ]) {
    const calls = [];
    runInNewContext(script, { window: { location: new URL(url) }, URLSearchParams, fetch: async (_, options) => calls.push(JSON.parse(options.body)) });
    assert.deepEqual(calls.map(({ page }) => page), expected ? [expected] : []);
  }
});
