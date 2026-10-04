import test from 'node:test';
import assert from 'node:assert/strict';
import releases from '../api/releases.mjs';
import download from '../api/download.mjs';

const asset = { id: 42, name: 'DepthWizard-0.1.3-win-x64-with-model.exe', state: 'uploaded', size: 123, browser_download_url: 'https://github.com/AbhayKale332/DepthWizard/releases/download/desktop-v0.1.3/app.exe' };
const published = { id: 3, tag_name: 'desktop-v0.1.3', draft: false, assets: [asset] };
function response() { return { statusCode: 200, headers: {}, setHeader(key, value) { this.headers[key] = value; }, end(body) { this.body = body; } }; }
test('published downloads keep repository credentials server-side and validate redirects', async () => {
  const originalFetch = globalThis.fetch;
  const originalToken = process.env.GITHUB_RELEASES_TOKEN;
  process.env.GITHUB_RELEASES_TOKEN = 'test-secret';
  let location = 'https://release-assets.githubusercontent.com/example?signature=test';
  globalThis.fetch = async (url, options) => {
    assert.equal(options.headers.Authorization, 'Bearer test-secret');
    if (url.endsWith('/releases/assets/42')) {
      assert.equal(options.redirect, 'manual');
      assert.equal(options.headers.Accept, 'application/octet-stream');
      return new Response(null, { status: 302, headers: { Location: location } });
    }
    return Response.json([published, { ...published, id: 4, draft: true, assets: [{ ...asset, id: 43 }] }]);
  };
  try {
    let res = response();
    await releases({ method: 'GET' }, res);
    const body = JSON.parse(res.body);
    assert.equal(body.length, 1);
    assert.equal(body[0].assets[0].browser_download_url, '/api/download?asset=42');
    assert.ok(!res.body.includes('test-secret'));
    res = response();
    await download({ method: 'GET', url: '/api/download?asset=42' }, res);
    assert.equal(res.statusCode, 302);
    assert.equal(res.headers.Location, location);
    assert.equal(res.headers['Cache-Control'], 'no-store');
    for (const id of ['43', '999', '../42']) {
      res = response();
      await download({ method: 'GET', url: `/api/download?asset=${id}` }, res);
      assert.ok([400, 404].includes(res.statusCode));
    }
    location = 'https://attacker.example/download';
    res = response();
    await download({ method: 'GET', url: '/api/download?asset=42' }, res);
    assert.equal(res.statusCode, 503);
    assert.equal(res.headers.Location, undefined);
    delete process.env.GITHUB_RELEASES_TOKEN;
    globalThis.fetch = async (_, options) => {
      assert.equal(options.headers.Authorization, undefined);
      return Response.json([published]);
    };
    res = response();
    await releases({ method: 'GET' }, res);
    assert.equal(JSON.parse(res.body)[0].assets[0].browser_download_url, asset.browser_download_url);
    res = response();
    await download({ method: 'POST', url: '/api/download?asset=42' }, res);
    assert.equal(res.statusCode, 405);
  } finally {
    globalThis.fetch = originalFetch;
    if (originalToken === undefined) delete process.env.GITHUB_RELEASES_TOKEN;
    else process.env.GITHUB_RELEASES_TOKEN = originalToken;
  }
});

test('standard version release tags are listed and authorized for installer downloads', async () => {
  const originalFetch = globalThis.fetch;
  const location = 'https://release-assets.githubusercontent.com/standard-version?signature=test';
  const versioned = { ...published, tag_name: 'v1.0.1', assets: [{ ...asset, name: 'DepthWizard-1.0.1-win-x64-with-model.exe' }] };
  globalThis.fetch = async (url) => url.endsWith('/releases/assets/42')
    ? new Response(null, { status: 302, headers: { Location: location } })
    : Response.json([versioned, { ...versioned, draft: true, assets: [{ ...asset, id: 43 }] },
      { ...versioned, tag_name: 'model-v1.0.1', assets: [{ ...asset, id: 44 }] }]);
  try {
    let res = response();
    await releases({ method: 'GET' }, res);
    assert.equal(JSON.parse(res.body).length, 1);
    assert.equal(JSON.parse(res.body)[0].tag_name, 'v1.0.1');
    res = response();
    await download({ method: 'GET', url: '/api/download?asset=42' }, res);
    assert.equal(res.statusCode, 302);
    assert.equal(res.headers.Location, location);
    for (const id of ['43', '44']) {
      res = response();
      await download({ method: 'GET', url: `/api/download?asset=${id}` }, res);
      assert.equal(res.statusCode, 404);
    }
  } finally { globalThis.fetch = originalFetch; }
});
