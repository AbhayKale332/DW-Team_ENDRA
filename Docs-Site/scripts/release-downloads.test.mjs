import assert from 'node:assert/strict';
import { test } from 'node:test';
import { desktopReleases, installer, sizeLabel } from '../src/lib/releases.mjs';

const asset = (name, url = 'https://github.com/AbhayKale332/DepthWizard/releases/download/desktop-v0.1.3/' + name) => ({ name, browser_download_url: url, state: 'uploaded', size: 1800000000 });
const release = { id: 3, tag_name: 'desktop-v0.1.3', name: 'DepthWizard 0.1.3', draft: false, published_at: '2026-10-04T06:00:00Z', assets: [
  asset('DepthWizard-0.1.3-win-x64-with-model.exe'), asset('DepthWizard-0.1.3-win-x64-without-model.exe'),
  asset('DepthWizard-0.1.3-mac-arm64-with-model.dmg'), asset('DepthWizard-0.1.3-mac-x64-with-model.dmg'),
  asset('DepthWizard-0.1.3-linux-x86_64-with-model.AppImage'),
] };

test('downloads follow published releases, model choice and architecture', () => {
  const older = { ...release, id: 2, published_at: '2026-10-03T06:00:00Z' };
  assert.deepEqual(desktopReleases([older, { ...release, draft: true }, release, { ...release, assets: [] }]).map((item) => item.id), [3, 2]);
  assert.match(installer(release, 'win', 'x64', true).name, /-with-model.exe$/);
  assert.match(installer(release, 'win', 'x64', false).name, /-without-model.exe$/);
  assert.match(installer(release, 'mac', 'arm64', true).name, /-arm64-with-model.dmg$/);
  assert.match(installer(release, 'mac', 'x64', true).name, /-x64-with-model.dmg$/);
  assert.match(installer(release, 'linux', 'x64', true).name, /-x86_64-with-model.AppImage$/);
  assert.equal(installer(release, 'mac', 'arm64', false), null);
  assert.equal(installer({ ...release, assets: [asset('DepthWizard-0.1.3-win-x64-with-model.exe', 'https://evil.test/file.exe')] }, 'win', 'x64', true), null);
  assert.equal(sizeLabel(1800000000), '1.80 GB');
});
