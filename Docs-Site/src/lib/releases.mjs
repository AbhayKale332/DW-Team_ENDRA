export const RELEASES_URL = 'https://api.github.com/repos/AbhayKale332/DW-Team_ENDRA/releases?per_page=100';
export const RELEASES_PAGE = 'https://github.com/AbhayKale332/DW-Team_ENDRA/releases';

export function isDesktopReleaseTag(tag) {
  return /^desktop-v/.test(tag) || /^v\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.-]+)?$/.test(tag);
}

/** Only public desktop releases with real application installers, newest first. */
export function desktopReleases(releases) {
  if (!Array.isArray(releases)) throw new Error('GitHub returned an invalid release list.');
  return releases.filter((release) => !release.draft && isDesktopReleaseTag(release.tag_name) &&
    Array.isArray(release.assets) && release.assets.some((asset) => /\.(exe|dmg|AppImage)$/.test(asset.name)))
    .sort((a, b) => Date.parse(b.published_at) - Date.parse(a.published_at));
}

export function installer(release, platform, arch, withModel) {
  const suffix = platform === 'win' ? '.exe' : platform === 'mac' ? '.dmg' : '.AppImage';
  const arches = arch === 'x64' ? ['x64', 'x86_64'] : [arch];
  const asset = release.assets.find((asset) => asset.state === 'uploaded' && asset.name.startsWith('DepthWizard-') &&
    arches.some((value) => asset.name.includes(`-${platform}-${value}`)) && asset.name.endsWith(suffix) &&
    asset.name.includes('-with-model.') === withModel);
  if (!asset) return null;
  if (/^\/api\/download\?asset=\d{1,16}$/.test(asset.browser_download_url)) return asset;
  let url;
  try { url = new URL(asset.browser_download_url); } catch { return null; }
  if (url.protocol !== 'https:' || url.hostname !== 'github.com' ||
      !url.pathname.startsWith('/AbhayKale332/DW-Team_ENDRA/releases/download/')) return null;
  return asset;
}

export function sizeLabel(bytes) {
  return bytes >= 1e9 ? `${(bytes / 1e9).toFixed(2)} GB` : `${Math.round(bytes / 1e6)} MB`;
}
