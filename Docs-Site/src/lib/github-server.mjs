const repository = 'AbhayKale332/DW-Team_ENDRA';
export const assetNameAllowed = (name) => /^DepthWizard-.*\.(exe|dmg|AppImage|zip)(\.sha256)?$/.test(name) || name === 'SHA256SUMS.txt';

export function githubHeaders(accept = 'application/vnd.github+json') {
  const token = process.env.GITHUB_RELEASES_TOKEN;
  return { ...(token ? { Authorization: `Bearer ${token}` } : {}), Accept: accept, 'X-GitHub-Api-Version': '2022-11-28', 'User-Agent': 'DepthWizard-downloads' };
}

export async function github(path, options = {}) {
  const response = await fetch(`https://api.github.com/repos/${repository}${path ? `/${path}` : ''}`, {
    headers: githubHeaders(), signal: AbortSignal.timeout(15000), ...options,
  });
  if (!response.ok && response.status !== 302) throw new Error('GitHub release service is unavailable');
  return response;
}
