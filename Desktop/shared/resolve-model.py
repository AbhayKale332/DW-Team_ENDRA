"""Resolve the newest published model bundle, or use explicit workflow overrides."""
import json
import os
from urllib.request import Request, urlopen

url = os.environ.get('DESKTOP_MODEL_URL', '')
checksum = os.environ.get('DESKTOP_MODEL_SHA256', '')
if not url:
    repository = os.environ.get('GITHUB_REPOSITORY', 'AbhayKale332/DW-Team_ENDRA')
    headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'DepthWizard-release'}
    if os.environ.get('GH_TOKEN'):
        headers['Authorization'] = 'Bearer ' + os.environ['GH_TOKEN']
    request = Request(f'https://api.github.com/repos/{repository}/releases?per_page=100', headers=headers)
    with urlopen(request, timeout=30) as response:
        releases = json.load(response)
    releases = sorted((release for release in releases if not release['draft']), key=lambda release: release.get('published_at') or '', reverse=True)
    for release in releases:
        asset = next((asset for asset in sorted(release['assets'], key=lambda asset: asset.get('created_at') or '', reverse=True) if asset['state'] == 'uploaded' and
                      asset['name'].startswith('DepthWizard-V5-') and '-model' in asset['name'] and asset['name'].endswith('.zip')), None)
        if asset:
            url = asset['url']
            digest = asset.get('digest') or ''
            if not digest.startswith('sha256:'):
                raise ValueError('Published model bundle must have a SHA-256 asset digest')
            checksum = digest.removeprefix('sha256:')
            break
if not url:
    raise ValueError('No published model bundle found; supply model_url and model_sha256')
if not url.startswith('https://') or '\n' in url or '\r' in url:
    raise ValueError('Model URL must be a single HTTPS URL')
if len(checksum) != 64 or any(character not in '0123456789abcdefABCDEF' for character in checksum):
    raise ValueError('Model bundle must have a valid SHA-256 checksum')
with open(os.environ['GITHUB_ENV'], 'a', encoding='utf-8') as environment:
    environment.write(f'DESKTOP_MODEL_URL={url}\nDESKTOP_MODEL_SHA256={checksum.lower()}\n')
print('Resolved a verified model bundle for model-included installers')
