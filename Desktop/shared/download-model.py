"""Optional CI model bundle download. SHA-256 is required for reproducible releases."""
import hashlib
import os
from pathlib import Path
import tempfile
from urllib.request import Request, urlopen, build_opener, HTTPRedirectHandler
import zipfile

url = os.environ.get("DESKTOP_MODEL_URL", "")
if url:
    checksum = os.environ.get("DESKTOP_MODEL_SHA256", "").lower()
    if len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum):
        raise ValueError("Set DESKTOP_MODEL_SHA256 to the model ZIP's SHA-256")
    if not url.startswith("https://"):
        raise ValueError("The model ZIP URL must use HTTPS")
    target = Path(__file__).parent / "resources" / "model"
    target.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        digest = hashlib.sha256()
        headers = {}
        if url.startswith('https://api.github.com/repos/AbhayKale332/DepthWizard/releases/assets/'):
            headers = {'Accept': 'application/octet-stream', 'User-Agent': 'DepthWizard-release'}
            if os.environ.get('GH_TOKEN'):
                headers['Authorization'] = 'Bearer ' + os.environ['GH_TOKEN']
        class AssetRedirect(HTTPRedirectHandler):
            def redirect_request(self, request, *args, **kwargs):
                redirected = super().redirect_request(request, *args, **kwargs)
                if redirected is not None:
                    redirected.remove_header('Authorization')
                return redirected
        request = Request(url, headers=headers)
        with build_opener(AssetRedirect).open(request, timeout=120) as response:
            while chunk := response.read(1024 * 1024):
                digest.update(chunk)
                archive.write(chunk)
        if digest.hexdigest() != checksum:
            raise ValueError("Model ZIP checksum mismatch")
        archive.seek(0)
        with zipfile.ZipFile(archive) as bundle:
            # Exported model files must be at the ZIP root, including .data and .json.
            for entry in bundle.infolist():
                if entry.is_dir() or Path(entry.filename).name != entry.filename or "\\" in entry.filename:
                    raise ValueError("Model ZIP must contain files at its root, without folders")
            bundle.extractall(target)
    graphs = list(target.glob("*.onnx"))
    if len(graphs) != 1:
        raise ValueError("Model ZIP must contain exactly one .onnx graph")
    from inference import validate_model
    validate_model(graphs[0])
    print(f"Verified bundled model: {graphs[0].name}")
else:
    print("Building without bundled weights; use Model → Install ONNX model after installation.")
