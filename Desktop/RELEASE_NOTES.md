DepthWizard desktop for Windows x64, macOS Apple Silicon / Intel, and Linux x64.

- Windows: download the `.exe` installer.
- macOS: choose the `arm64` or `x64` `.dmg`, then copy DepthWizard to Applications.
- Linux: download the `.AppImage`, mark it executable, and run it.

Local CPU inference, 3D viewing, analysis, project files and exports work without a Hugging Face account. If the release does not include weights, download and extract the model bundle, then use **Model → Install ONNX model…**. Keep the `.onnx`, `.onnx.json` and all external weight files together. No Python or Node installation is needed on the user's computer.

The model must be a compatible DepthWizard V5 export. Missing weights do not prevent opening the application or existing projects / sample scenes. Basemaps, remote elevation data and live OSM lookups require internet access; offline geographic data can be loaded from local files or included in projects.

Unsigned builds may trigger Windows SmartScreen or macOS Gatekeeper. Check this release's signing status before distribution. Verify downloads against `SHA256SUMS.txt`.
