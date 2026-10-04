# DepthWizard Desktop v0.1.4

Download DepthWizard for Windows, macOS or Linux, with local model weights included or as a smaller application using hosted inference.

## Changes

- Added a download page that follows published GitHub releases automatically, with Windows, Linux and Apple Silicon / Intel Mac choices.
- Added **with-model** installers containing the verified V5 forest ONNX model. Local CPU inference is selected automatically on first launch; no separate model installation or settings step is needed.
- Added **without-model** installers that use hosted Hugging Face inference by default. An internet connection is required for these predictions, and uploaded images are sent to the hosted service.
- Import a compatible model through **Model → Install ONNX model…** to switch to local inference automatically. Explicit inference choices are saved across restarts.
- Updated the release workflow to verify the model checksum and test both packaged variants on every supported platform.

## Installation

| Platform | Download | Install |
| --- | --- | --- |
| Windows 10/11, x64 | `.exe` | Run the installer. |
| macOS, Apple Silicon | `arm64` `.dmg` | Open the disk image and copy DepthWizard to Applications. |
| macOS, Intel | `x64` `.dmg` | Open the disk image and copy DepthWizard to Applications. |
| Linux, x64 | `.AppImage` | Mark the file executable and run it. |

Choose a filename containing `with-model` for offline local inference, or `without-model` for the smaller hosted-inference download. Both include the application runtime; Python and Node.js are not required. Verify installers against `SHA256SUMS.txt`.

## Notes

- Local inference uses the CPU. Large images can take several minutes.
- Basemaps, remote elevation data and live OSM features require internet even with a local model.
- Custom models must use the compatible DepthWizard V5 contract. Keep the `.onnx.json` metadata and any external weight files alongside the `.onnx` graph when importing.
- Unsigned installers can show an operating system security prompt.
- Automatic application updates are not enabled; use the download page for future releases.
