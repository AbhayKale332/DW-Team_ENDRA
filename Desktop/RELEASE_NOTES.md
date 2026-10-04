# DepthWizard Desktop v1.0.1

Download DepthWizard for Windows, macOS or Linux, with local model weights included or as a smaller application using hosted inference.

## Changes

- Smaller installers and native runtime by excluding unused inference tooling, native test programs, duplicate sample assets, and default source maps.
- Linux native libraries discard unused local symbols while preserving their dynamic exports.
- Lower transient memory use for large raster overviews, tiled prediction, image textures, and worker inputs. Model weights, precision, and numerical outputs are unchanged.
- Offline samples now include only **PNG Wankhede Stadium, Mumbai** and the **georeferenced Cartosat-2S TIFF project**, with their previews.
- Both **with-model** and **without-model** installers are checked on Windows x64, Linux x64, Apple Silicon, and Intel Mac.

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
