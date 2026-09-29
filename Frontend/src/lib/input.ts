import type { Georef } from '@/domain/types';
import { DepthWizardError } from '@/api/errors';
import { decodeTiff, isTiffName } from './geotiff';

export const ACCEPTED_IMAGE_TYPES = ['image/png', 'image/jpeg', 'image/tiff', 'image/tif'];
export const ACCEPTED_EXT = /\.(png|jpe?g|tiff?)$/i;
export const MODEL_MAX_SIDE = 8192;
export const MAX_FILE_BYTES = 512 * 1024 * 1024;

export interface PreparedInput {
  file: File;
  name: string;
  /** What is sent to the model: the original PNG/JPG, or a PNG re-encode of a TIFF. */
  upload: Blob;
  uploadName: string;
  /** Displayable image at (near-)original resolution — used as the preview and the 3D drape. */
  display: Blob;
  previewUrl: string;
  width: number;
  height: number;
  bands: number;
  bitsPerSample: number;
  isTiff: boolean;
  georef: Georef | null;
  /** GSD derived from the GeoTIFF transform, in metres/pixel. */
  fileGsd: number | null;
}

export function stemOf(name: string) {
  return name.replace(/\.[^.]+$/, '') || 'scene';
}

async function rgbaToPng(rgba: Uint8ClampedArray, w: number, h: number): Promise<Blob> {
  const canvas = new OffscreenCanvas(w, h);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), w, h), 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

export async function prepareInput(file: File): Promise<PreparedInput> {
  if (!ACCEPTED_EXT.test(file.name) && !ACCEPTED_IMAGE_TYPES.includes(file.type))
    throw new DepthWizardError('invalid-input', 'Unsupported file type', `${file.name} is not a PNG, JPG or TIFF image.`);
  if (file.size > MAX_FILE_BYTES) throw new DepthWizardError('invalid-input', 'File too large', 'Images up to 512 MB are supported.');

  if (isTiffName(file.name) || file.type.includes('tif')) {
    const t = await decodeTiff(await file.arrayBuffer());
    const png = await rgbaToPng(t.rgba, t.rgbaWidth, t.rgbaHeight);
    // If the TIFF had to be downsampled for the browser, the upload GSD scales with it.
    return {
      file,
      name: file.name,
      upload: png,
      uploadName: `${stemOf(file.name)}.png`,
      display: png,
      previewUrl: URL.createObjectURL(png),
      width: t.width,
      height: t.height,
      bands: t.bands,
      bitsPerSample: t.bitsPerSample,
      isTiff: true,
      georef: t.georef,
      fileGsd: t.gsd,
    };
  }

  let bmp: ImageBitmap;
  try {
    bmp = await createImageBitmap(file);
  } catch {
    throw new DepthWizardError('invalid-input', 'Could not decode image', `${file.name} could not be read as an image.`);
  }
  const { width, height } = bmp;
  bmp.close();
  return {
    file,
    name: file.name,
    upload: file,
    uploadName: file.name,
    display: file,
    previewUrl: URL.createObjectURL(file),
    width,
    height,
    bands: 3,
    bitsPerSample: 8,
    isTiff: false,
    georef: null,
    fileGsd: null,
  };
}

/** Effective GSD and size after the model's own downsampling (read_scene max_side = DW_MAX_SIDE, default 8192). */
export function effectiveModelGrid(width: number, height: number, gsd: number) {
  const longSide = Math.max(width, height);
  const scale = longSide > MODEL_MAX_SIDE ? MODEL_MAX_SIDE / longSide : 1;
  return {
    resampled: scale < 1,
    width: Math.round(width * scale),
    height: Math.round(height * scale),
    gsd: gsd / scale,
  };
}
