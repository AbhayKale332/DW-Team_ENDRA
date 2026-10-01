/** Canvas side of cloud masking (works in a worker or on the main thread: OffscreenCanvas only). */
import { analysisSize, detectClouds, fillImage, resampleMask, type CloudMask } from './cloud';

/** Long side the colour fill is computed at; the fill is smooth, so it is drawn up to full size. */
const FILL_SIDE = 2048;

async function pixels(source: Blob, width: number, height: number) {
  const bmp = await createImageBitmap(source, { resizeWidth: width, resizeHeight: height, resizeQuality: 'high' });
  const canvas = new OffscreenCanvas(width, height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  return ctx.getImageData(0, 0, width, height).data;
}

async function sizeOf(image: Blob) {
  const bmp = await createImageBitmap(image);
  const s = { width: bmp.width, height: bmp.height };
  bmp.close();
  return s;
}

/** Find clouds in an image file. */
export async function detectCloudsInImage(image: Blob): Promise<CloudMask> {
  const full = await sizeOf(image);
  const a = analysisSize(full.width, full.height);
  return detectClouds(await pixels(image, a.width, a.height), a.width, a.height);
}

/** The image at full size with the clouds painted over by the surrounding colours (PNG). */
export async function cloudFreeImage(image: Blob, cloud: CloudMask): Promise<Blob> {
  const full = await sizeOf(image);
  const f = analysisSize(full.width, full.height, FILL_SIDE);
  const rgba = await pixels(image, f.width, f.height);
  const mask = resampleMask(cloud.mask, cloud.width, cloud.height, f.width, f.height);
  const filled = fillImage(rgba, mask, f.width, f.height);
  // fill layer: opaque over the cloud, fading out over its hazy edge, transparent elsewhere
  for (let i = 0; i < mask.length; i++) filled[i * 4 + 3] = Math.min(255, Math.round((mask[i] / 160) * 255));
  const layer = new OffscreenCanvas(f.width, f.height);
  layer.getContext('2d')!.putImageData(new ImageData(filled, f.width, f.height), 0, 0);

  const canvas = new OffscreenCanvas(full.width, full.height);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2D canvas unavailable');
  const bmp = await createImageBitmap(image);
  ctx.drawImage(bmp, 0, 0);
  bmp.close();
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(layer, 0, 0, full.width, full.height);
  return canvas.convertToBlob({ type: 'image/png' });
}

/** A small preview of the image with the cloud mask tinted cyan (PNG), for the "clouds detected" prompt. */
export async function cloudPreview(image: Blob, cloud: CloudMask, side = 360): Promise<Blob> {
  const full = await sizeOf(image);
  const p = analysisSize(full.width, full.height, side);
  const rgba = await pixels(image, p.width, p.height);
  const mask = resampleMask(cloud.mask, cloud.width, cloud.height, p.width, p.height);
  for (let i = 0; i < mask.length; i++) {
    const a = (mask[i] / 255) * 0.6;
    rgba[i * 4] = rgba[i * 4] * (1 - a);
    rgba[i * 4 + 1] = rgba[i * 4 + 1] * (1 - a) + 230 * a;
    rgba[i * 4 + 2] = rgba[i * 4 + 2] * (1 - a) + 255 * a;
  }
  const canvas = new OffscreenCanvas(p.width, p.height);
  canvas.getContext('2d')!.putImageData(new ImageData(rgba, p.width, p.height), 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}

/** The mask alone as a cyan, semi-transparent PNG, to lay over a preview of the image. */
export async function cloudOverlay(cloud: CloudMask): Promise<Blob> {
  const rgba = new Uint8ClampedArray(cloud.width * cloud.height * 4);
  for (let i = 0; i < cloud.mask.length; i++) {
    rgba[i * 4 + 1] = 230;
    rgba[i * 4 + 2] = 255;
    rgba[i * 4 + 3] = Math.round(cloud.mask[i] * 0.6);
  }
  const canvas = new OffscreenCanvas(cloud.width, cloud.height);
  canvas.getContext('2d')!.putImageData(new ImageData(rgba, cloud.width, cloud.height), 0, 0);
  return canvas.convertToBlob({ type: 'image/png' });
}
