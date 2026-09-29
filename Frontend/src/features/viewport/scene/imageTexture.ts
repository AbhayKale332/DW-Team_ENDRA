import * as THREE from 'three';

/** The optical image as a mipmapped sRGB texture, downscaled to `maxSize`. Row 0 at v = 0 (flipY = false),
 *  matching the terrain's and the roofs' uv convention. */
export async function loadImageTexture(blob: Blob, maxSize: number, anisotropy: number) {
  const probe = await createImageBitmap(blob);
  const scale = Math.min(1, maxSize / Math.max(probe.width, probe.height));
  let bmp = probe;
  if (scale < 1) {
    bmp = await createImageBitmap(blob, { resizeWidth: Math.round(probe.width * scale), resizeHeight: Math.round(probe.height * scale), resizeQuality: 'high' });
    probe.close();
  }
  const tex = new THREE.Texture(bmp as unknown as HTMLImageElement);
  tex.flipY = false;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = anisotropy;
  tex.generateMipmaps = true;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.wrapS = THREE.ClampToEdgeWrapping;
  tex.wrapT = THREE.ClampToEdgeWrapping;
  tex.needsUpdate = true;
  return tex;
}
