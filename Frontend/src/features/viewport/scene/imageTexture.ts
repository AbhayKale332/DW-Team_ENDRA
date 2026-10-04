import * as THREE from 'three';

/** The optical image as a mipmapped sRGB texture, downscaled to `maxSize`. Row 0 at v = 0 (flipY = false),
 *  matching the terrain's and the roofs' uv convention. */
export async function loadImageTexture(blob: Blob, maxSize: number, anisotropy: number) {
  const probe = await createImageBitmap(blob);
  const scale = Math.min(1, maxSize / Math.max(probe.width, probe.height));
  let bmp = probe;
  if (scale < 1) {
    try {
      bmp = await createImageBitmap(blob, { resizeWidth: Math.max(1, Math.round(probe.width * scale)), resizeHeight: Math.max(1, Math.round(probe.height * scale)), resizeQuality: 'high' });
    } finally {
      probe.close();
    }
  }
  const tex = new THREE.Texture(bmp as unknown as HTMLImageElement);
  // Texture.dispose() releases GPU storage, but ImageBitmaps need explicit CPU-side cleanup.
  tex.addEventListener('dispose', () => bmp.close());
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
