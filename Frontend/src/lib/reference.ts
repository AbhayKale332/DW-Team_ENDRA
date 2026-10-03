import type { ReferenceKind, ReferenceSurface, Scene } from '@/domain/types';
import { decodeTiffBand } from './geotiff';
import { parseNpy } from './npy';
import { makePixelMapper } from './georef';
import { resampleGrid, sampleBilinear } from './heights';

/** Use the original input, since TIFF drapes and model uploads are also PNGs. */
export function referenceKinds(scene: Pick<Scene, 'meta' | 'georef' | 'product'>): ReferenceKind[] {
  const name = scene.meta.source_image_name ?? scene.meta.scene?.path ?? '';
  if (/\.(?:geo)?tiff?$/i.test(name)) return ['DSM', 'nDSM'];
  if (/\.(png|jpe?g)$/i.test(name)) return ['nDSM'];
  // Older samples and result bundles may have no original filename.
  return scene.georef || scene.product === 'DSM' ? ['DSM', 'nDSM'] : ['nDSM'];
}

/** Load a reference surface and align it onto the prediction grid.
 *  • both georeferenced → each prediction pixel centre is mapped into the reference raster (CRS-aware)
 *  • otherwise → the reference is assumed to cover the same extent and is resampled bilinearly */
export async function loadReference(file: File, kind: ReferenceKind, scene: Scene): Promise<ReferenceSurface> {
  const { width: W, height: H } = scene.heights;
  const name = file.name;
  const lower = name.toLowerCase();
  const notes: string[] = [];

  if (lower.endsWith('.npy')) {
    const arr = parseNpy(await file.arrayBuffer());
    if (arr.shape.length !== 2) throw new Error('Reference .npy must be a 2-D array (rows, cols).');
    const [rh, rw] = arr.shape;
    let data = arr.data;
    if (rw !== W || rh !== H) {
      data = resampleGrid(arr.data, rw, rh, W, H);
      notes.push(`Resampled from ${rw} × ${rh} to the prediction grid (${W} × ${H}), assuming the same extent.`);
    } else notes.push('Same pixel grid as the prediction.');
    return { name, kind, data, alignment: 'same-extent', notes };
  }

  if (/\.(?:geo)?tiff?$/.test(lower)) {
    const ref = await decodeTiffBand(await file.arrayBuffer());
    if (ref.nodata !== null) notes.push(`NoData value ${ref.nodata} masked.`);
    const grid = { data: ref.data, width: ref.width, height: ref.height };
    if (scene.georef && ref.georef) {
      const mapper = makePixelMapper(scene.georef, ref.georef);
      if (mapper) {
        const out = new Float32Array(W * H);
        // Map on a coarse lattice and interpolate — exact for affine pairs, sub-pixel for reprojection.
        const S = scene.georef.epsg && scene.georef.epsg === ref.georef.epsg ? 1 : 16;
        const lw = Math.ceil((W - 1) / S) + 1;
        const lh = Math.ceil((H - 1) / S) + 1;
        const lc = new Float64Array(lw * lh);
        const lr = new Float64Array(lw * lh);
        for (let j = 0; j < lh; j++)
          for (let i = 0; i < lw; i++) {
            const [c, r] = mapper(Math.min(W - 1, i * S), Math.min(H - 1, j * S));
            lc[j * lw + i] = c;
            lr[j * lw + i] = r;
          }
        let inside = 0;
        for (let row = 0; row < H; row++) {
          const fj = row / S;
          const j0 = Math.min(lh - 1, Math.floor(fj));
          const j1 = Math.min(lh - 1, j0 + 1);
          const ty = fj - j0;
          for (let col = 0; col < W; col++) {
            const fi = col / S;
            const i0 = Math.min(lw - 1, Math.floor(fi));
            const i1 = Math.min(lw - 1, i0 + 1);
            const tx = fi - i0;
            const lerp2 = (a: Float64Array) =>
              (a[j0 * lw + i0] * (1 - tx) + a[j0 * lw + i1] * tx) * (1 - ty) + (a[j1 * lw + i0] * (1 - tx) + a[j1 * lw + i1] * tx) * ty;
            const c = lerp2(lc);
            const r = lerp2(lr);
            if (c < -0.5 || r < -0.5 || c > ref.width - 0.5 || r > ref.height - 0.5) {
              out[row * W + col] = NaN;
              continue;
            }
            out[row * W + col] = sampleBilinear(grid, c, r);
            inside++;
          }
        }
        notes.push(`Aligned by georeferencing (EPSG:${ref.georef.epsg ?? '?'} → EPSG:${scene.georef.epsg ?? '?'}); ${((inside / (W * H)) * 100).toFixed(1)} % of the scene is covered.`);
        if (inside === 0) throw new Error('The reference does not overlap the scene.');
        return { name, kind, data: out, alignment: 'georeferenced', notes };
      }
      notes.push('The two coordinate systems could not be related; falling back to same-extent alignment.');
    } else if (ref.georef && !scene.georef) {
      notes.push('The prediction is not georeferenced, so the reference is assumed to cover exactly the same area.');
    }
    const data = ref.width === W && ref.height === H ? ref.data : resampleGrid(ref.data, ref.width, ref.height, W, H);
    if (ref.width !== W || ref.height !== H) notes.push(`Resampled from ${ref.width} × ${ref.height} to ${W} × ${H}.`);
    return { name, kind, data, alignment: 'same-extent', notes };
  }

  throw new Error('Use a GeoTIFF (.tif, .tiff, .geotiff) or NumPy (.npy) reference height raster.');
}
