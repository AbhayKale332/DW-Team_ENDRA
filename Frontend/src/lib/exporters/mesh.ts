import * as THREE from 'three';
import { zipSync, strToU8 } from 'fflate';
import type { ObjectKinds, Scene } from '@/domain/types';
import { activeObjectKinds, anyObjectKind } from '@/store/view';
import { terrainWorker } from '@/workers/clients';
import { colorAt, type ColormapId } from '@/theme/colormaps';
import { buildObjectLayer } from '@/features/viewport/scene/objectLayer';
import { loadObjectSurface } from '@/features/viewport/objectSurfaceClient';

export interface MeshExportOptions {
  budget: number;
  wallThreshold: number;
  exaggeration: number;
  colormap: ColormapId;
  range: [number, number];
  /** Object classes to include as models over the terrain flattened under them (GLB only; what the 3D view
   *  shows). The measurement formats (OBJ, PLY, STL) always carry the model's raw heights. */
  objects?: ObjectKinds;
}

async function buildGeometry(scene: Scene, o: MeshExportOptions, heights: Float32Array = scene.heights.data) {
  const out = await terrainWorker().build({
    heights: heights.slice(),
    width: scene.heights.width,
    height: scene.heights.height,
    gsd: scene.gsd,
    base: scene.stats.min,
    budget: o.budget,
    wallThreshold: o.wallThreshold,
    skirtDepth: 0,
  });
  const g = new THREE.BufferGeometry();
  const pos = out.terrain.positions;
  if (o.exaggeration !== 1) for (let i = 1; i < pos.length; i += 3) pos[i] *= o.exaggeration;
  g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(out.terrain.normals, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(out.terrain.uvs, 2));
  g.setIndex(new THREE.BufferAttribute(out.terrain.index, 1));
  return g;
}

async function imageCanvas(blob: Blob, maxSide = 4096) {
  const probe = await createImageBitmap(blob);
  const s = Math.min(1, maxSide / Math.max(probe.width, probe.height));
  const w = Math.round(probe.width * s);
  const h = Math.round(probe.height * s);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d')!.drawImage(probe, 0, 0, w, h);
  probe.close();
  return canvas;
}

/** Binary glTF (Y-up, metres, origin at the scene centre) with the optical image draped. */
export async function exportGlb(scene: Scene, o: MeshExportOptions): Promise<Blob> {
  // with objects, the terrain is the flattened surface they stand on (same worker result as the 3D view)
  const kinds = o.objects ? activeObjectKinds(o.objects, scene.objects) : null;
  const surface = kinds && anyObjectKind(kinds) ? await loadObjectSurface(scene, kinds) : null;
  const [{ GLTFExporter }, geometry, canvas] = await Promise.all([
    import('three/addons/exporters/GLTFExporter.js'),
    buildGeometry(scene, o, surface ?? scene.heights.data),
    imageCanvas(scene.image),
  ]);
  const tex = new THREE.CanvasTexture(canvas);
  tex.flipY = false; // UVs follow glTF convention (v = 0 at the image top)
  tex.colorSpace = THREE.SRGBColorSpace;
  const mesh = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({ map: tex, roughness: 0.95, metalness: 0 }));
  mesh.name = `${scene.name} terrain`;
  // With objects, the file is a small scene: terrain + trees (EXT_mesh_gpu_instancing) + roofs, walls, water.
  const layer = surface && kinds ? buildObjectLayer(scene, surface, kinds) : null;
  let root: THREE.Object3D = mesh;
  if (layer) {
    layer.setRoofTexture(tex);
    layer.group.scale.y = o.exaggeration;
    root = new THREE.Group();
    root.name = scene.name;
    root.add(mesh, layer.group);
  }
  root.userData = {
    generator: 'DepthWizard',
    product: scene.product,
    gsd_m: scene.gsd,
    height_units: 'metres',
    base_height_m: scene.stats.min,
    vertical_exaggeration: o.exaggeration,
    axes: 'x east, y up, z south; origin at scene centre',
    crs_epsg: scene.georef?.epsg ?? null,
    transform: scene.georef?.transform ?? null,
    ...(layer && kinds && scene.objects
      ? {
          objects: {
            trees: kinds.trees ? scene.objects.trees.length : 0,
            buildings: kinds.buildings ? scene.objects.buildings.length : 0,
            water: kinds.water ? scene.objects.water.length : 0,
          },
          terrain_note: 'terrain flattened under the object footprints; raw heights are in ndsm_m.npy',
        }
      : {}),
  };
  const exporter = new GLTFExporter();
  const result = (await exporter.parseAsync(root, { binary: true, maxTextureSize: 4096 })) as ArrayBuffer;
  geometry.dispose();
  layer?.dispose();
  tex.dispose();
  return new Blob([result], { type: 'model/gltf-binary' });
}

/** Scene axes (x east, y up, z south) → GIS-style Z-up (X east, Y north, Z = absolute height). */
function toZUp(geometry: THREE.BufferGeometry, base: number, exaggeration: number) {
  const src = geometry.getAttribute('position') as THREE.BufferAttribute;
  const out = new Float32Array(src.count * 3);
  for (let i = 0; i < src.count; i++) {
    out[i * 3] = src.getX(i);
    out[i * 3 + 1] = -src.getZ(i);
    out[i * 3 + 2] = src.getY(i) / exaggeration + base;
  }
  return out;
}

function header(scene: Scene, o: MeshExportOptions) {
  return [
    `DepthWizard terrain export — ${scene.name}`,
    `product ${scene.product}, gsd ${scene.gsd} m/px, units metres, axes X east / Y north / Z height`,
    `origin at scene centre${scene.georef ? `; CRS EPSG:${scene.georef.epsg ?? 'unknown'} transform ${scene.georef.transform.join(' ')}` : ' (not georeferenced)'}`,
    o.exaggeration !== 1 ? `vertical exaggeration ${o.exaggeration}` : 'true vertical scale',
  ];
}

/** OBJ + MTL + texture in a zip. */
export async function exportObjZip(scene: Scene, o: MeshExportOptions): Promise<Blob> {
  const [geometry, canvas] = await Promise.all([buildGeometry(scene, { ...o, exaggeration: 1 }), imageCanvas(scene.image)]);
  const pos = toZUp(geometry, scene.stats.min, 1);
  const uv = geometry.getAttribute('uv') as THREE.BufferAttribute;
  const nrm = geometry.getAttribute('normal') as THREE.BufferAttribute;
  const idx = geometry.getIndex()!;
  const lines: string[] = header(scene, o).map((l) => `# ${l}`);
  lines.push('mtllib terrain.mtl', 'o terrain', 'usemtl terrain');
  for (let i = 0; i < pos.length; i += 3) lines.push(`v ${pos[i].toFixed(3)} ${pos[i + 1].toFixed(3)} ${pos[i + 2].toFixed(3)}`);
  // OBJ texture space has v = 0 at the image bottom.
  for (let i = 0; i < uv.count; i++) lines.push(`vt ${uv.getX(i).toFixed(6)} ${(1 - uv.getY(i)).toFixed(6)}`);
  for (let i = 0; i < nrm.count; i++) lines.push(`vn ${nrm.getX(i).toFixed(4)} ${(-nrm.getZ(i)).toFixed(4)} ${nrm.getY(i).toFixed(4)}`);
  for (let i = 0; i < idx.count; i += 3) {
    // (x, y, z) → (x, −z, y) is a proper rotation, so the winding is preserved.
    const a = idx.getX(i) + 1;
    const b = idx.getX(i + 1) + 1;
    const c = idx.getX(i + 2) + 1;
    lines.push(`f ${a}/${a}/${a} ${b}/${b}/${b} ${c}/${c}/${c}`);
  }
  const mtl = ['newmtl terrain', 'Ka 1 1 1', 'Kd 1 1 1', 'Ks 0 0 0', 'd 1', 'illum 1', 'map_Kd texture.png'].join('\n');
  const png = await new Promise<Blob>((res) => canvas.toBlob((b) => res(b!), 'image/png'));
  geometry.dispose();
  const zip = zipSync({
    'terrain.obj': strToU8(lines.join('\n')),
    'terrain.mtl': strToU8(mtl),
    'texture.png': new Uint8Array(await png.arrayBuffer()),
  });
  return new Blob([zip as BlobPart], { type: 'application/zip' });
}

/** Binary PLY with per-vertex colormap colours (Z-up, metres). */
export async function exportPly(scene: Scene, o: MeshExportOptions): Promise<Blob> {
  const geometry = await buildGeometry(scene, { ...o, exaggeration: 1 });
  const pos = toZUp(geometry, scene.stats.min, 1);
  const idx = geometry.getIndex()!;
  const n = pos.length / 3;
  const faces = idx.count / 3;
  const head = [
    'ply',
    'format binary_little_endian 1.0',
    ...header(scene, o).map((l) => `comment ${l}`),
    `element vertex ${n}`,
    'property float x',
    'property float y',
    'property float z',
    'property uchar red',
    'property uchar green',
    'property uchar blue',
    `element face ${faces}`,
    'property list uchar int vertex_indices',
    'end_header\n',
  ].join('\n');
  const headBytes = strToU8(head);
  const buf = new ArrayBuffer(headBytes.length + n * 15 + faces * 13);
  const bytes = new Uint8Array(buf);
  bytes.set(headBytes, 0);
  const dv = new DataView(buf);
  let off = headBytes.length;
  const [lo, hi] = o.range;
  for (let i = 0; i < n; i++) {
    dv.setFloat32(off, pos[i * 3], true);
    dv.setFloat32(off + 4, pos[i * 3 + 1], true);
    dv.setFloat32(off + 8, pos[i * 3 + 2], true);
    const [r, g, b] = colorAt(o.colormap, (pos[i * 3 + 2] - lo) / (hi - lo || 1));
    bytes[off + 12] = r;
    bytes[off + 13] = g;
    bytes[off + 14] = b;
    off += 15;
  }
  for (let f = 0; f < faces; f++) {
    bytes[off] = 3;
    dv.setInt32(off + 1, idx.getX(f * 3), true);
    dv.setInt32(off + 5, idx.getX(f * 3 + 1), true);
    dv.setInt32(off + 9, idx.getX(f * 3 + 2), true);
    off += 13;
  }
  geometry.dispose();
  return new Blob([buf], { type: 'application/octet-stream' });
}

/** Binary STL (Z-up, metres) — e.g. for 3D printing a terrain model. */
export async function exportStl(scene: Scene, o: MeshExportOptions): Promise<Blob> {
  const geometry = await buildGeometry(scene, o);
  const pos = toZUp(geometry, scene.stats.min, 1);
  const idx = geometry.getIndex()!;
  const faces = idx.count / 3;
  const buf = new ArrayBuffer(84 + faces * 50);
  const bytes = new Uint8Array(buf);
  bytes.set(strToU8(`DepthWizard ${scene.name} (metres, Z-up)`.slice(0, 80)), 0);
  const dv = new DataView(buf);
  dv.setUint32(80, faces, true);
  let off = 84;
  const v = (k: number) => [pos[k * 3], pos[k * 3 + 1], pos[k * 3 + 2]];
  for (let f = 0; f < faces; f++) {
    const a = v(idx.getX(f * 3));
    const b = v(idx.getX(f * 3 + 1));
    const c = v(idx.getX(f * 3 + 2));
    const ux = b[0] - a[0], uy = b[1] - a[1], uz = b[2] - a[2];
    const wx = c[0] - a[0], wy = c[1] - a[1], wz = c[2] - a[2];
    let nx = uy * wz - uz * wy, ny = uz * wx - ux * wz, nz = ux * wy - uy * wx;
    const l = Math.hypot(nx, ny, nz) || 1;
    nx /= l; ny /= l; nz /= l;
    for (const val of [nx, ny, nz, ...a, ...b, ...c]) {
      dv.setFloat32(off, val, true);
      off += 4;
    }
    off += 2;
  }
  geometry.dispose();
  return new Blob([buf], { type: 'model/stl' });
}
