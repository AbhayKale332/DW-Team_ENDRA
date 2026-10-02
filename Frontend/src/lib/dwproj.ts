/** The .dwproj container: a zip holding everything needed to reopen a workspace on any machine, offline.
 *
 *  manifest.json      scene, view, tools, scenarios, anchoring settings (Manifest below)
 *  input.png|jpg      the drape image
 *  ndsm_m.npy         height above ground, Float32 metres
 *  ndsm_std_m.npy     per-pixel σ of those heights, Float32 metres (optional; v5 backends only)
 *  seg.png            object classes (optional)
 *  cloud_mask.png     clouds masked out of the model input, soft alpha (optional; heights already filled)
 *  objects.json       3D objects, the Space's schema (optional)
 *  reference_m.npy    validation reference on the same grid (optional)
 *  dem_cells.json     DEM cells the scene was anchored with, so it re-anchors without a network (optional)
 *  osm.json           OpenStreetMap features on the scene grid (optional)
 *  pois.json          facilities on the scene grid (optional)
 *  source/<name>      the original input image, so the model can be run again (optional)
 *  outputs/<name>     the model's output files (optional)
 *
 *  Writers store manifest.json first and source/ + outputs/ last, so a project read as it downloads (projectStream)
 *  can be shown before its heavy extras arrive. Version 1 projects (no v2 fields and files) still open. */
import { strFromU8, Unzip, UnzipInflate, unzipSync } from 'fflate';
import type { AnchoringInfo, CameraBookmark, ClassMap, Georef, GroundControlPoint, HeightGrid, Provenance, ReferenceSurface, SceneMeta, SceneObjects } from '@/domain/types';
import { parseNpy } from './npy';
import { parseObjects } from './objects';
import { parseUncertainty } from './uncertainty';
import { decodeGray8Png } from './png';

export interface Manifest {
  format: 'dwproj';
  version: 1 | 2;
  name: string;
  imageName: string;
  gsd: number;
  gsdSource: 'user' | 'geotiff' | 'assumed';
  georef: Georef | null;
  meta: SceneMeta;
  statusLines: string[];
  provenance: Provenance;
  view: Record<string, unknown>;
  bookmarks: CameraBookmark[];
  tools: { probe: unknown; measure: unknown; profile: unknown };
  reference: { name: string; kind: ReferenceSurface['kind']; alignment: ReferenceSurface['alignment']; notes: string[] } | null;
  /** Present when the project carries `seg.png` (object classes). Absent in older projects. */
  classes?: { names: string[] } | null;
  // ---- version 2
  /** How the scene was anchored (the DEM cells are in dem_cells.json). */
  anchoring?: { source: string; sourceId: AnchoringInfo['sourceId']; structureShare: number; heightRef: 'dsm' | 'ndsm' } | null;
  /** The original input image, stored as `source/<name>`. */
  source?: { name: string; file: string } | null;
  /** Model output files, stored under `outputs/` (or pointing at a top-level file with the same content). */
  outputs?: Array<{ name: string; file: string }>;
  params?: { gsdMode: 'auto' | 'preset' | 'custom'; gsd: number; tta: boolean; cloudMask?: boolean };
  usecases?: Record<string, unknown>;
  removeOffset?: boolean;
  dismissed?: string[];
  /** Ground control points applied to a scene without its own georeferencing. */
  gcps?: GroundControlPoint[] | null;
  /** Present when the project carries `cloud_mask.png`. */
  cloud?: { coverage: number } | null;
}

export interface ProjectCore {
  manifest: Manifest;
  files: Record<string, Uint8Array>;
  image: Blob;
  heights: HeightGrid;
  /** σ of the heights (`ndsm_std_m.npy`), when the project carries one. */
  uncertainty: HeightGrid | null;
  classes: ClassMap | null;
  objects: SceneObjects | null;
}

export const buf = (u: Uint8Array) => u.slice().buffer as ArrayBuffer;
export const json = <T>(u: Uint8Array | undefined): T | null => {
  if (!u) return null;
  try {
    return JSON.parse(strFromU8(u)) as T;
  } catch {
    return null;
  }
};

/** Unzip a project and decode the parts every consumer needs. Throws when it is not a project. */
export const readProject = (bytes: Uint8Array): ProjectCore => readProjectFiles(unzipSync(bytes));

/** Decode the parts every consumer needs from a project's unzipped files. Throws when it is not a project. */
export function readProjectFiles(files: Record<string, Uint8Array>): ProjectCore {
  const manifest = json<Manifest>(files['manifest.json']);
  if (!manifest || manifest.format !== 'dwproj') throw new Error('Not a DepthWizard project');
  if (!files['ndsm_m.npy'] || !files[manifest.imageName]) throw new Error('The project is incomplete (no height map or image).');
  const arr = parseNpy(buf(files['ndsm_m.npy']));
  const [h, w] = arr.shape;
  const image = new Blob([files[manifest.imageName] as BlobPart], { type: /\.jpe?g$/i.test(manifest.imageName) ? 'image/jpeg' : 'image/png' });
  let classes: ClassMap | null = null;
  if (manifest.classes && files['seg.png']) {
    try {
      classes = { ...decodeGray8Png(buf(files['seg.png'])), names: manifest.classes.names };
    } catch (e) {
      console.warn('Project seg.png unreadable; class layer disabled', e);
    }
  }
  let objects: SceneObjects | null = null;
  if (files['objects.json']) {
    try {
      objects = parseObjects(json(files['objects.json']), { width: w, height: h, gsd: manifest.gsd });
    } catch (e) {
      console.warn('Project objects.json unreadable; 3D objects disabled', e);
    }
  }
  const uncertainty = files['ndsm_std_m.npy'] ? parseUncertainty(buf(files['ndsm_std_m.npy']), w, h) : null;
  return { manifest, files, image, heights: { data: arr.data, width: w, height: h }, uncertainty, classes, objects };
}

/** The heavy, optional part of a project: not needed to show the scene. */
export const isProjectExtra = (name: string) => name.startsWith('source/') || name.startsWith('outputs/');

/** Whether `files` already hold what showing the scene needs: the manifest, its image and the height map. */
export function hasProjectCore(files: Record<string, Uint8Array>) {
  const m = json<Manifest>(files['manifest.json']);
  return m?.format === 'dwproj' && !!files[m.imageName] && !!files['ndsm_m.npy'];
}

const concat = (parts: Uint8Array[]) => {
  if (parts.length === 1) return parts[0];
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let at = 0;
  for (const p of parts) {
    out.set(p, at);
    at += p.length;
  }
  return out;
};

/** Unzip a project as its bytes arrive. `onExtras` fires once, when the first source/ or outputs/ entry starts:
 *  every file stored before it is complete in `files` by then. Throws from `push` on a broken archive. */
export function projectStream(onExtras: () => void) {
  const files: Record<string, Uint8Array> = {};
  let extras = false;
  const unzip = new Unzip((f) => {
    if (!extras && isProjectExtra(f.name)) {
      extras = true;
      onExtras();
    }
    const parts: Uint8Array[] = [];
    f.ondata = (err, chunk, final) => {
      if (err) throw err;
      parts.push(chunk);
      if (final) files[f.name] = concat(parts);
    };
    f.start();
  });
  unzip.register(UnzipInflate);
  return { files, push: (chunk: Uint8Array, final = false) => unzip.push(chunk, final) };
}

const TYPES: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', tif: 'image/tiff', tiff: 'image/tiff', json: 'application/json', glb: 'model/gltf-binary' };
export const mimeOf = (name: string) => TYPES[name.split('.').pop()?.toLowerCase() ?? ''] ?? 'application/octet-stream';
