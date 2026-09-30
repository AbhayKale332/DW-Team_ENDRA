/** The .dwproj container: a zip holding everything needed to reopen a workspace on any machine, offline.
 *
 *  manifest.json      scene, view, tools, scenarios, anchoring settings (Manifest below)
 *  input.png|jpg      the drape image
 *  ndsm_m.npy         height above ground, Float32 metres
 *  seg.png            object classes (optional)
 *  objects.json       3D objects, the Space's schema (optional)
 *  reference_m.npy    validation reference on the same grid (optional)
 *  dem_cells.json     DEM cells the scene was anchored with, so it re-anchors without a network (optional)
 *  osm.json           OpenStreetMap features on the scene grid (optional)
 *  pois.json          facilities on the scene grid (optional)
 *  source/<name>      the original input image, so the model can be run again (optional)
 *  outputs/<name>     the model's output files (optional)
 *
 *  Version 1 projects (no v2 fields and files) still open. */
import { strFromU8, unzipSync } from 'fflate';
import type { AnchoringInfo, CameraBookmark, ClassMap, Georef, HeightGrid, Provenance, ReferenceSurface, SceneMeta, SceneObjects } from '@/domain/types';
import { parseNpy } from './npy';
import { parseObjects } from './objects';
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
  params?: { gsdMode: 'auto' | 'preset' | 'custom'; gsd: number; tta: boolean };
  usecases?: Record<string, unknown>;
  removeOffset?: boolean;
  dismissed?: string[];
}

export interface ProjectCore {
  manifest: Manifest;
  files: Record<string, Uint8Array>;
  image: Blob;
  heights: HeightGrid;
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
export function readProject(bytes: Uint8Array): ProjectCore {
  const files = unzipSync(bytes);
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
  return { manifest, files, image, heights: { data: arr.data, width: w, height: h }, classes, objects };
}

const TYPES: Record<string, string> = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', tif: 'image/tiff', tiff: 'image/tiff', json: 'application/json', glb: 'model/gltf-binary' };
export const mimeOf = (name: string) => TYPES[name.split('.').pop()?.toLowerCase() ?? ''] ?? 'application/octet-stream';
