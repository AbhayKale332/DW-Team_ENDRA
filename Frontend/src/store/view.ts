import { create } from 'zustand';
import type { ColormapId } from '@/theme/colormaps';
import type { ObjectKind, ObjectKinds, SceneObjects } from '@/domain/types';
import type { PoiCategory } from '@/lib/poi';

export type ViewMode = 'dsm3d' | 'heightmap' | 'image';
export type DrapeLayer = 'tint' | 'optical' | 'height' | 'hillshade' | 'slope' | 'classes' | 'reference' | 'error';
export type RangeMode = 'robust' | 'full' | 'custom';

export const OBJECT_KINDS: ObjectKind[] = ['buildings', 'trees', 'water'];
export const OBJECT_KIND_LABELS: Record<ObjectKind, string> = { buildings: 'Buildings', trees: 'Trees', water: 'Water' };
/** Only buildings by default: the cleanest objects, and the ones a flythrough reads best. */
export const DEFAULT_OBJECT_KINDS: ObjectKinds = { buildings: true, trees: false, water: false };

export const VIEW_LABELS: Record<ViewMode, string> = {
  dsm3d: '3D DSM View',
  heightmap: 'Height Map',
  image: 'Input Image',
};

export const LAYER_LABELS: Record<DrapeLayer, string> = {
  tint: 'Optical + height tint',
  optical: 'Optical image',
  height: 'Height (colormap)',
  hillshade: 'Hillshade',
  slope: 'Slope (°)',
  classes: 'Object classes',
  reference: 'Reference height',
  error: 'Error vs reference',
};

interface ViewState {
  mode: ViewMode;
  /** Drape used by the 3D DSM view. The 2D views have fixed layers (height map / optical). */
  layer3d: DrapeLayer;
  /** Layer for the Height Map view. */
  layer2d: DrapeLayer;
  colormap: ColormapId;
  rangeMode: RangeMode;
  customRange: [number, number];
  tintOpacity: number;
  hillshadeStrength: number;
  slopeMax: number;
  exaggeration: number;
  sunAzimuth: number;
  sunElevation: number;
  shadows: boolean;
  walls: boolean;
  wallThreshold: number;
  contours: boolean;
  contourInterval: number;
  wireframe: boolean;
  /** Which object classes are drawn as 3D models (over a mesh flattened under them); all off = the raw mesh.
   *  Only has an effect when the scene carries objects (scene.objects). */
  objectKinds: ObjectKinds;
  /** Hover card with object / area details under the pointer. */
  hoverInfo: boolean;
  /** OpenStreetMap overlay (georeferenced scenes only; fetched from the Overpass API when switched on). */
  osm: boolean;
  /** Map tiles around the processed area (georeferenced scenes only; display only, never processed). */
  basemap: boolean;
  /** OpenStreetMap facilities (hospitals, schools, stations …) over the scene and its surroundings. */
  poi: boolean;
  poiCategories: Record<PoiCategory, boolean>;
  compareSwipe: boolean;
  swipe: number;
  /** Raw model output: 3D objects and post-processing are off (the previous object selection is restored on exit). */
  raw: boolean;
  /** Masked clouds: hatch and outline the filled areas. */
  cloudHatch: boolean;
  /** Masked clouds: drape the original (cloudy) image instead of the cloud-free one. */
  cloudOriginal: boolean;
  set: (p: Partial<Omit<ViewState, 'set' | 'reset'>>) => void;
  reset: () => void;
}

const defaults = {
  mode: 'dsm3d' as ViewMode,
  layer3d: 'optical' as DrapeLayer,
  layer2d: 'height' as DrapeLayer,
  colormap: 'terrain' as ColormapId,
  rangeMode: 'robust' as RangeMode,
  customRange: [0, 30] as [number, number],
  tintOpacity: 0.55,
  hillshadeStrength: 0.45,
  slopeMax: 60,
  exaggeration: 1,
  sunAzimuth: 315,
  sunElevation: 45,
  shadows: true,
  walls: true,
  wallThreshold: 2.5,
  contours: false,
  contourInterval: 5,
  wireframe: false,
  objectKinds: DEFAULT_OBJECT_KINDS,
  hoverInfo: true,
  // off by default: switching it on sends the scene's bounding box to a third-party service
  osm: false,
  // on by default, but only for scenes that can be placed on the globe (canGeolocate)
  basemap: true,
  poi: true,
  poiCategories: { emergency: true, education: true, civic: true, transport: true } as Record<PoiCategory, boolean>,
  compareSwipe: false,
  swipe: 0.5,
  raw: false,
  cloudHatch: true,
  cloudOriginal: false,
};

export const useView = create<ViewState>()((set) => ({
  ...defaults,
  set: (p) => set(p),
  reset: () => set(defaults),
}));

/** The classes actually drawn for a scene: switched on *and* present in its objects. */
export function activeObjectKinds(kinds: ObjectKinds, objects: SceneObjects | null | undefined): ObjectKinds {
  return {
    buildings: kinds.buildings && !!objects?.buildings.length,
    trees: kinds.trees && !!objects?.trees.length,
    water: kinds.water && !!objects?.water.length,
  };
}

export const anyObjectKind = (k: ObjectKinds) => k.buildings || k.trees || k.water;

/** Stable cache / dependency key for a class selection, e.g. "b1t0w0". */
export const objectKindsKey = (k: ObjectKinds) => `b${+k.buildings}t${+k.trees}w${+k.water}`;

/** One class on/off (tool palette, Tools menu, Layers tab). */
export function toggleObjectKind(kind: ObjectKind) {
  const v = useView.getState();
  v.set({ objectKinds: { ...v.objectKinds, [kind]: !v.objectKinds[kind] } });
}

let lastSelection: ObjectKinds = DEFAULT_OBJECT_KINDS;

/** Key O: everything off, or back to the last selection (buildings only if there was none). */
export function toggleAllObjects() {
  const v = useView.getState();
  if (anyObjectKind(v.objectKinds)) {
    lastSelection = v.objectKinds;
    v.set({ objectKinds: { buildings: false, trees: false, water: false } });
  } else {
    v.set({ objectKinds: lastSelection });
  }
}

export function activeLayer(s: Pick<ViewState, 'mode' | 'layer3d' | 'layer2d'>): DrapeLayer {
  if (s.mode === 'image') return 'optical';
  if (s.mode === 'heightmap') return s.layer2d;
  return s.layer3d;
}

let selectionBeforeRaw: ObjectKinds | null = null;

/** Raw mode: show the model's height field as is. 3D object models (and the mesh flattening under them) and the
 *  ambient-occlusion post effect go off; leaving restores the object selection that was on. */
export function setRawOutput(on: boolean) {
  const v = useView.getState();
  if (on === v.raw) return;
  if (on) {
    selectionBeforeRaw = v.objectKinds;
    v.set({ raw: true, objectKinds: { buildings: false, trees: false, water: false } });
  } else {
    v.set({ raw: false, objectKinds: selectionBeforeRaw ?? DEFAULT_OBJECT_KINDS });
    selectionBeforeRaw = null;
  }
}
