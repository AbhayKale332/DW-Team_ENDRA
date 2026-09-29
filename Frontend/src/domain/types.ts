/** Normalised domain model. Everything the UI renders is expressed in these types,
 *  independent of which inference provider produced it. */

export type Product = 'rDSM' | 'nDSM' | 'DSM';
export type GsdSource = 'user' | 'geotiff' | 'assumed';

/** GDAL-order affine transform: x = a*col + b*row + c ; y = d*col + e*row + f  (pixel corner). */
export type Affine = [number, number, number, number, number, number];

export interface Georef {
  epsg: number | null;
  /** proj4 definition string when known (EPSG:4326 and UTM are generated locally). */
  proj4: string | null;
  transform: Affine;
}

export interface HeightStats {
  min: number;
  max: number;
  mean: number;
  median: number;
  p2: number;
  p98: number;
  fracBelow1m: number;
  valid: number;
}

export interface HeightGrid {
  data: Float32Array;
  width: number;
  height: number;
}

/** Per-pixel object class from the model's segmentation head, on the height grid (`seg.png`). */
export interface ClassMap {
  /** Class id per pixel, row-major, same width/height as the scene's HeightGrid. */
  data: Uint8Array;
  width: number;
  height: number;
  /** Class name by id (e.g. names[6] = 'tree'), from meta.json `classes`. */
  names: string[];
}

/** A point in height-grid pixels, origin at the grid's top-left *corner*: pixel (row, col) spans
 *  x ∈ [col, col + 1), y ∈ [row, row + 1), so its centre is (col + 0.5, row + 0.5). */
export type GridPoint = [number, number];

export interface TreeObject {
  /** Crown centroid, grid pixels (corner origin). */
  x: number;
  y: number;
  /** Top of the crown, metres above ground. */
  h: number;
  /** Crown radius, metres. */
  r: number;
}

export interface BuildingObject {
  /** Outer footprint ring, grid pixels, first point not repeated, clockwise on screen (x right, y down). */
  poly: GridPoint[];
  /** Median roof height, metres. */
  h: number;
  /** 95th-percentile roof height, metres. */
  hMax: number;
  areaM2: number;
}

export interface WaterObject {
  poly: GridPoint[];
  areaM2: number;
}

/** Discrete 3D objects extracted from the class + height maps (`objects.json`), on the height grid. */
export interface SceneObjects {
  version: 1;
  /** Grid the positions refer to — always equal to the scene's HeightGrid after parsing. */
  width: number;
  height: number;
  gsd: number;
  /** Tallest first. */
  trees: TreeObject[];
  /** Largest first. */
  buildings: BuildingObject[];
  /** Largest first. */
  water: WaterObject[];
  /** True where the backend hit its cap and dropped the smallest objects. */
  truncated: { trees: boolean; buildings: boolean; water: boolean };
}

/** Object classes that can be drawn as 3D models, each switched on or off independently. */
export type ObjectKind = 'buildings' | 'trees' | 'water';
export type ObjectKinds = Record<ObjectKind, boolean>;

/** Scene contract written by the backend (meta.json). Only fields the UI reads are typed. */
export interface SceneMeta {
  stem?: string;
  product?: Product;
  units?: string;
  height_min_m?: number;
  height_max_m?: number;
  height_mean_m?: number;
  height_median_m?: number;
  frac_below_1m?: number;
  size_px?: [number, number];
  scene?: {
    width?: number;
    height?: number;
    gsd_m?: number;
    gsd_source?: GsdSource;
    georeferenced?: boolean;
    crs?: string | null;
    crs_epsg?: number | null;
    transform?: number[] | null;
    path?: string;
  };
  preproc?: Record<string, unknown>;
  /** `seg.png` legend: pixel value (as a string key) -> class name. */
  classes?: Record<string, string>;
  class_fractions?: Record<string, number>;
  files?: string[];
  ndsm16_encode?: string;
  note?: string;
  [k: string]: unknown;
}

export interface Artefact {
  name: string;
  url?: string;
  blob?: Blob;
  size?: number;
}

export type WarningLevel = 'info' | 'warning';
export interface SceneWarning {
  id: string;
  level: WarningLevel;
  title: string;
  message: string;
}

export interface Provenance {
  provider: string;
  source: 'inference' | 'sample' | 'bundle' | 'project';
  createdAt: string;
  params?: { gsd: number | null; tta: boolean };
  modelVersion?: string;
}

export interface Scene {
  id: string;
  name: string;
  /** The image the user supplied (original resolution) — used as the 3D drape. */
  image: Blob;
  imageWidth: number;
  imageHeight: number;
  heights: HeightGrid;
  /** Object classes on the height grid; absent for older backends, samples and imported bundles. */
  classes?: ClassMap | null;
  /** Trees, buildings and water as 3D objects; absent for older backends, samples and imported bundles. */
  objects?: SceneObjects | null;
  gsd: number;
  gsdSource: GsdSource;
  product: Product;
  stats: HeightStats;
  georef: Georef | null;
  meta: SceneMeta;
  artefacts: Artefact[];
  warnings: SceneWarning[];
  statusLines: string[];
  provenance: Provenance;
}

export type ReferenceKind = 'nDSM' | 'DSM';

export interface ReferenceSurface {
  name: string;
  kind: ReferenceKind;
  /** Resampled onto the prediction grid (same width/height as scene.heights). NaN = no data. */
  data: Float32Array;
  alignment: 'georeferenced' | 'same-extent';
  notes: string[];
}

export interface Metrics {
  n: number;
  rmse: number;
  mae: number;
  bias: number;
  r: number;
  within1m: number;
  within2m: number;
}

export interface StratumMetrics extends Metrics {
  label: string;
  lo: number;
  hi: number;
}

export interface ValidationResult {
  all: Metrics;
  strata: StratumMetrics[];
  balancedRmse: number;
  biasRemoved: boolean;
  offset: number;
  scatter: { bins: number; lo: number; hi: number; counts: Uint32Array };
  errorHist: { lo: number; hi: number; counts: Uint32Array };
}

export interface CameraBookmark {
  id: string;
  name: string;
  position: [number, number, number];
  target: [number, number, number];
}
