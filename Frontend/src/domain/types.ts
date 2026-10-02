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

/** Per-pixel uncertainty of the heights (`ndsm_std_m.npy`, v5 backends): the spread of the model's height bins, one
 *  standard deviation in metres, on the height grid. NaN = none (masked, under cloud). */
export interface UncertaintyGrid extends HeightGrid {
  /** Pixels with σ at or below this are "confident": the backend's cut, max(1 m, the scene's median σ). */
  confidentM: number;
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
  /** Summary of `ndsm_std_m.npy`, when the backend wrote one (v5 infer/predict.py `uncertainty_summary`). */
  uncertainty?: {
    std_median_m?: number;
    std_p90_m?: number;
    confident_threshold_m?: number;
    confident_frac?: number;
  };
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
  params?: { gsd: number | null; tta: boolean; cloudMask?: boolean };
  modelVersion?: string;
}

/** Where an absolute DSM's elevations come from. Set only by DEM anchoring (lib/dem). */
export type VerticalDatum = 'EGM96' | 'EGM2008' | 'ellipsoid' | 'unknown';

export interface AnchoringInfo {
  /** Human label of the DEM, e.g. "AWS Terrain Tiles (SRTM-based mosaic)". */
  source: string;
  sourceId: 'terrain-tiles' | 'local-file' | 'bundled-cache' | 'gcp';
  datum: VerticalDatum;
  /** Anchor cell size in metres (the DEM's native resolution, ~30 m). */
  cellM: number;
  cellPx: number;
  demMinM: number;
  demMaxM: number;
  /** RMS over cells of (mean of DSM in the cell) - DEM. Near 0 by construction; a sanity check, not an accuracy. */
  cellMeanRmseM: number;
  detailGain: number;
  /** Share of cell-mean structure height assumed to be inside the DEM (0 = bare terrain, 1 = full surface model). */
  structureShare: number;
  /** Mean of (DSM cell mean - DEM cell), metres: how far the DSM sits above the DEM on average. */
  meanOffsetM: number;
  /** The sampled DEM cells (row-major, `cellPx` pixels each), kept so the share can change without a re-fetch. */
  demCells?: { rows: number; cols: number; data: Float64Array };
  fetchedAt: string;
  tileZoom?: number;
  notes: string[];
}

/** A ground control point: a scene pixel with known coordinates and/or elevation. */
export interface GroundControlPoint {
  id: string;
  /** Grid pixel, fractional, pixel centres at integers. */
  col: number;
  row: number;
  /** WGS84 degrees; both or neither. Three or more georeference the scene. */
  lat: number | null;
  lon: number | null;
  /** Surface elevation at this pixel, metres. Any makes the heights absolute. */
  elev: number | null;
}

/** Clouds masked out of a scene: the model saw the cloud-free image, the heights under them are filled. */
export interface SceneCloud {
  /** Soft alpha (0 clear .. 255 cloud, >= 128 the cloud proper) on its own grid, same extent as the image. */
  mask: Uint8Array;
  width: number;
  height: number;
  /** Share of the image under cloud. */
  coverage: number;
  /** The image with the clouds painted over (the drape, unless the original is asked for). */
  image: Blob;
}

export interface Scene {
  id: string;
  name: string;
  /** The image the user supplied (original resolution) — used as the 3D drape. */
  image: Blob;
  imageWidth: number;
  imageHeight: number;
  /** The primary product: absolute elevation when `product === 'DSM'`, else height above ground. */
  heights: HeightGrid;
  /** Height above ground on the same grid; kept when `heights` was replaced by the absolute DSM. */
  ndsm?: HeightGrid;
  /** Bare-earth terrain elevation (absolute), only for DEM-anchored scenes. */
  terrain?: HeightGrid;
  anchoring?: AnchoringInfo;
  /** Ground control points applied to a scene without its own georeferencing. */
  gcps?: GroundControlPoint[];
  /** Per-pixel σ of the heights; absent when the backend did not return `ndsm_std_m.npy` (the v3 Space does not). */
  uncertainty?: UncertaintyGrid | null;
  /** Object classes on the height grid; absent for older backends, samples and imported bundles. */
  classes?: ClassMap | null;
  /** Trees, buildings and water as 3D objects; absent for older backends, samples and imported bundles. */
  objects?: SceneObjects | null;
  /** Clouds masked out of the model input, when there were any and masking was on. */
  cloud?: SceneCloud | null;
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

/** Sparsification curves of one scene, after Poggi et al., CVPR 2020 (arXiv 2005.06209), with the maths of their reference
 *  code (mono-uncertainty `compute_aucs`): remove the most uncertain pixels first, 1/K at a time, and take the RMSE of the
 *  rest. The oracle removes by the true error, the best any σ could do; random removal keeps the RMSE flat. */
export interface Sparsification {
  /** Share of pixels removed: 0, 1/K … 1. */
  removed: number[];
  /** RMSE of the pixels left, most uncertain removed first (ends at 0 at 1, as the reference code). */
  bySigma: number[];
  /** The same, removed by true error. */
  oracle: number[];
  /** RMSE of every pixel: the random-removal line. */
  rmse: number;
  /** Area between the σ curve and the oracle, metres: 0 = σ ranks pixels exactly as their error. */
  ause: number;
  /** RMSE minus the area under the σ curve, metres: > 0 = better than removing pixels at random. */
  aurg: number;
  /** Pixels used (evenly strided on big scenes). */
  n: number;
}

/** What σ says about a validated scene: the error on its confident pixels, and its sparsification curves. */
export interface UncertaintyValidation {
  /** The confident cut, metres. */
  confidentM: number;
  /** Metrics on compared pixels with σ at or below the cut. */
  confident: Metrics;
  /** Share of compared pixels that are confident (pixels without σ count as not confident). */
  coverage: number;
  sparsification: Sparsification | null;
}

export interface ValidationResult {
  all: Metrics;
  strata: StratumMetrics[];
  balancedRmse: number;
  biasRemoved: boolean;
  offset: number;
  scatter: { bins: number; lo: number; hi: number; counts: Uint32Array };
  errorHist: { lo: number; hi: number; counts: Uint32Array };
  /** Only when the scene carries σ. */
  uncertainty?: UncertaintyValidation | null;
}

export interface CameraBookmark {
  id: string;
  name: string;
  position: [number, number, number];
  target: [number, number, number];
}
