import * as THREE from 'three';
import { mergeGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import { TREE_BROADLEAF, TREE_CLUSTER, TREE_CONIFER, TREE_VARIANTS } from '@/lib/treeLayout';

/** Procedural low-poly tree models, unit-sized: base at y = 0, top at y = 1, crown radius 1 in x/z.
 *  An instance is scaled by (crown radius, height, crown radius) in metres, so a model never needs
 *  per-asset calibration. Colours are baked as vertex colours so one material serves every variant;
 *  the per-instance tint multiplies them. */

const TRUNK = new THREE.Color('#6b4a32');
const LEAF = new THREE.Color('#4f8a3c');
const LEAF_DEEP = new THREE.Color('#3f7a36');
const NEEDLE = new THREE.Color('#2f6138');

/** Crack-free organic wobble: displacement is keyed on the vertex *position*, so the copies of a shared
 *  vertex in a non-indexed mesh all move together. */
function wobble(geo: THREE.BufferGeometry, amount: number, seed: number) {
  const p = geo.getAttribute('position') as THREE.BufferAttribute;
  for (let i = 0; i < p.count; i++) {
    const x = p.getX(i);
    const y = p.getY(i);
    const z = p.getZ(i);
    const k = Math.sin(Math.round(x * 1e3) * 12.9898 + Math.round(y * 1e3) * 78.233 + Math.round(z * 1e3) * 37.719 + seed) * 43758.5453;
    const f = 1 + amount * ((k - Math.floor(k)) * 2 - 1);
    p.setXYZ(i, x * f, y * f, z * f);
  }
}

/** Normalise a primitive into the shared attribute layout (position, normal, color; non-indexed). */
function part(src: THREE.BufferGeometry, color: THREE.Color, place: THREE.Matrix4, wobbleAmount = 0, seed = 0) {
  const geo = src.index ? src.toNonIndexed() : src.clone();
  src.dispose();
  geo.deleteAttribute('uv');
  geo.deleteAttribute('normal');
  if (wobbleAmount) wobble(geo, wobbleAmount, seed);
  geo.applyMatrix4(place);
  const n = geo.getAttribute('position').count;
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) color.toArray(colors, i * 3);
  geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geo.computeVertexNormals();
  return geo;
}

const at = (x: number, y: number, z: number, sx = 1, sy = 1, sz = 1) => new THREE.Matrix4().compose(new THREE.Vector3(x, y, z), new THREE.Quaternion(), new THREE.Vector3(sx, sy, sz));

function merge(parts: THREE.BufferGeometry[]) {
  const out = mergeGeometries(parts, false);
  parts.forEach((g) => g.dispose());
  if (!out) throw new Error('tree model: incompatible parts');
  out.computeBoundingBox();
  out.computeBoundingSphere();
  return out;
}

/** One dome: the usual broadleaf. Crown spans y 0.3–1.0. */
function broadleaf() {
  return merge([
    part(new THREE.CylinderGeometry(0.07, 0.1, 0.5, 6), TRUNK, at(0, 0.25, 0)),
    part(new THREE.IcosahedronGeometry(1, 1), LEAF, at(0, 0.65, 0, 0.97, 0.35, 0.97), 0.06, 1),
  ]);
}

/** Three lobes: a broader, clumpier broadleaf for variety. */
function cluster() {
  return merge([
    part(new THREE.CylinderGeometry(0.07, 0.1, 0.5, 6), TRUNK, at(0, 0.25, 0)),
    part(new THREE.IcosahedronGeometry(1, 1), LEAF_DEEP, at(-0.34, 0.6, 0.12, 0.62, 0.3, 0.62), 0.07, 2),
    part(new THREE.IcosahedronGeometry(1, 1), LEAF, at(0.34, 0.62, -0.12, 0.62, 0.3, 0.62), 0.07, 3),
    part(new THREE.IcosahedronGeometry(1, 1), LEAF, at(0, 0.78, 0, 0.55, 0.22, 0.55), 0.07, 4),
  ]);
}

/** Stacked cones: for trees much taller than they are wide. */
function conifer() {
  return merge([
    part(new THREE.CylinderGeometry(0.06, 0.09, 0.3, 6), TRUNK, at(0, 0.15, 0)),
    part(new THREE.ConeGeometry(1, 0.5, 8), NEEDLE, at(0, 0.4, 0)),
    part(new THREE.ConeGeometry(0.72, 0.42, 8), NEEDLE, at(0, 0.6, 0)),
    part(new THREE.ConeGeometry(0.45, 0.36, 8), NEEDLE, at(0, 0.82, 0)),
  ]);
}

/** Geometries indexed by variant (see lib/treeLayout). The caller owns and disposes them. */
export function createTreeModels(): THREE.BufferGeometry[] {
  const models: THREE.BufferGeometry[] = [];
  models[TREE_BROADLEAF] = broadleaf();
  models[TREE_CLUSTER] = cluster();
  models[TREE_CONIFER] = conifer();
  if (models.length !== TREE_VARIANTS) throw new Error('tree model list out of step with TREE_VARIANTS');
  return models;
}
