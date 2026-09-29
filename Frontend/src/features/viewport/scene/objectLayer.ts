import * as THREE from 'three';
import type { ObjectKinds, Scene, TreeObject } from '@/domain/types';
import { sampleBilinear } from '@/lib/heights';
import { layoutTrees, TREE_VARIANTS, type TreeInstance } from '@/lib/treeLayout';
import { buildingGeometry, waterGeometry, type MeshArrays, type ObjectFrame } from '@/lib/objectGeometry';
import { createTreeModels } from './treeModels';

/** Everything in scene.objects as one three.js group — trees (instanced), building roofs and walls, water —
 *  in the terrain's world frame. Used by the viewport (Objects.tsx) and the GLB exporter, so the export is
 *  exactly what is on screen. The group is unscaled: the caller applies vertical exaggeration. */
export interface ObjectLayer {
  group: THREE.Group;
  /** The optical image for the roofs (uv = image position); null restores the plain roof colour. */
  setRoofTexture(t: THREE.Texture | null): void;
  setShadows(on: boolean): void;
  dispose(): void;
}

/** The frame objects stand in: `surface` is the flattened 3D-objects surface (objectSurfaceClient), so nothing
 *  rests on a bump it replaced. */
export function objectFrame(scene: Scene, surface: Float32Array): ObjectFrame {
  const { width, height } = scene.heights;
  const ground = { data: surface, width, height };
  // the raw height above ground: the nDSM of an anchored scene, else the heights themselves
  const agl = scene.ndsm ?? scene.heights;
  return { width, height, gsd: scene.gsd, base: scene.stats.min, roofOnGround: scene.product === 'DSM', groundAt: (c, r) => sampleBilinear(ground, c, r), aglAt: (c, r) => sampleBilinear(agl, c, r) };
}

function toGeometry(a: MeshArrays) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(a.positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(a.normals, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(a.uvs, 2));
  g.setIndex(new THREE.BufferAttribute(a.index, 1));
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

/** One window per facade tile (a bay wide, a storey tall — see objectGeometry). White wall so the material
 *  colour sets it. Null where there is no 2D canvas (tests). */
function facadeTexture(): THREE.CanvasTexture | null {
  if (typeof document === 'undefined') return null;
  const canvas = document.createElement('canvas');
  canvas.width = 64;
  canvas.height = 64;
  let ctx: CanvasRenderingContext2D | null = null;
  try {
    ctx = canvas.getContext('2d');
  } catch {
    return null;
  }
  if (!ctx) return null;
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, 64, 64);
  ctx.fillStyle = '#e9e6e0'; // floor band
  ctx.fillRect(0, 58, 64, 6);
  ctx.fillStyle = '#56626e'; // window
  ctx.fillRect(17, 14, 30, 30);
  ctx.fillStyle = '#7d8a96'; // sky reflection in the upper pane
  ctx.fillRect(19, 16, 26, 10);
  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

function treeMeshes(trees: TreeObject[], frame: ObjectFrame, models: THREE.BufferGeometry[], material: THREE.Material) {
  if (!trees.length) return [];
  const placed = layoutTrees(trees, frame, frame.groundAt);
  const byVariant: TreeInstance[][] = Array.from({ length: TREE_VARIANTS }, () => []);
  for (const t of placed) byVariant[t.variant].push(t);
  const m4 = new THREE.Matrix4();
  const q = new THREE.Quaternion();
  const up = new THREE.Vector3(0, 1, 0);
  const pos = new THREE.Vector3();
  const scl = new THREE.Vector3();
  const col = new THREE.Color();
  const out: THREE.InstancedMesh[] = [];
  byVariant.forEach((list, v) => {
    if (!list.length) return;
    const mesh = new THREE.InstancedMesh(models[v], material, list.length);
    mesh.name = `trees-${v}`;
    list.forEach((t, i) => {
      mesh.setMatrixAt(i, m4.compose(pos.set(t.x, t.y, t.z), q.setFromAxisAngle(up, t.rot), scl.set(t.sx, t.sy, t.sz)));
      mesh.setColorAt(i, col.setScalar(t.tint));
    });
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
    mesh.computeBoundingSphere();
    mesh.computeBoundingBox();
    out.push(mesh);
  });
  return out;
}

/** `surface` must be the one flattened for the same `kinds` (objectSurfaceClient.loadObjectSurface). */
export function buildObjectLayer(scene: Scene, surface: Float32Array, kinds: ObjectKinds): ObjectLayer | null {
  const all = scene.objects;
  if (!all) return null;
  const objects = { trees: kinds.trees ? all.trees : [], buildings: kinds.buildings ? all.buildings : [], water: kinds.water ? all.water : [] };
  if (!objects.trees.length && !objects.buildings.length && !objects.water.length) return null;
  const frame = objectFrame(scene, surface);
  const group = new THREE.Group();
  group.name = 'objects';
  const disposables: Array<{ dispose(): void }> = [];

  // trees
  const models = objects.trees.length ? createTreeModels() : [];
  const treeMat = new THREE.MeshStandardMaterial({ vertexColors: true, flatShading: true, roughness: 0.95, metalness: 0 });
  const trees = treeMeshes(objects.trees, frame, models, treeMat);
  disposables.push(...models, treeMat, ...trees);
  if (trees.length) {
    const g = new THREE.Group();
    g.name = 'trees';
    trees.forEach((m) => g.add(m));
    group.add(g);
  }

  // buildings
  const roofMat = new THREE.MeshStandardMaterial({ color: '#b9b4ac', roughness: 0.9, metalness: 0 });
  const facade = facadeTexture();
  const wallMat = new THREE.MeshStandardMaterial({ color: '#e2ddd3', map: facade, roughness: 0.85, metalness: 0 });
  disposables.push(roofMat, wallMat);
  if (facade) disposables.push(facade);
  const meshes: THREE.Mesh[] = [];
  if (objects.buildings.length) {
    const { roof, walls } = buildingGeometry(objects.buildings, frame);
    const roofMesh = new THREE.Mesh(toGeometry(roof), roofMat);
    roofMesh.name = 'building-roofs';
    const wallMesh = new THREE.Mesh(toGeometry(walls), wallMat);
    wallMesh.name = 'building-walls';
    meshes.push(roofMesh, wallMesh);
  }

  // water
  const waterMat = new THREE.MeshStandardMaterial({ color: '#2f6fa8', roughness: 0.12, metalness: 0.2 });
  disposables.push(waterMat);
  if (objects.water.length) {
    const waterMesh = new THREE.Mesh(toGeometry(waterGeometry(objects.water, frame)), waterMat);
    waterMesh.name = 'water';
    meshes.push(waterMesh);
  }
  for (const m of meshes) {
    group.add(m);
    disposables.push(m.geometry);
  }

  return {
    group,
    setRoofTexture(t) {
      roofMat.map = t;
      roofMat.color.set(t ? '#ffffff' : '#b9b4ac');
      roofMat.needsUpdate = true;
    },
    setShadows(on) {
      group.traverse((o) => {
        if ((o as THREE.Mesh).isMesh) {
          o.castShadow = on && o.name !== 'water';
          o.receiveShadow = on;
        }
      });
    },
    dispose() {
      disposables.forEach((d) => d.dispose());
    },
  };
}
