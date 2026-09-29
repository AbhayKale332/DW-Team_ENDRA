import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { useScene } from '@/store/scene';
import { activeLayer, activeObjectKinds, objectKindsKey, useView } from '@/store/view';
import { useSettings, type Quality } from '@/store/settings';
import { terrainWorker } from '@/workers/clients';
import type { TerrainGeometryData } from '@/workers/terrainMesh';
import { markRunDone } from '@/features/processing/runPrediction';
import { LUT_ROWS, LAYER_INDEX, createClassPalette, createClassTexture, createHeightTexture, createLutAtlas, createTerrainMaterial, createUniforms, sunDirection } from './terrainMaterial';
import { classPaletteRGBA } from '@/theme/classes';
import { loadObjectSurface } from '../objectSurfaceClient';
import { loadImageTexture } from './imageTexture';
import { FLAT_SCALE, displayRange, sceneBase, useTerrainInfo } from '../terrainState';
import { useBasemapShown } from '@/features/basemap/basemapState';
import { computeStats } from '@/lib/heights';
import { useAnalysisOverlay } from './useAnalysisOverlay';

// Regular-grid vertex budgets (wall cells add vertices on top in built-up / forested scenes).
const BUDGET: Record<Quality, number> = { fast: 150_000, balanced: 450_000, full: 1_800_000 };

function toGeometry(d: TerrainGeometryData) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.BufferAttribute(d.positions, 3));
  g.setAttribute('normal', new THREE.BufferAttribute(d.normals, 3));
  g.setAttribute('uv', new THREE.BufferAttribute(d.uvs, 2));
  g.setAttribute('shade', new THREE.BufferAttribute(d.shade, 1));
  g.setIndex(new THREE.BufferAttribute(d.index, 1));
  g.computeBoundingBox();
  g.computeBoundingSphere();
  return g;
}

export function Terrain() {
  const scene = useScene((s) => s.scene);
  const reference = useScene((s) => s.reference);
  const quality = useSettings((s) => s.quality);
  const walls = useView((s) => s.walls);
  const wallThreshold = useView((s) => s.wallThreshold);
  // which object classes the mesh is flattened under (a string, so the rebuild runs only when it changes)
  const objectsKey = useView((s) => objectKindsKey(activeObjectKinds(s.objectKinds, scene?.objects)));
  const mode = useView((s) => s.mode);
  const wireframe = useView((s) => s.wireframe);
  const shadows = useView((s) => s.shadows);
  // with the surroundings on, the ground continues past the edge: no diorama skirt
  const basemap = useBasemapShown();
  const gl = useThree((s) => s.gl);
  const size = useThree((s) => s.size);
  const invalidate = useThree((s) => s.invalidate);

  const uniforms = useMemo(() => createUniforms(), []);
  const lit = useMemo(() => createTerrainMaterial(uniforms, 'lit'), [uniforms]);
  const flat = useMemo(() => createTerrainMaterial(uniforms, 'flat'), [uniforms]);
  const skirtMat = useMemo(() => new THREE.MeshStandardMaterial({ color: '#8d949c', roughness: 1, metalness: 0 }), []);
  const [geo, setGeo] = useState<{ terrain: THREE.BufferGeometry; skirt: THREE.BufferGeometry } | null>(null);
  const group = useRef<THREE.Group>(null);

  // Use-case overlays (telecom coverage, flood water) drawn by the same shader.
  useAnalysisOverlay(uniforms, invalidate);

  // LUT atlas — once.
  useEffect(() => {
    const t = createLutAtlas();
    uniforms.uLut.value = t;
    return () => t.dispose();
  }, [uniforms]);

  // Height + optical textures per scene.
  useEffect(() => {
    if (!scene) return;
    const base = sceneBase(scene);
    const hTex = createHeightTexture(scene.heights.data, scene.heights.width, scene.heights.height, base);
    uniforms.uHeight.value = hTex;
    uniforms.uBase.value = base;
    uniforms.uGsd.value = scene.gsd;
    uniforms.uTexel.value.set(1 / scene.heights.width, 1 / scene.heights.height);
    let rgb: THREE.Texture | null = null;
    let cancelled = false;
    loadImageTexture(scene.image, Math.min(gl.capabilities.maxTextureSize, 8192), gl.capabilities.getMaxAnisotropy()).then((t) => {
      if (cancelled) return t.dispose();
      rgb = t;
      uniforms.uRgb.value = t;
      invalidate();
    });
    invalidate();
    return () => {
      cancelled = true;
      hTex.dispose();
      rgb?.dispose();
      uniforms.uRgb.value = null;
    };
  }, [scene, uniforms, gl, invalidate]);

  // Height above ground for the tint layer of an absolute DSM.
  useEffect(() => {
    const nd = scene?.product === 'DSM' ? scene.ndsm : null;
    if (!scene || !nd) {
      uniforms.uHasAgl.value = 0;
      invalidate();
      return;
    }
    const t = createHeightTexture(nd.data, nd.width, nd.height, 0);
    uniforms.uAgl.value = t;
    uniforms.uAglMax.value = Math.max(computeStats(nd.data).p98, 3);
    uniforms.uHasAgl.value = 1;
    invalidate();
    return () => {
      t.dispose();
      uniforms.uAgl.value = null;
      uniforms.uHasAgl.value = 0;
    };
  }, [scene, uniforms, invalidate]);

  // Reference texture (resampled to the prediction grid if needed).
  useEffect(() => {
    if (!scene || !reference) {
      uniforms.uHasRef.value = 0;
      invalidate();
      return;
    }
    // References are aligned to the prediction grid when loaded (see features/validation).
    const { width, height } = scene.heights;
    const t = createHeightTexture(reference.data, width, height, sceneBase(scene));
    uniforms.uRef.value = t;
    uniforms.uHasRef.value = 1;
    invalidate();
    return () => {
      t.dispose();
      uniforms.uRef.value = null;
      uniforms.uHasRef.value = 0;
    };
  }, [scene, reference, uniforms, invalidate]);

  // Object-class texture + palette (only when the backend supplied a class map).
  useEffect(() => {
    const classes = scene?.classes;
    if (!classes) {
      uniforms.uHasClass.value = 0;
      invalidate();
      return;
    }
    const t = createClassTexture(classes.data, classes.width, classes.height);
    const lut = createClassPalette(classPaletteRGBA(classes.names));
    uniforms.uClass.value = t;
    uniforms.uClassLut.value = lut;
    uniforms.uHasClass.value = 1;
    invalidate();
    return () => {
      t.dispose();
      lut.dispose();
      uniforms.uClass.value = null;
      uniforms.uClassLut.value = null;
      uniforms.uHasClass.value = 0;
    };
  }, [scene, uniforms, invalidate]);

  // Geometry via the worker.
  useEffect(() => {
    if (!scene) {
      setGeo(null);
      return;
    }
    let cancelled = false;
    useScene.getState().set({ meshBuilding: true });
    const extentMax = Math.max(scene.heights.width, scene.heights.height) * scene.gsd;
    const relief = scene.stats.max - scene.stats.min;
    // With objects drawn as models, mesh the ground under them instead of their bumps (computed in the worker;
    // the raw heights when no class is on).
    loadObjectSurface(scene, useView.getState().objectKinds)
      .then((heights) =>
        terrainWorker().build({
          heights: heights.slice(),
          width: scene.heights.width,
          height: scene.heights.height,
          gsd: scene.gsd,
          base: sceneBase(scene),
          budget: BUDGET[quality],
          wallThreshold: walls ? wallThreshold : 0,
          skirtDepth: Math.max(relief * 0.08, extentMax * 0.015, 1),
        }),
      )
      .then((out) => {
        if (cancelled) return;
        setGeo((prev) => {
          prev?.terrain.dispose();
          prev?.skirt.dispose();
          return { terrain: toGeometry(out.terrain), skirt: toGeometry(out.skirt) };
        });
        useTerrainInfo.getState().set({ info: { vertices: out.vertices, step: out.step, wallCells: out.wallCells } });
        useScene.getState().set({ meshBuilding: false });
        markRunDone();
        invalidate();
      })
      .catch((e) => {
        console.error('Terrain build failed', e);
        useScene.getState().set({ meshBuilding: false });
      });
    return () => {
      cancelled = true;
    };
  }, [scene, quality, walls, wallThreshold, objectsKey, invalidate]);

  useEffect(
    () => () => {
      lit.dispose();
      flat.dispose();
      skirtMat.dispose();
    },
    [lit, flat, skirtMat],
  );

  // Push view-store changes into uniforms without React re-renders.
  useEffect(() => {
    const apply = () => {
      const v = useView.getState();
      const sc = useScene.getState().scene;
      uniforms.uLayer.value = LAYER_INDEX[activeLayer(v)];
      const row = LUT_ROWS.indexOf(v.colormap);
      uniforms.uCmapRow.value = row < 0 ? 0 : row;
      uniforms.uTint.value = v.tintOpacity;
      uniforms.uHillshade.value = v.mode === 'dsm3d' ? v.hillshadeStrength * 0.6 : v.hillshadeStrength;
      uniforms.uSlopeMax.value = v.slopeMax;
      uniforms.uContour.value = v.contours ? v.contourInterval : 0;
      uniforms.uCompare.value = v.compareSwipe ? 1 : 0;
      uniforms.uSwipe.value = v.swipe;
      sunDirection(v.sunAzimuth, v.sunElevation, uniforms.uSunDir.value);
      if (sc) {
        const base = sceneBase(sc);
        const [lo, hi] = displayRange(sc.stats, v.rangeMode, v.customRange);
        uniforms.uRange.value.set(lo - base, hi - base);
        uniforms.uErrRange.value = Math.max(1, (sc.stats.p98 - sc.stats.p2) * 0.25);
      }
      if (group.current) group.current.scale.y = v.mode === 'dsm3d' ? v.exaggeration : FLAT_SCALE;
      invalidate();
    };
    apply();
    const u1 = useView.subscribe(apply);
    const u2 = useScene.subscribe((s, p) => s.scene !== p.scene && apply());
    return () => {
      u1();
      u2();
    };
  }, [uniforms, invalidate]);

  useEffect(() => {
    uniforms.uViewport.value.set(size.width * gl.getPixelRatio(), size.height * gl.getPixelRatio());
    invalidate();
  }, [size, gl, uniforms, invalidate]);

  useEffect(() => {
    lit.wireframe = wireframe;
    flat.wireframe = wireframe;
    invalidate();
  }, [wireframe, lit, flat, invalidate]);

  if (!scene || !geo) return null;
  const is3d = mode === 'dsm3d';
  return (
    <group ref={group} name="terrain-root" scale={[1, is3d ? useView.getState().exaggeration : FLAT_SCALE, 1]}>
      <mesh name="terrain" geometry={geo.terrain} material={is3d ? lit : flat} castShadow={is3d && shadows} receiveShadow={is3d && shadows} />
      {is3d && !basemap && <mesh name="terrain-skirt" geometry={geo.skirt} material={skirtMat} receiveShadow={shadows} />}
    </group>
  );
}
