import { useEffect } from 'react';
import * as THREE from 'three';
import { useScene } from '@/store/scene';
import { useUseCases } from '@/store/usecases';
import { analysisGridFor } from '@/features/usecases/actions';
import type { CoverageResult } from '@/lib/usecases/telecom';
import type { FloodModel } from '@/lib/usecases/flood';
import type { TerrainUniforms } from './terrainMaterial';

function coverageTexture(c: CoverageResult): THREE.DataTexture {
  const data = new Uint8Array(c.w * c.h * 4);
  for (let i = 0; i < c.cls.length; i++) {
    const k = c.cls[i];
    if (k === 255) continue; // no data: alpha 0
    data[i * 4] = k * 85;
    data[i * 4 + 1] = c.nlos[i];
    data[i * 4 + 3] = 255;
  }
  const t = new THREE.DataTexture(data, c.w, c.h, THREE.RGBAFormat, THREE.UnsignedByteType);
  t.magFilter = THREE.NearestFilter;
  t.minFilter = THREE.NearestFilter;
  t.flipY = false;
  t.needsUpdate = true;
  return t;
}

/** Arrival and ground as metres above the source's normal level, so half floats keep centimetres even at altitude.
 *  Cells that never flood are capped just above the slider's range so linear filtering stays smooth. */
function floodTexture(m: FloodModel): THREE.DataTexture {
  const half = new Uint16Array(m.w * m.h * 4);
  const cap = m.maxRise + 2;
  for (let i = 0; i < m.arrival.length; i++) {
    const a = m.arrival[i];
    const g = m.ground[i];
    half[i * 4] = THREE.DataUtils.toHalfFloat(Number.isFinite(a) ? Math.min(a - m.baseLevel, cap) : cap * 4);
    half[i * 4 + 1] = THREE.DataUtils.toHalfFloat(Number.isFinite(g) ? g - m.baseLevel : 0);
    half[i * 4 + 3] = THREE.DataUtils.toHalfFloat(1);
  }
  const t = new THREE.DataTexture(half, m.w, m.h, THREE.RGBAFormat, THREE.HalfFloatType);
  t.magFilter = THREE.LinearFilter;
  t.minFilter = THREE.LinearFilter;
  t.flipY = false;
  t.needsUpdate = true;
  return t;
}

/** Feeds the telecom coverage / flood results to the terrain shader as an overlay (see applyOverlay). */
export function useAnalysisOverlay(uniforms: TerrainUniforms, invalidate: () => void) {
  const scene = useScene((s) => s.scene);
  const open = useUseCases((s) => s.open);
  const active = useUseCases((s) => s.active);
  const coverage = useUseCases((s) => s.coverage);
  const telecomOverlay = useUseCases((s) => s.telecomOverlay);
  const flood = useUseCases((s) => s.flood);
  const floodOverlay = useUseCases((s) => s.floodOverlay);

  const showCoverage = open && active === 'telecom' && telecomOverlay && !!coverage;
  const showFlood = open && active === 'flood' && floodOverlay && !!flood;

  useEffect(() => {
    if (!scene || (!showCoverage && !showFlood)) {
      uniforms.uOvl.value = 0;
      uniforms.uOvlTex.value = null;
      invalidate();
      return;
    }
    const src = showCoverage ? coverage! : flood!;
    const { grid } = analysisGridFor(scene);
    // the analysis grid may overhang the scene by part of a cell
    uniforms.uOvlScale.value.set(grid.sceneW / (src.w * grid.f), grid.sceneH / (src.h * grid.f));
    const tex = showCoverage ? coverageTexture(coverage!) : floodTexture(flood!);
    uniforms.uOvlTex.value = tex;
    uniforms.uOvl.value = showCoverage ? 1 : 2;
    if (showFlood) uniforms.uMaxRise.value = flood!.maxRise;
    invalidate();
    return () => {
      tex.dispose();
      uniforms.uOvlTex.value = null;
      uniforms.uOvl.value = 0;
    };
  }, [scene, showCoverage, showFlood, coverage, flood, uniforms, invalidate]);

  // the slider moves the water without rebuilding any texture
  useEffect(() => {
    const apply = () => {
      const s = useUseCases.getState();
      uniforms.uWater.value = s.rise;
      uniforms.uVuln.value = s.vulnerability ? 1 : 0;
      invalidate();
    };
    apply();
    return useUseCases.subscribe(apply);
  }, [uniforms, invalidate]);
}
