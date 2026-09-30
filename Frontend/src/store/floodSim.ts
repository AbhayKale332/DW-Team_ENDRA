import * as THREE from 'three';
import { create } from 'zustand';

/** What the running flood simulation reports to the UI, a few times a second. */
export interface FloodLive {
  share: number;
  areaM2: number;
  meanDepthM: number;
  maxDepthM: number;
  /** Water surface at each building (index into scene.objects.buildings), absolute metres; NaN = dry. */
  stages: Float32Array;
  /** Simulated seconds since the water was last at rest. */
  timeS: number;
}

interface FloodSimState {
  live: FloodLive | null;
  set: (p: Partial<Omit<FloodSimState, 'set'>>) => void;
}

export const useFloodSim = create<FloodSimState>()((set) => ({ live: null, set: (p) => set(p) }));

/** The simulation state on the GPU, shared by every material that draws water (the terrain drape, the water
 *  surface, the flooded buildings). One uniform object each, so updating a value reaches all of them. */
export const simUniforms = {
  /** RGBA32F per simulation cell (FloodSim.writeFrame); null when no simulation runs. */
  uSim: { value: null as THREE.Texture | null },
  uSimSize: { value: new THREE.Vector2(1, 1) },
  /** World (x, z) → simulation cell coordinates (cell centres at integers): xz * xy + zw. */
  uSimWorld: { value: new THREE.Vector4(1, 1, 0, 0) },
  /** Scene uv → simulation cell coordinates. */
  uSimUv: { value: new THREE.Vector4(1, 1, 0, 0) },
  uHasSim: { value: 0 },
  /** The terrain drape leaves the water's colour to the 3D water surface. */
  uWaterMesh: { value: 0 },
  uExag: { value: 1 },
};

/** Bilinear read of the simulation texture at continuous cell coordinates (RGBA32F is not filterable everywhere). */
export const SIM_GLSL = /* glsl */ `
  vec4 simTexel(ivec2 c) {
    return texelFetch(uSim, clamp(c, ivec2(0), ivec2(uSimSize) - 1), 0);
  }
  vec4 simAt(vec2 c) {
    vec2 f = floor(c);
    vec2 t = c - f;
    ivec2 i = ivec2(f);
    vec4 a = mix(simTexel(i), simTexel(i + ivec2(1, 0)), t.x);
    vec4 b = mix(simTexel(i + ivec2(0, 1)), simTexel(i + ivec2(1, 1)), t.x);
    return mix(a, b, t.y);
  }
`;

export const SIM_UNIFORMS_GLSL = /* glsl */ `
  uniform sampler2D uSim;
  uniform vec2 uSimSize;
  uniform vec4 uSimWorld;
  uniform vec4 uSimUv;
  uniform float uHasSim;
  uniform float uWaterMesh;
  uniform float uExag;
`;
