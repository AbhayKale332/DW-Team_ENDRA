import { create } from 'zustand';
import type { CameraBookmark } from '@/domain/types';

export type CameraMode = 'orbit' | 'walk' | 'flight' | 'tour';

export const CAMERA_MODE_LABELS: Record<CameraMode, string> = {
  orbit: 'Orbit',
  walk: 'First person',
  flight: 'Flight simulator',
  tour: 'Drone tour',
};

interface CameraState {
  mode: CameraMode;
  /** Camera heading in degrees clockwise from north (drives the compass). */
  heading: number;
  isHome: boolean;
  /** Metres per CSS pixel in the 2D views (orthographic), for the scale bar. */
  mpp: number;
  fov: number;
  moveSpeed: number;
  tourSpeed: number;
  tourPlaying: boolean;
  bookmarks: CameraBookmark[];
  /** Live flight telemetry for the HUD. */
  flight: { altitude: number; agl: number; speed: number; pitch: number; roll: number };
  /** Camera ground position (scene x east, z south) for the mini-map. */
  camXZ: [number, number];
  set: (p: Partial<Omit<CameraState, 'set'>>) => void;
}

export const useCamera = create<CameraState>()((set) => ({
  mode: 'orbit',
  heading: 0,
  isHome: true,
  mpp: 1,
  fov: 50,
  moveSpeed: 1,
  tourSpeed: 1,
  tourPlaying: true,
  bookmarks: [],
  flight: { altitude: 0, agl: 0, speed: 0, pitch: 0, roll: 0 },
  camXZ: [0, 0],
  set: (p) => set(p),
}));

/** Imperative bridge between DOM controls (zoom buttons, compass, menus) and the active camera rig. */
export interface ViewportApi {
  zoom: (dir: 1 | -1) => void;
  reset: () => void;
  faceNorth: () => void;
  preset: (p: 'top' | 'north' | 'east' | 'south' | 'west' | 'fit') => void;
  getPose: () => { position: [number, number, number]; target: [number, number, number] } | null;
  setPose: (pose: { position: [number, number, number]; target: [number, number, number] }) => void;
  screenshot: () => Promise<Blob | null>;
  orbitKey: (dx: number, dy: number) => void;
}

const noop: ViewportApi = {
  zoom: () => {},
  reset: () => {},
  faceNorth: () => {},
  preset: () => {},
  getPose: () => null,
  setPose: () => {},
  screenshot: async () => null,
  orbitKey: () => {},
};

export const viewportApi: ViewportApi = { ...noop };

export function registerViewportApi(api: Partial<ViewportApi>) {
  Object.assign(viewportApi, api);
  return () => {
    for (const k of Object.keys(api) as Array<keyof ViewportApi>) {
      (viewportApi as unknown as Record<string, unknown>)[k] = noop[k];
    }
  };
}
