import { useEffect, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import { registerViewportApi } from '@/store/camera';
import { useTerrainInfo } from '../terrainState';

/** Publishes a smoothed frames-per-second value for the status bar (2 Hz). */
export function FpsReporter() {
  const frames = useRef(0);
  const last = useRef(performance.now());
  useFrame(() => {
    frames.current++;
    const now = performance.now();
    if (now - last.current >= 500) {
      useTerrainInfo.getState().set({ fps: Math.round((frames.current * 1000) / (now - last.current)) });
      frames.current = 0;
      last.current = now;
    }
  });
  return null;
}

/** Renders one frame on demand and returns the canvas as PNG. */
export function ScreenshotProvider() {
  const { gl, scene, camera } = useThree();
  useEffect(
    () =>
      registerViewportApi({
        screenshot: () =>
          new Promise<Blob | null>((resolve) => {
            gl.render(scene, camera);
            gl.domElement.toBlob((b) => resolve(b), 'image/png');
          }),
      }),
    [gl, scene, camera],
  );
  return null;
}
