import { Suspense, useMemo, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrthographicCamera, PerspectiveCamera, PerformanceMonitor } from '@react-three/drei';
import { EffectComposer, N8AO, SMAA } from '@react-three/postprocessing';
import { ErrorBoundary } from 'react-error-boundary';
import { Box } from '@mantine/core';
import { useScene } from '@/store/scene';
import { useView, VIEW_LABELS } from '@/store/view';
import { useCamera } from '@/store/camera';
import { useSettings } from '@/store/settings';
import { Terrain } from './scene/Terrain';
import { Objects } from './scene/Objects';
import { OsmOverlay } from '@/features/osm/OsmOverlay';
import { BasemapLayer } from '@/features/basemap/BasemapLayer';
import { PoiLayer } from '@/features/poi/PoiLayer';
import { Environment } from './scene/Environment';
import { Markers } from './scene/Markers';
import { TowerMarkers } from './scene/TowerMarkers';
import { FloodBuildings } from './scene/FloodBuildings';
import { FloodWater } from './scene/FloodWater';
import { PickLayer } from './scene/PickLayer';
import { FpsReporter, ScreenshotProvider } from './scene/Helpers';
import { OrbitRig } from './cameras/OrbitRig';
import { WalkRig } from './cameras/WalkRig';
import { FlightRig } from './cameras/FlightRig';
import { TourRig } from './cameras/TourRig';
import { sceneExtent } from './terrainState';
import { ViewportOverlays } from './overlays/ViewportOverlays';
import { WebGLFallback } from './overlays/WebGLFallback';
import classes from './Viewport.module.css';

function Rig() {
  const mode = useView((s) => s.mode);
  const camMode = useCamera((s) => s.mode);
  if (mode !== 'dsm3d') return <OrbitRig flat />;
  if (camMode === 'walk') return <WalkRig />;
  if (camMode === 'flight') return <FlightRig />;
  if (camMode === 'tour') return <TourRig />;
  return <OrbitRig flat={false} />;
}

function Cameras() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const fov = useCamera((s) => s.fov);
  const e = scene ? sceneExtent(scene) : { x: 200, z: 200, h: 20 };
  const span = Math.max(e.x, e.z, 50);
  if (mode === 'dsm3d') return <PerspectiveCamera makeDefault fov={fov} near={Math.max(0.1, span / 5000)} far={span * 60 + 5000} position={[0, span, span]} />;
  return <OrthographicCamera makeDefault near={0.1} far={span * 20 + 1000} position={[0, span * 2 + 100, 0]} />;
}

function Effects() {
  const postFx = useSettings((s) => s.postFx);
  const mode = useView((s) => s.mode);
  const [degraded, setDegraded] = useState(false);
  const enabled = postFx && mode === 'dsm3d' && !degraded;
  return (
    <PerformanceMonitor onDecline={() => setDegraded(true)} flipflops={2} onFallback={() => setDegraded(true)}>
      {enabled && (
        <EffectComposer multisampling={0}>
          <N8AO halfRes aoRadius={4} intensity={1.6} distanceFalloff={1} quality="performance" />
          <SMAA />
        </EffectComposer>
      )}
    </PerformanceMonitor>
  );
}

/** The graded centrepiece: one WebGL canvas serving the 3D DSM view and the two 2D map views. */
export function Viewport() {
  const scene = useScene((s) => s.scene);
  const mode = useView((s) => s.mode);
  const camMode = useCamera((s) => s.mode);
  const [glError, setGlError] = useState<string | null>(null);
  const label = useMemo(() => {
    if (!scene) return 'Empty viewport. Open an image or a sample scene to begin.';
    const s = scene.stats;
    return `${VIEW_LABELS[mode]} of ${scene.name}: ${scene.heights.width} by ${scene.heights.height} pixels at ${scene.gsd.toFixed(2)} metres per pixel, heights from ${s.min.toFixed(1)} to ${s.max.toFixed(1)} metres.`;
  }, [scene, mode]);

  const animating = mode === 'dsm3d';

  return (
    <Box className={classes.viewport} id="dw-viewport" data-view={mode} data-camera={camMode}>
      {glError ? (
        <WebGLFallback message={glError} />
      ) : (
        <ErrorBoundary FallbackComponent={({ error }) => <WebGLFallback message={String((error as Error)?.message ?? error)} />}>
          <Canvas
            id="dw-canvas"
            className={classes.canvas}
            shadows
            dpr={[1, 2]}
            frameloop={animating ? 'always' : 'demand'}
            gl={{ antialias: true, preserveDrawingBuffer: false, powerPreference: 'high-performance' }}
            onCreated={({ gl }) => {
              gl.domElement.addEventListener('webglcontextlost', (e) => {
                e.preventDefault();
                setGlError('The graphics context was lost (GPU reset or memory pressure).');
              });
            }}
            aria-label={label}
            role="img"
            tabIndex={0}
          >
            <Suspense fallback={null}>
              <Cameras />
              <Environment />
              <BasemapLayer />
              <Terrain />
              <Objects />
              <OsmOverlay />
              <PoiLayer />
              <Markers />
              <TowerMarkers />
              <FloodBuildings />
              <FloodWater />
              <Rig />
              <PickLayer />
              <FpsReporter />
              <ScreenshotProvider />
              <Effects />
            </Suspense>
          </Canvas>
        </ErrorBoundary>
      )}
      <p className="dw-sr-only" aria-live="polite">
        {label}
      </p>
      <ViewportOverlays />
    </Box>
  );
}
