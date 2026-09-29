import { useScene } from '@/store/scene';
import { canGeolocate } from '@/lib/osm';
import { useCamera } from '@/store/camera';
import { ProcessingOverlay } from '@/features/processing/ProcessingOverlay';
import { ScenarioOverlay } from '@/features/usecases/UseCasesTab';
import { ProductChip } from '@/features/anchoring/ProductChip';
import { ToolPalette } from '@/features/analysis/ToolPalette';
import { ViewSwitcher } from './ViewSwitcher';
import { NavControls } from './NavControls';
import { Compass } from './Compass';
import { ClassLegend, Colorbar, ScaleBar } from './Legend';
import { HoverCard } from './HoverCard';
import { OsmLegend } from '@/features/osm/OsmLegend';
import { SurroundingsLegend } from '@/features/basemap/SurroundingsLegend';
import { EmptyState } from './EmptyState';
import { SceneWarnings } from './SceneWarnings';
import { KeyHints, ModeBar, NavigationHud } from './NavigationHud';
import classes from './overlays.module.css';

/** Everything floating over the canvas. Layout follows the product mockups. */
export function ViewportOverlays() {
  const hasScene = useScene((s) => !!s.scene);
  // A known CRS, not just a pixel transform: only then is "north" real (same test as the OSM overlay).
  const georeferenced = useScene((s) => canGeolocate(s.scene?.georef));
  const building = useScene((s) => s.meshBuilding && s.run.status !== 'running');
  const camMode = useCamera((s) => s.mode);
  const orbit = camMode === 'orbit';
  return (
    <div className={`${classes.layer} dw-no-print`}>
      <EmptyState />
      {hasScene && <NavigationHud />}
      <div className={classes.topLeft}>
        <ViewSwitcher />
        {hasScene && orbit && <ToolPalette />}
        {hasScene && <ProductChip />}
      </div>
      <div className={classes.topCenter}>
        {!orbit && <ModeBar />}
        <SceneWarnings />
      </div>
      <div className={classes.topRight}>
        <NavControls />
      </div>
      <ScenarioOverlay />
      <ProcessingOverlay />
      {hasScene && (
        <>
          {/* north is only meaningful when the image is georeferenced */}
          {georeferenced && (
            <div className={classes.bottomLeft}>
              <Compass />
            </div>
          )}
          {orbit && (
            <div className={classes.bottomRight}>
              <ScaleBar />
              <Colorbar />
              <ClassLegend />
              <OsmLegend />
              <SurroundingsLegend />
            </div>
          )}
          {orbit && <HoverCard />}
          {!orbit && (
            <div className={classes.bottomCenter}>
              <KeyHints />
            </div>
          )}
        </>
      )}
      {building && (
        <div className={classes.bottomCenter}>
          <div className="dw-float" style={{ padding: '5.5px 11px', fontSize: 11.5 }} role="status">
            Building 3D mesh…
          </div>
        </div>
      )}
    </div>
  );
}
