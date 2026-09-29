import { Badge, Checkbox, Divider, Group, RangeSlider, SegmentedControl, Select, Slider, Stack, Switch, Text, type SelectProps } from '@mantine/core';
import { useScene } from '@/store/scene';
import { LAYER_LABELS, OBJECT_KIND_LABELS, OBJECT_KINDS, toggleObjectKind, useView, type DrapeLayer, type RangeMode } from '@/store/view';
import { useSettings, type Quality } from '@/store/settings';
import { COLORMAPS, cssGradient, type ColormapId } from '@/theme/colormaps';
import { useTerrainInfo } from '@/features/viewport/terrainState';
import { setViewMode } from '@/features/viewport/overlays/ViewSwitcher';
import { canGeolocate } from '@/lib/osm';
import { BASEMAPS, type BasemapId } from '@/lib/tiles';
import { POI_CATEGORIES, POI_CATEGORY_LABELS } from '@/lib/poi';

function Field({ label, value, children }: { label: string; value?: string; children: React.ReactNode }) {
  return (
    <div>
      <Group justify="space-between" mb={8}>
        <Text size="sm" fw={500}>
          {label}
        </Text>
        {value && (
          <Text size="xs" c="dimmed" className="dw-mono">
            {value}
          </Text>
        )}
      </Group>
      {children}
    </div>
  );
}

/** A titled group of options; sections sit further apart than the options inside them. */
function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <Stack gap="lg" component="section" aria-label={title}>
      <Divider label={<span className="dw-section-title">{title}</span>} labelPosition="left" />
      {children}
    </Stack>
  );
}

/** Label on top of its input, hint underneath it; spaced like Field. */
const SELECT_PROPS: Pick<SelectProps, 'size' | 'inputWrapperOrder' | 'styles'> = {
  size: 'sm',
  inputWrapperOrder: ['label', 'input', 'description', 'error'],
  styles: { label: { marginBottom: 8, fontWeight: 500 }, description: { marginTop: 6, lineHeight: 1.5 } },
};

const LAYERS_3D: DrapeLayer[] = ['tint', 'optical', 'height', 'hillshade', 'slope', 'classes', 'reference', 'error'];
const LAYERS_2D: DrapeLayer[] = ['height', 'hillshade', 'slope', 'classes', 'reference', 'error'];

/** Map tiles and OpenStreetMap facilities around a georeferenced scene. */
function SurroundingsSection() {
  const geo = useScene((s) => canGeolocate(s.scene?.georef));
  const basemap = useView((s) => s.basemap);
  const poi = useView((s) => s.poi);
  const categories = useView((s) => s.poiCategories);
  const provider = useSettings((s) => s.basemapProvider);
  const set = useView((s) => s.set);
  return (
    <Section title="Surroundings">
      {!geo && (
        <Text size="xs" c="dimmed" lh={1.5}>
          Needs a georeferenced image (e.g. a GeoTIFF) with a known coordinate system.
        </Text>
      )}
      <Field label="Basemap">
        <Stack gap="sm">
          <Switch label="Show the area around the scene" checked={geo && basemap} disabled={!geo} onChange={(e) => set({ basemap: e.currentTarget.checked })} />
          <SegmentedControl
            fullWidth
            size="sm"
            value={provider}
            disabled={!geo || !basemap}
            onChange={(p) => useSettings.getState().set({ basemapProvider: p as BasemapId })}
            data={(Object.keys(BASEMAPS) as BasemapId[]).map((id) => ({ value: id, label: BASEMAPS[id].label }))}
            aria-label="Basemap"
          />
          <Text size="xs" c="dimmed" lh={1.5}>
            Map tiles two scene widths beyond every edge, for context only; they are not processed.
          </Text>
        </Stack>
      </Field>
      <Field label="Facilities">
        <Stack gap="sm">
          <Switch label="Show facilities from OpenStreetMap" checked={geo && poi} disabled={!geo} onChange={(e) => set({ poi: e.currentTarget.checked })} />
          {POI_CATEGORIES.map((c) => (
            <Checkbox
              key={c}
              label={POI_CATEGORY_LABELS[c]}
              checked={categories[c]}
              disabled={!geo || !poi}
              onChange={(e) => set({ poiCategories: { ...categories, [c]: e.currentTarget.checked } })}
            />
          ))}
        </Stack>
      </Field>
    </Section>
  );
}

/** Visual layer stack: drape, colormap, range, relief, lighting and mesh options. */
export function LayersTab() {
  const scene = useScene((s) => s.scene);
  const hasRef = useScene((s) => !!s.reference);
  const hasClasses = useScene((s) => !!s.scene?.classes);
  const v = useView();
  const quality = useSettings((s) => s.quality);
  const postFx = useSettings((s) => s.postFx);
  const info = useTerrainInfo((s) => s.info);
  const is3d = v.mode === 'dsm3d';
  const layer = is3d ? v.layer3d : v.mode === 'heightmap' ? v.layer2d : 'optical';
  const layerOptions = (is3d ? LAYERS_3D : LAYERS_2D).map((l) => ({
    value: l,
    label: LAYER_LABELS[l],
    disabled: ((l === 'reference' || l === 'error') && !hasRef) || (l === 'classes' && !hasClasses),
  }));
  const layerHints = [
    !hasRef && 'Load a reference in the Validation tab to enable reference and error layers.',
    scene && !hasClasses && 'Object classes need a result from a model that publishes a class map (seg.png).',
  ].filter(Boolean);
  const st = scene?.stats;

  return (
    <Stack gap="xl" px="md" py="lg">
      <Field label="View">
        <SegmentedControl
          fullWidth
          size="sm"
          value={v.mode}
          onChange={(m) => setViewMode(m as never)}
          data={[
            { value: 'dsm3d', label: '3D DSM' },
            { value: 'heightmap', label: 'Height map' },
            { value: 'image', label: 'Image' },
          ]}
          aria-label="View"
        />
      </Field>

      {v.mode !== 'image' && (
        <Section title="Colour">
          <Select
            {...SELECT_PROPS}
            label="Drape layer"
            description={layerHints.length ? layerHints.join(' ') : undefined}
            data={layerOptions}
            value={layer}
            onChange={(l) => l && v.set(is3d ? { layer3d: l as DrapeLayer } : { layer2d: l as DrapeLayer })}
          />
          <div>
            <Select
              {...SELECT_PROPS}
              label="Colormap"
              data={COLORMAPS.map((c) => ({ value: c.id, label: c.cvdSafe ? `${c.label} (colour-blind safe)` : c.label }))}
              value={v.colormap}
              onChange={(c) => c && v.set({ colormap: c as ColormapId })}
            />
            <div style={{ height: 10, marginTop: 8, borderRadius: 2, background: cssGradient(v.colormap, 'to right'), border: '1px solid var(--dw-line)' }} aria-hidden />
          </div>
          <Field label="Colour range">
            <SegmentedControl
              fullWidth
              size="sm"
              value={v.rangeMode}
              onChange={(m) => {
                if (m === 'custom' && st) v.set({ rangeMode: 'custom', customRange: [Number(st.p2.toFixed(1)), Number(st.p98.toFixed(1))] });
                else v.set({ rangeMode: m as RangeMode });
              }}
              data={[
                { value: 'robust', label: '2–98 %' },
                { value: 'full', label: 'Full' },
                { value: 'custom', label: 'Custom' },
              ]}
              aria-label="Colour range"
            />
            {v.rangeMode === 'custom' && st && (
              <RangeSlider
                mt="lg"
                mb="xs"
                min={Math.floor(st.min)}
                max={Math.ceil(st.max)}
                step={0.1}
                minRange={0.2}
                value={v.customRange}
                onChange={(r) => v.set({ customRange: r })}
                label={(x) => `${x.toFixed(1)} m`}
                aria-label="Custom colour range"
              />
            )}
          </Field>
          {layer === 'tint' && (
            <Field label="Height tint" value={`${Math.round(v.tintOpacity * 100)} %`}>
              <Slider min={0} max={1} step={0.05} value={v.tintOpacity} onChange={(x) => v.set({ tintOpacity: x })} label={null} aria-label="Height tint opacity" />
            </Field>
          )}
          {layer === 'slope' && (
            <Field label="Slope colour limit" value={`${v.slopeMax}°`}>
              <Slider min={10} max={90} step={5} value={v.slopeMax} onChange={(x) => v.set({ slopeMax: x })} label={null} aria-label="Slope colour limit" />
            </Field>
          )}
          <Field label="Relief shading" value={`${Math.round(v.hillshadeStrength * 100)} %`}>
            <Slider min={0} max={1} step={0.05} value={v.hillshadeStrength} onChange={(x) => v.set({ hillshadeStrength: x })} label={null} aria-label="Relief shading" />
          </Field>
          {/* <Group align="flex-end" gap="sm" wrap="nowrap">
            <Switch label="Contours" checked={v.contours} onChange={(e) => v.set({ contours: e.currentTarget.checked })} style={{ flex: 1 }} />
            <NumberInput
              w={110}
              aria-label="Contour interval"
              disabled={!v.contours}
              min={0.5}
              max={100}
              step={0.5}
              value={v.contourInterval}
              onChange={(x) => typeof x === 'number' && x > 0 && v.set({ contourInterval: x })}
              suffix=" m"
            />
          </Group> */}
        </Section>
      )}

      {is3d && (
        <Section title="Terrain">
          <div>
            <Text size="sm" fw={500} mb={10}>
              3D objects
            </Text>
            {scene?.objects ? (
              <Stack gap="sm" role="group" aria-label="3D objects">
                {OBJECT_KINDS.map((k) => {
                  const n = scene.objects?.[k].length ?? 0;
                  return (
                    <Switch
                      key={k}
                      label={`${OBJECT_KIND_LABELS[k]} (${n})`}
                      checked={n > 0 && v.objectKinds[k]}
                      disabled={n === 0}
                      onChange={() => toggleObjectKind(k)}
                    />
                  );
                })}
                <Text size="xs" c="dimmed" lh={1.5}>
                  Each class on is drawn as models over ground flattened under it; off shows the model&apos;s raw heights there.
                </Text>
              </Stack>
            ) : (
              scene && (
                <Text size="xs" c="dimmed" lh={1.5}>
                  3D objects need a result from a model that publishes objects.json.
                </Text>
              )
            )}
          </div>
          <Field label="Vertical exaggeration" value={`${v.exaggeration.toFixed(1)}×`}>
            <Slider
              min={0.2}
              max={8}
              step={0.1}
              value={v.exaggeration}
              onChange={(x) => v.set({ exaggeration: x })}
              marks={[{ value: 1, label: '1×' }, { value: 4, label: '4×' }, { value: 8, label: '8×' }]}
              label={null}
              aria-label="Vertical exaggeration"
              mb="lg"
            />
          </Field>
          {/* <Group align="flex-end" gap="sm" wrap="nowrap">
            <Switch label="Vertical walls on structures" checked={v.walls} onChange={(e) => v.set({ walls: e.currentTarget.checked })} style={{ flex: 1 }} />
            <NumberInput
              w={110}
              aria-label="Wall threshold"
              disabled={!v.walls}
              min={0.5}
              max={50}
              step={0.5}
              value={v.wallThreshold}
              onChange={(x) => typeof x === 'number' && x > 0 && v.set({ wallThreshold: x })}
              suffix=" m"
            />
          </Group> */}
          <Field label="Mesh detail" value={info ? `${(info.vertices / 1000).toFixed(0)}k vertices` : undefined}>
            <SegmentedControl
              fullWidth
              size="sm"
              value={quality}
              onChange={(q) => useSettings.getState().set({ quality: q as Quality })}
              data={[
                { value: 'fast', label: 'Fast' },
                { value: 'balanced', label: 'Balanced' },
                { value: 'full', label: 'Full' },
              ]}
              aria-label="Mesh detail"
            />
          </Field>
          <div>
            <Text size="sm" fw={500} mb={10}>
              Rendering
            </Text>
            <Stack gap="sm" role="group" aria-label="Rendering">
              <Switch label="Wireframe" checked={v.wireframe} onChange={(e) => v.set({ wireframe: e.currentTarget.checked })} />
              <Switch label="Shadows" checked={v.shadows} onChange={(e) => v.set({ shadows: e.currentTarget.checked })} />
              <Switch label="Ambient occlusion" checked={postFx} onChange={(e) => useSettings.getState().set({ postFx: e.currentTarget.checked })} />
            </Stack>
          </div>
        </Section>
      )}

      {v.mode !== 'image' && (
        <Section title="Sun">
          <Field label="Azimuth" value={`${v.sunAzimuth}°`}>
            <Slider min={0} max={360} step={1} value={v.sunAzimuth} onChange={(x) => v.set({ sunAzimuth: x })} label={null} aria-label="Sun azimuth" />
          </Field>
          <Field label="Elevation" value={`${v.sunElevation}°`}>
            <Slider min={5} max={88} step={1} value={v.sunElevation} onChange={(x) => v.set({ sunElevation: x })} label={null} aria-label="Sun elevation" />
          </Field>
        </Section>
      )}
      {scene && <SurroundingsSection />}
      {st && (
        <Text size="xs" c="dimmed" lh={1.5}>
          <Badge size="xs" variant="outline" mr={6}>
            tip
          </Badge>
          Heights span {st.min.toFixed(1)}–{st.max.toFixed(1)} m; the 2–98 % range ignores outliers.
        </Text>
      )}
    </Stack>
  );
}
