import { useEffect, useMemo, useState } from 'react';
import * as THREE from 'three';
import { useThree } from '@react-three/fiber';
import { Html } from '@react-three/drei';
import { useScene } from '@/store/scene';
import { activeObjectKinds, objectKindsKey, useView } from '@/store/view';
import { useCamera } from '@/store/camera';
import { sampleBilinear } from '@/lib/heights';
import { canGeolocate } from '@/lib/osm';
import { POI_KIND_META, POI_KINDS, visiblePois, type Poi } from '@/lib/poi';
import { objectSurfaceIfReady } from '@/features/viewport/objectSurfaceClient';
import { FLAT_SCALE, groundLevel, sceneBase, sceneExtent } from '@/features/viewport/terrainState';
import { ATLAS_COLS, poiAtlas } from './poiIcons';
import { loadPois, poisFor, usePoi } from './poiStore';

/** On-screen pin size (CSS px) and hover pick radius. */
const PIN_PX = 26;
const HOVER_PX = 16;
/** Most pins drawn at once (emergency services are kept first). */
const MAX_PINS = 400;

const vertexShader = /* glsl */ `
  attribute float aIcon;
  uniform float uSize;
  varying float vIcon;
  void main() {
    vIcon = aIcon;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = uSize;
  }
`;

const fragmentShader = /* glsl */ `
  uniform sampler2D uAtlas;
  uniform float uCols;
  varying float vIcon;
  void main() {
    vec2 cell = vec2(mod(vIcon, uCols), floor(vIcon / uCols));
    vec4 c = texture2D(uAtlas, (cell + gl_PointCoord) / uCols);
    if (c.a < 0.04) discard;
    gl_FragColor = c;
    #include <colorspace_fragment>
  }
`;

const KIND_INDEX = new Map(POI_KINDS.map((k, i) => [k, i]));

/** Hover label: name, kind and the few tags that matter in an emergency. */
function PoiLabel({ poi }: { poi: Poi }) {
  const meta = POI_KIND_META[poi.kind];
  const t = poi.tags;
  const extra = [
    t.emergency === 'yes' && 'Emergency department',
    t.beds && `${t.beds} beds`,
    t.operator,
    t['addr:street'] && [t['addr:housenumber'], t['addr:street']].filter(Boolean).join(' '),
  ].filter(Boolean) as string[];
  return (
    <div className="dw-float" style={{ padding: '6px 10px', fontSize: 12, lineHeight: 1.4, whiteSpace: 'nowrap', pointerEvents: 'none', transform: `translate(-50%, calc(-100% - ${PIN_PX / 2 + 6}px))` }}>
      <div style={{ fontWeight: 600 }}>{poi.name ?? meta.label}</div>
      <div style={{ color: meta.color, fontWeight: 500 }}>{meta.label}</div>
      {extra.map((x) => (
        <div key={x} style={{ opacity: 0.75 }}>
          {x}
        </div>
      ))}
    </div>
  );
}

/** OpenStreetMap facilities (hospitals, fire stations, schools, shelters, stations …) as map pins over the scene
 *  and its surroundings. Georeferenced scenes only; fetched when the layer is on. */
export function PoiLayer() {
  const scene = useScene((s) => s.scene);
  const on = useView((s) => s.poi);
  const categories = useView((s) => s.poiCategories);
  const mode = useView((s) => s.mode);
  const exaggeration = useView((s) => s.exaggeration);
  const meshBuilding = useScene((s) => s.meshBuilding);
  const objectsKey = useView((s) => objectKindsKey(activeObjectKinds(s.objectKinds, scene?.objects)));
  const pois = usePoi((s) => poisFor(s, scene));
  const gl = useThree((s) => s.gl);
  const camera = useThree((s) => s.camera);
  const dpr = useThree((s) => s.viewport.dpr);
  const invalidate = useThree((s) => s.invalidate);
  const [atlas, setAtlas] = useState<THREE.Texture | null>(null);
  const [hovered, setHovered] = useState<number | null>(null);
  const geo = on && canGeolocate(scene?.georef);

  useEffect(() => {
    if (geo && scene) void loadPois(scene);
  }, [geo, scene]);

  useEffect(() => {
    if (!geo || atlas) return;
    let live = true;
    poiAtlas().then(
      (t) => live && setAtlas(t),
      () => {}, // pins stay hidden; the legend still lists the facilities
    );
    return () => {
      live = false;
    };
  }, [geo, atlas]);

  // pin anchors (world space, exaggeration applied) and the ground points their stems start from
  const layout = useMemo(() => {
    if (!scene || !pois?.length || meshBuilding) return null;
    const list = visiblePois(pois, categories, MAX_PINS);
    if (!list.length) return null;
    const { width: W, height: H } = scene.heights;
    const gsd = scene.gsd;
    const surface = objectSurfaceIfReady(scene, useView.getState().objectKinds) ?? scene.heights.data;
    const grid = { data: surface, width: W, height: H };
    const base = sceneBase(scene);
    const ground = groundLevel(scene);
    const is3d = mode === 'dsm3d';
    const sy = is3d ? exaggeration : FLAT_SCALE;
    const e = sceneExtent(scene);
    const stem = is3d ? Math.max(e.x, e.z) / 45 : 0;
    const pins = new Float32Array(list.length * 3);
    const icons = new Float32Array(list.length);
    const stems = new Float32Array(list.length * 6);
    list.forEach((p, i) => {
      const inside = p.col >= -0.5 && p.col <= W - 0.5 && p.row >= -0.5 && p.row <= H - 0.5;
      const h = inside ? sampleBilinear(grid, p.col, p.row) - base : ground;
      const x = (p.col - (W - 1) / 2) * gsd;
      const y = (Number.isFinite(h) ? h : ground) * sy;
      const z = (p.row - (H - 1) / 2) * gsd;
      pins.set([x, y + stem, z], i * 3);
      stems.set([x, y, z, x, y + stem, z], i * 6);
      icons[i] = KIND_INDEX.get(p.kind)!;
    });
    return { list, pins, icons, stems: stem > 0 ? stems : null };
    // objectsKey: the surface read above depends on it
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, pois, categories, mode, exaggeration, meshBuilding, objectsKey]);

  const pointGeometry = useMemo(() => {
    if (!layout) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(layout.pins, 3));
    g.setAttribute('aIcon', new THREE.BufferAttribute(layout.icons, 1));
    g.computeBoundingSphere();
    return g;
  }, [layout]);
  const stemGeometry = useMemo(() => {
    if (!layout?.stems) return null;
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(layout.stems, 3));
    return g;
  }, [layout]);
  useEffect(() => () => pointGeometry?.dispose(), [pointGeometry]);
  useEffect(() => () => stemGeometry?.dispose(), [stemGeometry]);

  const pointMaterial = useMemo(
    () =>
      new THREE.ShaderMaterial({
        vertexShader,
        fragmentShader,
        uniforms: { uAtlas: { value: null }, uCols: { value: ATLAS_COLS }, uSize: { value: PIN_PX } },
        transparent: true,
        depthTest: false,
        depthWrite: false,
        toneMapped: false,
      }),
    [],
  );
  const stemMaterial = useMemo(() => new THREE.LineBasicMaterial({ color: '#ffffff', transparent: true, opacity: 0.8, depthTest: false, toneMapped: false }), []);
  useEffect(
    () => () => {
      pointMaterial.dispose();
      stemMaterial.dispose();
    },
    [pointMaterial, stemMaterial],
  );
  useEffect(() => {
    pointMaterial.uniforms.uAtlas.value = atlas;
    pointMaterial.uniforms.uSize.value = PIN_PX * dpr;
    invalidate();
  }, [pointMaterial, atlas, dpr, invalidate]);

  // hover: nearest pin on screen within HOVER_PX (pins are drawn in screen space, so pick them there too)
  useEffect(() => {
    setHovered(null);
    if (!layout) return;
    const el = gl.domElement;
    const v = new THREE.Vector3();
    let last = 0;
    const onMove = (e: PointerEvent) => {
      const now = performance.now();
      if (now - last < 45 || e.buttons || useCamera.getState().mode !== 'orbit') return;
      last = now;
      const r = el.getBoundingClientRect();
      const mx = e.clientX - r.left;
      const my = e.clientY - r.top;
      let best: number | null = null;
      let bestD = HOVER_PX * HOVER_PX;
      for (let i = 0; i < layout.list.length; i++) {
        v.fromArray(layout.pins, i * 3).project(camera);
        if (v.z > 1) continue; // behind the camera
        const d = ((v.x + 1) / 2 * r.width - mx) ** 2 + ((1 - v.y) / 2 * r.height - my) ** 2;
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
      setHovered(best);
    };
    const onLeave = () => setHovered(null);
    el.addEventListener('pointermove', onMove);
    el.addEventListener('pointerleave', onLeave);
    return () => {
      el.removeEventListener('pointermove', onMove);
      el.removeEventListener('pointerleave', onLeave);
    };
  }, [layout, gl, camera]);

  if (!geo || !layout || !pointGeometry || !atlas) return null;
  const h = hovered !== null && hovered < layout.list.length ? hovered : null;
  return (
    <group name="poi-layer">
      {stemGeometry && <lineSegments geometry={stemGeometry} material={stemMaterial} renderOrder={11} />}
      <points geometry={pointGeometry} material={pointMaterial} renderOrder={12} frustumCulled={false} />
      {h !== null && (
        <Html position={[layout.pins[h * 3], layout.pins[h * 3 + 1], layout.pins[h * 3 + 2]]} zIndexRange={[20, 0]}>
          <PoiLabel poi={layout.list[h]} />
        </Html>
      )}
    </group>
  );
}
