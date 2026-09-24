/* DepthWizard v5 viewer — dual-pane DSM inspector + 3D flythrough.
 *
 * v5 adds: the absolute surfaces (dsm_m.npy / dtm_m.npy — the scored product was
 * never rendered in v4, so hilly scenes flew flat), a Surface / Structures /
 * Terrain toggle, lon/lat on the probe (the transform now reaches meta.json),
 * NoData (NaN) handling, and three shadow layers: shadows cast by the heights
 * (live, driven by the sun sliders), shadows found in the image
 * (shadow_img.png), and their agreement.  The sun sliders start at the product
 * metadata's sun (meta.sun) when there is one.
 *
 * Loads a prediction directory written by `infer/predict.py`:
 *
 *     ndsm_m.npy   float32 (H, W), metres above ground   <- preferred, exact
 *     ndsm16.png   16-bit fallback + the affine in meta.json
 *     rgb.png      the optical image that produced it
 *     meta.json    the contract, the scene metadata, the diagnostics
 *     gt_ndsm_m.npy (optional) reference surface -> live metrics + error layer
 *
 * The .npy path exists because a 16-bit PNG drawn into a 2-D canvas is
 * down-converted to 8 bits by the browser, which would quantise a 0-60 m range
 * into 0.23 m steps before anything is rendered.  Metres are the product here,
 * so the viewer reads the float array directly and only falls back to the PNG.
 *
 * No build step, no bundler, no network: three.js r128 is vendored beside this
 * file, so `index.html` opened from a USB stick works exactly like the served
 * copy.  That is the "standalone deployment" half of the rubric.
 */
'use strict';

// ---------------------------------------------------------------------------
// small utilities
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const fmt = (v, n = 2) => (v === null || v === undefined || !isFinite(v) ? '—' : v.toFixed(n));

/** Minimal NumPy .npy v1/v2 reader — enough for a 2-D float or int array. */
function parseNpy(buffer) {
  const b = new Uint8Array(buffer);
  const magic = String.fromCharCode(...b.slice(1, 6));
  if (magic !== 'NUMPY') throw new Error('not a .npy file');
  const major = b[6];
  const hlenBytes = major === 1 ? 2 : 4;
  const dv = new DataView(buffer);
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const start = 8 + hlenBytes;
  const header = new TextDecoder().decode(b.slice(start, start + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)[1] === 'True';
  const shape = /'shape':\s*\(([^)]*)\)/.exec(header)[1]
    .split(',').map((s) => s.trim()).filter(Boolean).map(Number);
  if (fortran) throw new Error('fortran-order .npy is not supported');
  const off = start + hlen;
  const n = shape.reduce((a, c) => a * c, 1);
  const kind = descr.replace(/^[<>|=]/, '');
  let data;
  if (kind === 'f4') data = new Float32Array(buffer, off, n);
  else if (kind === 'f8') data = Float32Array.from(new Float64Array(buffer, off, n));
  else if (kind === 'u1') data = Float32Array.from(new Uint8Array(buffer, off, n));
  else if (kind === 'i2') data = Float32Array.from(new Int16Array(buffer, off, n));
  else if (kind === 'i4') data = Float32Array.from(new Int32Array(buffer, off, n));
  else throw new Error('unsupported dtype ' + descr);
  return { data, shape };
}

/* Turbo colormap, 9-stop piecewise linear. Perceptually ordered and, unlike
 * jet, monotone in luminance — so a printed report still reads correctly. */
const TURBO = [
  [48, 18, 59], [70, 107, 227], [40, 187, 226], [61, 231, 154],
  [163, 244, 78], [231, 215, 53], [253, 152, 39], [232, 79, 13], [122, 4, 3],
];
function turbo(t) {
  t = clamp(t, 0, 1) * (TURBO.length - 1);
  const i = Math.min(TURBO.length - 2, Math.floor(t));
  const f = t - i, a = TURBO[i], b = TURBO[i + 1];
  return [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f, a[2] + (b[2] - a[2]) * f];
}
/* Diverging blue-white-red, for signed error. */
function diverging(t) {
  const x = clamp(t, -1, 1);
  return x < 0
    ? [255 * (1 + x * 0.85), 255 * (1 + x * 0.55), 255]
    : [255, 255 * (1 - x * 0.55), 255 * (1 - x * 0.85)];
}

// ---------------------------------------------------------------------------
// the loaded scene
// ---------------------------------------------------------------------------
const S = {
  W: 0, H: 0, gsd: 0.5,
  height: null,           // Float32Array, metres above S.base (NaN -> 0) — geometry
  raw: null,              // the active surface as loaded (absolute, may hold NaN)
  base: 0,                // subtracted for display so a 400 m plateau sits at y = 0
  surfaces: {},           // { ndsm, dsm, dtm } Float32Arrays, whichever exist
  surface: 'ndsm',
  shadowImg: null,        // Uint8Array mask from shadow_img.png, or null
  gt: null,               // Float32Array or null (a reference nDSM)
  rgb: null,              // HTMLCanvasElement
  meta: null,
  hMin: 0, hMax: 1,
  probeA: null, probeB: null,
};

function decodeHeightFromPng(img, meta) {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const g = c.getContext('2d', { willReadFrequently: true });
  g.drawImage(img, 0, 0);
  const px = g.getImageData(0, 0, c.width, c.height).data;
  const lo = meta.height_min_m ?? 0, hi = meta.height_max_m ?? 1;
  const out = new Float32Array(c.width * c.height);
  for (let i = 0; i < out.length; i++) out[i] = lo + (px[i * 4] / 255) * (hi - lo);
  warn('height came from an 8-bit canvas read of ndsm16.png — precision is ~' +
    ((hi - lo) / 255).toFixed(2) + ' m. Load ndsm_m.npy for exact metres.');
  return { data: out, shape: [c.height, c.width] };
}

function statsOf(a) {
  let lo = Infinity, hi = -Infinity, sum = 0, n = 0;
  for (let i = 0; i < a.length; i++) {
    const v = a[i];
    if (!isFinite(v)) continue;
    if (v < lo) lo = v; if (v > hi) hi = v; sum += v; n++;
  }
  return { min: lo, max: hi, mean: n ? sum / n : 0, n };
}

function compareToReference() {
  if (!S.gt) return null;
  let se = 0, ae = 0, sd = 0, n = 0;
  let sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
  const P = S.surfaces.ndsm || S.raw;
  for (let i = 0; i < P.length; i++) {
    const p = P[i], t = S.gt[i];
    if (!isFinite(p) || !isFinite(t)) continue;
    const d = p - t;
    se += d * d; ae += Math.abs(d); sd += d; n++;
    sx += p; sy += t; sxx += p * p; syy += t * t; sxy += p * t;
  }
  if (!n) return null;
  const cov = sxy / n - (sx / n) * (sy / n);
  const vx = sxx / n - (sx / n) ** 2, vy = syy / n - (sy / n) ** 2;
  return {
    rmse: Math.sqrt(se / n), mae: ae / n, bias: sd / n,
    r: cov / (Math.sqrt(vx * vy) + 1e-12), n,
  };
}

// ---------------------------------------------------------------------------
// texture layers  (2-D pane and the 3-D drape share these)
// ---------------------------------------------------------------------------
function layerCanvas(mode) {
  if (mode === 'rgb' && S.rgb) return S.rgb;
  const c = document.createElement('canvas');
  c.width = S.W; c.height = S.H;
  const g = c.getContext('2d');
  const img = g.createImageData(S.W, S.H);
  const d = img.data;
  const span = Math.max(S.hMax - S.hMin, 1e-3);

  if (mode === 'hillshade') {
    const sh = hillshade(S.height, S.W, S.H, S.gsd, +$('sunAz').value, +$('sunEl').value);
    for (let i = 0; i < sh.length; i++) {
      const v = sh[i]; d[i * 4] = d[i * 4 + 1] = d[i * 4 + 2] = v; d[i * 4 + 3] = 255;
    }
  } else if (mode.startsWith('shadow_')) {
    const cast = mode !== 'shadow_img' ? castShadowMask() : null;
    const img = S.shadowImg;
    for (let i = 0; i < S.W * S.H; i++) {
      const c = cast ? cast[i] : 0, m = img ? img[i] : 0;
      let rgb;
      if (mode === 'shadow_cast') rgb = c ? [20, 20, 40] : [225, 225, 215];
      else if (mode === 'shadow_img') rgb = m ? [20, 20, 40] : [225, 225, 215];
      // agreement: white = both, red = image only (height missing / too low),
      // blue = height only (too high or a spike), grey = neither
      else rgb = c && m ? [250, 250, 250] : m ? [220, 50, 40] : c ? [40, 90, 230] : [70, 70, 70];
      d[i * 4] = rgb[0]; d[i * 4 + 1] = rgb[1]; d[i * 4 + 2] = rgb[2]; d[i * 4 + 3] = 255;
    }
  } else if (mode === 'error' && S.gt) {
    const P = S.surfaces.ndsm || S.raw;
    let lim = 0;
    for (let i = 0; i < P.length; i++) {
      const e = Math.abs(P[i] - S.gt[i]);
      if (isFinite(e)) lim = Math.max(lim, e);
    }
    lim = Math.max(lim * 0.6, 1);
    for (let i = 0; i < P.length; i++) {
      const [r, gg, b] = diverging((P[i] - S.gt[i]) / lim);
      d[i * 4] = r; d[i * 4 + 1] = gg; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
    }
  } else {
    const src = mode === 'reference' && S.gt ? S.gt : S.height;
    for (let i = 0; i < src.length; i++) {
      const nd = !isFinite(S.raw[i]);
      const [r, gg, b] = nd ? [0, 0, 0] : turbo((src[i] - S.hMin) / span);
      d[i * 4] = r; d[i * 4 + 1] = gg; d[i * 4 + 2] = b; d[i * 4 + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
  return c;
}

/** Shadows the surface casts for the sun sliders — a port of viz/shadow.py
 * `cast_shadows` (same azimuth convention as hillshade: clockwise from up).
 * Computed at <= 768 px and nearest-upsampled: it is a scene-scale view. */
let _castCache = { key: '', mask: null };
function castShadowMask() {
  const az = +$('sunAz').value, el = +$('sunEl').value;
  const key = `${S.surface}|${az}|${el}|${S.W}x${S.H}`;
  if (_castCache.key === key) return _castCache.mask;
  const s = Math.max(1, Math.ceil(Math.max(S.W, S.H) / 768));
  const w = Math.ceil(S.W / s), h = Math.ceil(S.H / s);
  const hd = new Float32Array(w * h);
  let hmax = -Infinity, hmin = Infinity;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const v = S.raw[(y * s) * S.W + x * s];
      const q = isFinite(v) ? v : -Infinity;
      hd[y * w + x] = q;
      if (isFinite(q)) { hmax = Math.max(hmax, q); hmin = Math.min(hmin, q); }
    }
  }
  const a = (az * Math.PI) / 180, dr = -Math.cos(a), dc = Math.sin(a);
  const rise = S.gsd * s * Math.tan((Math.max(el, 0.5) * Math.PI) / 180);
  const steps = Math.min(512, Math.ceil((hmax - hmin) / Math.max(rise, 1e-6)) + 1);
  const sh = new Uint8Array(w * h);
  for (let t = 1; t <= steps; t++) {
    const oy = Math.round(t * dr), ox = Math.round(t * dc), lift = t * rise;
    if (Math.abs(oy) >= h || Math.abs(ox) >= w) break;
    for (let y = Math.max(0, -oy); y < Math.min(h, h - oy); y++) {
      const row = y * w, orow = (y + oy) * w;
      for (let x = Math.max(0, -ox); x < Math.min(w, w - ox); x++) {
        if (!sh[row + x] && hd[orow + x + ox] > hd[row + x] + lift) sh[row + x] = 1;
      }
    }
  }
  const out = new Uint8Array(S.W * S.H);
  for (let y = 0; y < S.H; y++) {
    const sy = Math.min(h - 1, Math.floor(y / s));
    for (let x = 0; x < S.W; x++) out[y * S.W + x] = sh[sy * w + Math.min(w - 1, Math.floor(x / s))];
  }
  _castCache = { key, mask: out };
  return out;
}

/** Horn hillshade — the honest test of whether a DSM is shaped like terrain. */
function hillshade(h, W, H, gsd, azDeg, elDeg) {
  const out = new Uint8ClampedArray(W * H);
  const az = ((360 - azDeg + 90) * Math.PI) / 180, el = (elDeg * Math.PI) / 180;
  const sinEl = Math.sin(el), cosEl = Math.cos(el);
  for (let y = 0; y < H; y++) {
    const y0 = Math.max(0, y - 1) * W, y1 = Math.min(H - 1, y + 1) * W;
    for (let x = 0; x < W; x++) {
      const x0 = Math.max(0, x - 1), x1 = Math.min(W - 1, x + 1);
      const dx = (h[y * W + x1] - h[y * W + x0]) / (gsd * (x1 - x0 || 1));
      const dy = (h[y1 + x] - h[y0 + x]) / (gsd * 2 || 1);
      const slope = Math.atan(Math.hypot(dx, dy));
      const aspect = Math.atan2(-dx, dy);
      out[y * W + x] = 255 * clamp(
        sinEl * Math.cos(slope) + cosEl * Math.sin(slope) * Math.cos(az - aspect), 0, 1);
    }
  }
  return out;
}

// ---------------------------------------------------------------------------
// three.js scene
// ---------------------------------------------------------------------------
let renderer, scene, camera, orbit, fpControls, mesh, sun, hemi, probeGroup, pathLine;
let mode3d = 'orbit';
let flying = false, flyT = 0, flyCurve = null;
const keys = Object.create(null);
let lastFrame = performance.now(), fps = 0;

function initThree() {
  const host = $('pane3d');
  renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  host.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0e1116);
  camera = new THREE.PerspectiveCamera(58, 1, 0.5, 40000);

  hemi = new THREE.HemisphereLight(0xbfd4ff, 0x2a2a28, 0.65);
  scene.add(hemi);
  sun = new THREE.DirectionalLight(0xfff3e0, 1.15);
  sun.castShadow = true;
  sun.shadow.mapSize.set(2048, 2048);
  scene.add(sun, sun.target);

  orbit = new THREE.OrbitControls(camera, renderer.domElement);
  orbit.enableDamping = true;
  orbit.dampingFactor = 0.08;
  orbit.maxPolarAngle = Math.PI * 0.495;

  fpControls = new THREE.PointerLockControls(camera, renderer.domElement);
  scene.add(fpControls.getObject ? fpControls.getObject() : new THREE.Object3D());
  fpControls.addEventListener('unlock', () => { if (mode3d === 'fp') setMode('orbit'); });

  probeGroup = new THREE.Group();
  scene.add(probeGroup);

  addEventListener('keydown', (e) => {
    keys[e.code] = true;
    if (e.code === 'KeyF' && !e.repeat) setMode(mode3d === 'fp' ? 'orbit' : 'fp');
    if (e.code === 'KeyR' && !e.repeat) resetCamera();
  });
  addEventListener('keyup', (e) => { keys[e.code] = false; });
  renderer.domElement.addEventListener('pointerdown', onPick);
  addEventListener('resize', resize);
  resize();
  animate();
}

function resize() {
  if (!renderer) return;
  const host = $('pane3d');
  const w = host.clientWidth, h = host.clientHeight;
  if (!w || !h) return;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  draw2d();
}

function decimation() {
  const budget = +$('quality').value;
  let step = 1;
  while (Math.ceil(S.H / step) * Math.ceil(S.W / step) > budget) step++;
  return step;
}

function buildMesh() {
  if (mesh) {
    mesh.geometry.dispose();
    mesh.material.map?.dispose();
    mesh.material.dispose();
    scene.remove(mesh);
  }
  const step = decimation();
  const gw = Math.floor((S.W - 1) / step) + 1;
  const gh = Math.floor((S.H - 1) / step) + 1;
  const spacing = S.gsd * step;

  // PlaneGeometry is built in XY then rotated to XZ, so (col, row) maps to
  // (x, z) with x growing east and z growing south — the same convention
  // viz/mesh.py writes, and the same one the UV mapping below assumes.
  const geo = new THREE.PlaneGeometry((gw - 1) * spacing, (gh - 1) * spacing, gw - 1, gh - 1);
  geo.rotateX(-Math.PI / 2);
  const pos = geo.attributes.position;
  for (let r = 0; r < gh; r++) {
    for (let c = 0; c < gw; c++) {
      const sr = Math.min(S.H - 1, r * step), sc = Math.min(S.W - 1, c * step);
      pos.setY(r * gw + c, S.height[sr * S.W + sc]);
    }
  }
  pos.needsUpdate = true;
  geo.computeVertexNormals();

  const tex = new THREE.CanvasTexture(layerCanvas($('layer').value));
  tex.encoding = THREE.sRGBEncoding;
  tex.anisotropy = renderer.capabilities.getMaxAnisotropy();
  const mat = new THREE.MeshStandardMaterial({
    map: tex, roughness: 0.95, metalness: 0.0,
    wireframe: $('wire').checked, side: THREE.DoubleSide,
  });
  mesh = new THREE.Mesh(geo, mat);
  mesh.receiveShadow = true;
  mesh.castShadow = true;
  mesh.position.set((gw - 1) * spacing / 2, 0, (gh - 1) * spacing / 2);
  scene.add(mesh);

  applyExaggeration();
  updateSun();
  buildFlyPath();
  resetCamera();
  $('vertCount').textContent = (gw * gh).toLocaleString() + ' verts (1:' + step + ')';
}

function applyExaggeration() {
  const v = +$('vex').value;
  $('vexLabel').textContent = v.toFixed(1) + '×';
  if (mesh) mesh.scale.y = v;
  redrawProbe();
}

function sceneExtent() {
  return { x: S.W * S.gsd, z: S.H * S.gsd };
}

function resetCamera() {
  const e = sceneExtent();
  const d = Math.max(e.x, e.z);
  orbit.target.set(e.x / 2, 0, e.z / 2);
  camera.position.set(e.x / 2, d * 0.55, e.z / 2 + d * 0.75);
  camera.up.set(0, 1, 0);
  orbit.update();
}

function updateSun() {
  const az = (+$('sunAz').value * Math.PI) / 180;
  const el = (+$('sunEl').value * Math.PI) / 180;
  const e = sceneExtent();
  const d = Math.max(e.x, e.z) * 1.5;
  sun.position.set(
    e.x / 2 + d * Math.cos(el) * Math.sin(az),
    d * Math.sin(el) + 10,
    e.z / 2 + d * Math.cos(el) * Math.cos(az));
  sun.target.position.set(e.x / 2, 0, e.z / 2);
  sun.target.updateMatrixWorld();
  const c = sun.shadow.camera;
  c.left = -d; c.right = d; c.top = d; c.bottom = -d; c.near = 1; c.far = d * 4;
  c.updateProjectionMatrix();
  sun.castShadow = $('shadows').checked;
}

/** A closed cinematic orbit for the demo video, elevated over the scene. */
function buildFlyPath() {
  if (pathLine) { scene.remove(pathLine); pathLine.geometry.dispose(); }
  const e = sceneExtent();
  const cx = e.x / 2, cz = e.z / 2;
  const rad = Math.max(e.x, e.z) * 0.42;
  const pts = [];
  for (let i = 0; i < 24; i++) {
    const a = (i / 24) * Math.PI * 2;
    const wob = 1 + 0.18 * Math.sin(a * 3);
    pts.push(new THREE.Vector3(
      cx + rad * wob * Math.cos(a),
      (S.hMax * 1.6 + 60) * (0.75 + 0.35 * Math.sin(a * 2)),
      cz + rad * wob * Math.sin(a)));
  }
  flyCurve = new THREE.CatmullRomCurve3(pts, true, 'catmullrom', 0.5);
  const geo = new THREE.BufferGeometry().setFromPoints(flyCurve.getPoints(300));
  pathLine = new THREE.Line(geo, new THREE.LineBasicMaterial({
    color: 0x2d6cdf, transparent: true, opacity: 0.35 }));
  pathLine.visible = false;
  scene.add(pathLine);
}

function setMode(m) {
  mode3d = m;
  flying = m === 'fly';
  if (pathLine) pathLine.visible = flying;
  orbit.enabled = m === 'orbit';
  if (m === 'fp') { try { fpControls.lock(); } catch (e) { /* needs a gesture */ } }
  else if (fpControls.isLocked) fpControls.unlock();
  for (const b of document.querySelectorAll('[data-mode]')) {
    b.classList.toggle('on', b.dataset.mode === m);
  }
  $('hint').textContent = {
    orbit: 'drag = orbit · scroll = zoom · click = probe · F = first person · R = reset',
    fp: 'WASD = move · Shift = sprint · Space/C = up/down · mouse = look · Esc = exit',
    fly: 'cinematic drone orbit — press Orbit to take control',
  }[m];
}

function animate() {
  requestAnimationFrame(animate);
  const now = performance.now();
  const dt = Math.min(0.1, (now - lastFrame) / 1000);
  lastFrame = now;
  fps = fps * 0.9 + (1 / Math.max(dt, 1e-4)) * 0.1;

  if (mode3d === 'orbit') orbit.update();
  else if (mode3d === 'fp' && fpControls.isLocked) {
    const base = Math.max(sceneExtent().x, sceneExtent().z) * 0.12;
    const sp = base * (keys.ShiftLeft || keys.ShiftRight ? 3 : 1) * dt;
    if (keys.KeyW) fpControls.moveForward(sp);
    if (keys.KeyS) fpControls.moveForward(-sp);
    if (keys.KeyA) fpControls.moveRight(-sp);
    if (keys.KeyD) fpControls.moveRight(sp);
    if (keys.Space) camera.position.y += sp;
    if (keys.KeyC) camera.position.y -= sp;
  } else if (flying && flyCurve) {
    flyT = (flyT + dt * 0.02) % 1;
    camera.position.copy(flyCurve.getPointAt(flyT));
    const e = sceneExtent();
    camera.lookAt(e.x / 2, S.hMax * 0.3, e.z / 2);
  }

  if (renderer && mesh) renderer.render(scene, camera);
  $('fps').textContent = fps.toFixed(0) + ' fps';
}

// ---------------------------------------------------------------------------
// probing:  height read-out, two-point slope, elevation profile
// ---------------------------------------------------------------------------
const ray = new THREE.Raycaster();
function onPick(ev) {
  if (!mesh || mode3d === 'fp') return;
  const r = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((ev.clientX - r.left) / r.width) * 2 - 1,
    -((ev.clientY - r.top) / r.height) * 2 + 1);
  ray.setFromCamera(ndc, camera);
  const hit = ray.intersectObject(mesh, false)[0];
  if (!hit) return;
  const p = hit.point;
  const col = clamp(Math.round(p.x / S.gsd), 0, S.W - 1);
  const row = clamp(Math.round(p.z / S.gsd), 0, S.H - 1);
  addProbe({ row, col });
}

function sampleAt(row, col) {
  const i = clamp(row, 0, S.H - 1) * S.W + clamp(col, 0, S.W - 1);
  return { h: S.raw[i], y: S.height[i], gt: S.gt ? S.gt[i] : null,
           ndsm: S.surfaces.ndsm ? S.surfaces.ndsm[i] : null };
}

function addProbe(p) {
  if (!S.probeA || (S.probeA && S.probeB)) { S.probeA = p; S.probeB = null; }
  else S.probeB = p;
  redrawProbe();
  reportProbe();
}

function redrawProbe() {
  while (probeGroup.children.length) {
    const c = probeGroup.children.pop();
    c.geometry?.dispose(); c.material?.dispose();
    probeGroup.remove(c);
  }
  const vex = mesh ? mesh.scale.y : 1;
  const marker = (p, color) => {
    const s = Math.max(sceneExtent().x, sceneExtent().z) * 0.006;
    const m = new THREE.Mesh(
      new THREE.SphereGeometry(s, 12, 10),
      new THREE.MeshBasicMaterial({ color }));
    m.position.set(p.col * S.gsd, sampleAt(p.row, p.col).y * vex, p.row * S.gsd);
    probeGroup.add(m);
  };
  if (S.probeA) marker(S.probeA, 0x2d6cdf);
  if (S.probeB) {
    marker(S.probeB, 0xe5793a);
    const a = new THREE.Vector3(S.probeA.col * S.gsd,
      sampleAt(S.probeA.row, S.probeA.col).y * vex, S.probeA.row * S.gsd);
    const b = new THREE.Vector3(S.probeB.col * S.gsd,
      sampleAt(S.probeB.row, S.probeB.col).y * vex, S.probeB.row * S.gsd);
    probeGroup.add(new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([a, b]),
      new THREE.LineBasicMaterial({ color: 0xffffff })));
  }
  draw2d();
}

function lonLatAt(row, col) {
  const t = S.meta?.scene?.transform;      // [a, b, c, d, e, f] affine, if written
  if (!Array.isArray(t) || t.length < 6) return null;
  const x = t[0] * (col + 0.5) + t[1] * (row + 0.5) + t[2];
  const y = t[3] * (col + 0.5) + t[4] * (row + 0.5) + t[5];
  const epsg = +S.meta?.scene?.crs_epsg;
  let ll = null;
  if (epsg === 4326) ll = { lon: x, lat: y };
  else if (epsg > 32600 && epsg < 32661) ll = utmToLonLat(x, y, epsg - 32600, false);
  else if (epsg > 32700 && epsg < 32761) ll = utmToLonLat(x, y, epsg - 32700, true);
  return { x, y, ...(ll || {}) };
}

/** WGS84 UTM -> lon/lat (Snyder's series; sub-metre over a zone).  Every
 * Cartosat / NRSC product is WGS84 UTM, so this covers the judged inputs
 * without a projection library in a no-network viewer. */
function utmToLonLat(E, N, zone, south) {
  const a = 6378137.0, f = 1 / 298.257223563, k0 = 0.9996;
  const e2 = f * (2 - f), ep2 = e2 / (1 - e2);
  const x = E - 500000, y = south ? N - 10000000 : N;
  const M = y / k0;
  const mu = M / (a * (1 - e2 / 4 - 3 * e2 * e2 / 64 - 5 * e2 ** 3 / 256));
  const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
  const p = mu + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu)
    + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu)
    + (151 * e1 ** 3 / 96) * Math.sin(6 * mu) + (1097 * e1 ** 4 / 512) * Math.sin(8 * mu);
  const C1 = ep2 * Math.cos(p) ** 2, T1 = Math.tan(p) ** 2;
  const N1 = a / Math.sqrt(1 - e2 * Math.sin(p) ** 2);
  const R1 = a * (1 - e2) / (1 - e2 * Math.sin(p) ** 2) ** 1.5;
  const D = x / (N1 * k0);
  const lat = p - (N1 * Math.tan(p) / R1) * (D * D / 2
    - (5 + 3 * T1 + 10 * C1 - 4 * C1 * C1 - 9 * ep2) * D ** 4 / 24
    + (61 + 90 * T1 + 298 * C1 + 45 * T1 * T1 - 252 * ep2 - 3 * C1 * C1) * D ** 6 / 720);
  const lon0 = ((zone - 1) * 6 - 180 + 3) * Math.PI / 180;
  const lon = lon0 + (D - (1 + 2 * T1 + C1) * D ** 3 / 6
    + (5 - 2 * C1 + 28 * T1 - 3 * C1 * C1 + 8 * ep2 + 24 * T1 * T1) * D ** 5 / 120) / Math.cos(p);
  return { lon: lon * 180 / Math.PI, lat: lat * 180 / Math.PI };
}

function reportProbe() {
  const out = [];
  if (S.probeA) {
    const a = sampleAt(S.probeA.row, S.probeA.col);
    out.push(`<b>A</b> px (${S.probeA.col}, ${S.probeA.row}) · <b>${fmt(a.h)} m</b> ` +
      `<span class="dim">${surfaceUnit()}</span>` +
      (S.surface !== 'ndsm' && a.ndsm !== null ? ` · above ground ${fmt(a.ndsm)} m` : '') +
      (a.gt !== null ? ` · reference ${fmt(a.gt)} m · err ${fmt((a.ndsm ?? a.h) - a.gt)} m` : ''));
    const ll = lonLatAt(S.probeA.row, S.probeA.col);
    if (ll && ll.lat !== undefined) {
      out.push(`&nbsp;&nbsp;<b>${ll.lat.toFixed(6)}° N, ${ll.lon.toFixed(6)}° E</b>` +
        ` <span class="dim">(map ${ll.x.toFixed(1)}, ${ll.y.toFixed(1)})</span>`);
    } else if (ll) out.push(`&nbsp;&nbsp;map (${ll.x.toFixed(2)}, ${ll.y.toFixed(2)})`);
  }
  if (S.probeA && S.probeB) {
    const a = sampleAt(S.probeA.row, S.probeA.col), b = sampleAt(S.probeB.row, S.probeB.col);
    const dx = (S.probeB.col - S.probeA.col) * S.gsd;
    const dz = (S.probeB.row - S.probeA.row) * S.gsd;
    const run = Math.hypot(dx, dz), rise = b.h - a.h;
    out.push(`<b>B</b> px (${S.probeB.col}, ${S.probeB.row}) · <b>${fmt(b.h)} m</b>`);
    out.push(`<b>ground distance</b> ${fmt(run, 1)} m · <b>Δh</b> ${fmt(rise)} m`);
    out.push(`<b>slope</b> ${fmt((rise / Math.max(run, 1e-6)) * 100, 1)} % ` +
      `(${fmt((Math.atan2(rise, Math.max(run, 1e-6)) * 180) / Math.PI, 1)}°)`);
    drawProfile();
  } else {
    $('profile').getContext('2d').clearRect(0, 0, 999, 999);
  }
  $('probe').innerHTML = out.join('<br>') ||
    'click the terrain to probe a height; click again for slope and a profile';
}

function drawProfile() {
  const cv = $('profile'), g = cv.getContext('2d');
  const w = cv.width = cv.clientWidth * devicePixelRatio;
  const h = cv.height = cv.clientHeight * devicePixelRatio;
  g.clearRect(0, 0, w, h);
  const n = 220, pred = [], ref = [];
  for (let i = 0; i < n; i++) {
    const t = i / (n - 1);
    const r = Math.round(S.probeA.row + (S.probeB.row - S.probeA.row) * t);
    const c = Math.round(S.probeA.col + (S.probeB.col - S.probeA.col) * t);
    const s = sampleAt(r, c);
    pred.push(s.h);
    if (s.gt !== null) ref.push(s.gt);
  }
  const all = ref.length ? pred.concat(ref) : pred;
  const lo = Math.min(...all), hi = Math.max(...all);
  const span = Math.max(hi - lo, 1);
  const pad = 4 * devicePixelRatio;
  const line = (arr, color, width) => {
    g.strokeStyle = color; g.lineWidth = width * devicePixelRatio;
    g.beginPath();
    arr.forEach((v, i) => {
      const x = pad + (i / (n - 1)) * (w - 2 * pad);
      const y = h - pad - ((v - lo) / span) * (h - 2 * pad);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  };
  if (ref.length) line(ref, '#8a949e', 1.2);
  line(pred, '#2d6cdf', 1.6);
  g.fillStyle = '#8a949e';
  g.font = `${10 * devicePixelRatio}px system-ui`;
  g.fillText(`${hi.toFixed(1)} m`, pad, 11 * devicePixelRatio);
  g.fillText(`${lo.toFixed(1)} m`, pad, h - pad);
}

// ---------------------------------------------------------------------------
// 2-D pane
// ---------------------------------------------------------------------------
function draw2d() {
  const cv = $('map'), g = cv.getContext('2d');
  const host = $('pane2d');
  const w = cv.width = Math.max(1, host.clientWidth * devicePixelRatio);
  const h = cv.height = Math.max(1, (host.clientHeight - 34) * devicePixelRatio);
  g.clearRect(0, 0, w, h);
  if (!S.height) return;

  const src = layerCanvas($('layer2d').value);
  const sc = Math.min(w / S.W, h / S.H);
  const dw = S.W * sc, dh = S.H * sc;
  const ox = (w - dw) / 2, oy = (h - dh) / 2;
  g.imageSmoothingEnabled = sc > 1.5;
  g.drawImage(src, ox, oy, dw, dh);

  if ($('contours').checked) drawContours(g, ox, oy, sc);

  // everything below is an overlay on the raster, so clip it to the raster —
  // the camera can sit outside the scene and its indicator should not scribble
  // across the letterbox
  g.save();
  g.beginPath();
  g.rect(ox, oy, dw, dh);
  g.clip();

  // camera footprint, so the two panes are visibly the same place
  if (camera) {
    const px = (camera.position.x / S.gsd) * sc + ox;
    const py = (camera.position.z / S.gsd) * sc + oy;
    const tx = (orbit.target.x / S.gsd) * sc + ox;
    const ty = (orbit.target.z / S.gsd) * sc + oy;
    g.strokeStyle = '#ffd166'; g.lineWidth = 1.4 * devicePixelRatio;
    g.beginPath(); g.moveTo(px, py); g.lineTo(tx, ty); g.stroke();
    g.fillStyle = '#ffd166';
    g.beginPath(); g.arc(px, py, 4 * devicePixelRatio, 0, 7); g.fill();
  }
  const dot = (p, col) => {
    if (!p) return;
    g.fillStyle = col;
    g.beginPath();
    g.arc(p.col * sc + ox, p.row * sc + oy, 4 * devicePixelRatio, 0, 7);
    g.fill();
  };
  if (S.probeA && S.probeB) {
    g.strokeStyle = '#fff'; g.lineWidth = 1.5 * devicePixelRatio;
    g.beginPath();
    g.moveTo(S.probeA.col * sc + ox, S.probeA.row * sc + oy);
    g.lineTo(S.probeB.col * sc + ox, S.probeB.row * sc + oy);
    g.stroke();
  }
  dot(S.probeA, '#2d6cdf');
  dot(S.probeB, '#e5793a');
  g.restore();
  cv._map = { ox, oy, sc };
}

function drawContours(g, ox, oy, sc) {
  const step = Math.max(1, Math.round((S.hMax - S.hMin) / 12));
  g.strokeStyle = 'rgba(255,255,255,.35)';
  g.lineWidth = 1;
  // marching-squares-lite: draw a pixel wherever a cell straddles a level
  for (let y = 0; y < S.H - 1; y += 1) {
    for (let x = 0; x < S.W - 1; x += 1) {
      const a = S.height[y * S.W + x], b = S.height[y * S.W + x + 1];
      const la = Math.floor(a / step), lb = Math.floor(b / step);
      if (la !== lb) g.fillRect(x * sc + ox, y * sc + oy, Math.max(sc, 1), Math.max(sc, 1));
    }
  }
  g.fillStyle = 'rgba(255,255,255,.35)';
}

$('map')?.addEventListener('pointerdown', (ev) => {
  const cv = $('map'), m = cv._map;
  if (!m || !S.height) return;
  const r = cv.getBoundingClientRect();
  const x = ((ev.clientX - r.left) * devicePixelRatio - m.ox) / m.sc;
  const y = ((ev.clientY - r.top) * devicePixelRatio - m.oy) / m.sc;
  if (x < 0 || y < 0 || x >= S.W || y >= S.H) return;
  addProbe({ row: Math.round(y), col: Math.round(x) });
});

// ---------------------------------------------------------------------------
// loading
// ---------------------------------------------------------------------------
function warn(msg) {
  const el = $('warn');
  el.style.display = 'block';
  el.innerHTML += (el.innerHTML ? '<br>' : '') + msg;
}

async function loadFromFiles(fileList) {
  const files = {};
  for (const f of fileList) files[f.name.toLowerCase()] = f;
  const read = (name, how) => {
    const f = files[name];
    if (!f) return null;
    return how === 'text' ? f.text() : f.arrayBuffer();
  };

  const metaTxt = await read('meta.json', 'text');
  const meta = metaTxt ? JSON.parse(metaTxt) : {};

  let hgt = null;
  const npy = await read('ndsm_m.npy', 'buf') || await read('pred_ndsm_m.npy', 'buf');
  if (npy) hgt = parseNpy(npy);
  else if (files['ndsm16.png'] || files['height16.png']) {
    const img = await fileToImage(files['ndsm16.png'] || files['height16.png']);
    hgt = decodeHeightFromPng(img, meta);
  }
  if (!hgt) { warn('no height array found (need ndsm_m.npy or ndsm16.png)'); return; }

  let gt = null;
  const gtBuf = await read('gt_ndsm_m.npy', 'buf');
  if (gtBuf) gt = parseNpy(gtBuf).data;
  const extraSurf = {};
  for (const k of ['dsm', 'dtm']) {
    const b = await read(`${k}_m.npy`, 'buf');
    if (b) extraSurf[k] = parseNpy(b).data;
  }

  let rgb = null;
  const rgbFile = files['rgb.png'] || files['rgb.jpg'] || files['texture.png'];
  if (rgbFile) rgb = imageToCanvas(await fileToImage(rgbFile));
  let shadow = null;
  if (files['shadow_img.png']) shadow = imageToCanvas(await fileToImage(files['shadow_img.png']));

  adopt({ hgt, gt, rgb, meta, extraSurf, shadow });
}

function fileToImage(file) {
  return new Promise((res, rej) => {
    const img = new Image();
    img.onload = () => res(img);
    img.onerror = rej;
    img.src = URL.createObjectURL(file);
  });
}

function imageToCanvas(img) {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  c.getContext('2d').drawImage(img, 0, 0);
  return c;
}

async function loadFromUrl(base) {
  base = base.replace(/\/+$/, '') + '/';
  const meta = await fetch(base + 'meta.json').then((r) => r.json());
  const hgt = parseNpy(await fetch(base + 'ndsm_m.npy').then((r) => r.arrayBuffer()));
  let gt = null;
  try {
    gt = parseNpy(await fetch(base + 'gt_ndsm_m.npy').then((r) => {
      if (!r.ok) throw new Error('no reference');
      return r.arrayBuffer();
    })).data;
  } catch (e) { /* optional */ }
  let rgb = null;
  try {
    const img = new Image();
    img.src = base + 'rgb.png';
    await img.decode();
    rgb = imageToCanvas(img);
  } catch (e) { /* optional */ }
  const extraSurf = {};
  for (const k of ['dsm', 'dtm']) {
    try {
      const r = await fetch(base + `${k}_m.npy`);
      if (r.ok) extraSurf[k] = parseNpy(await r.arrayBuffer()).data;
    } catch (e) { /* optional */ }
  }
  let shadow = null;
  try {
    const img = new Image();
    img.src = base + 'shadow_img.png';
    await img.decode();
    shadow = imageToCanvas(img);
  } catch (e) { /* optional */ }
  adopt({ hgt, gt, rgb, meta, extraSurf, shadow });
}

function surfaceUnit() {
  if (S.surface === 'ndsm') return 'above ground';
  const d = S.meta?.vertical_datum;
  return d ? `(${d})` : '(absolute)';
}

/** Point the geometry at one of the loaded surfaces. */
function useSurface(name) {
  if (!S.surfaces[name]) name = 'ndsm';
  S.surface = name;
  S.raw = S.surfaces[name];
  const st = statsOf(S.raw);
  S.base = name === 'ndsm' ? 0 : (isFinite(st.min) ? st.min : 0);
  const h = new Float32Array(S.raw.length);
  for (let i = 0; i < h.length; i++) h[i] = isFinite(S.raw[i]) ? S.raw[i] - S.base : 0;
  S.height = h;
  S.hMin = 0;
  S.hMax = Math.max((isFinite(st.max) ? st.max : 1) - S.base, 1);
  _castCache.key = '';
  return st;
}

function adopt({ hgt, gt, rgb, meta, extraSurf = {}, shadow = null }) {
  $('warn').style.display = 'none';
  $('warn').innerHTML = '';
  S.H = hgt.shape[0]; S.W = hgt.shape[1];
  S.surfaces = { ndsm: hgt.data };
  for (const [k, v] of Object.entries(extraSurf)) {
    if (v && v.length === hgt.data.length) S.surfaces[k] = v;
  }
  const sel = $('surface');
  if (sel) {
    for (const o of sel.options) o.disabled = !S.surfaces[o.value];
    if (!S.surfaces[sel.value]) sel.value = 'ndsm';
  }
  useSurface(sel ? sel.value : 'ndsm');
  S.shadowImg = null;
  if (shadow) {
    const c = document.createElement('canvas');
    c.width = S.W; c.height = S.H;
    const g2 = c.getContext('2d', { willReadFrequently: true });
    g2.imageSmoothingEnabled = false;
    g2.drawImage(shadow, 0, 0, S.W, S.H);
    const px = g2.getImageData(0, 0, S.W, S.H).data;
    S.shadowImg = new Uint8Array(S.W * S.H);
    for (let i = 0; i < S.shadowImg.length; i++) S.shadowImg[i] = px[i * 4] > 127 ? 1 : 0;
  }
  for (const sl of ['layer', 'layer2d']) {
    for (const v of ['shadow_img', 'shadow_agree']) {
      const o = $(sl).querySelector(`[value=${v}]`);
      if (o) o.disabled = !S.shadowImg;
    }
  }
  const sun = meta?.sun;
  if (sun && isFinite(sun.azimuth_deg) && isFinite(sun.elevation_deg)) {
    $('sunAz').value = Math.round(sun.azimuth_deg);
    $('sunEl').value = clamp(Math.round(sun.elevation_deg), 5, 88);
  }
  S.gt = gt && gt.length === hgt.data.length ? gt : null;
  if (gt && !S.gt) warn('reference array shape differs from the prediction — ignored');
  S.meta = meta || {};
  S.gsd = +(meta.scene?.gsd_m ?? meta.gsd_m ?? 0.5);
  if (rgb && (rgb.width !== S.W || rgb.height !== S.H)) {
    const c = document.createElement('canvas');
    c.width = S.W; c.height = S.H;
    c.getContext('2d').drawImage(rgb, 0, 0, S.W, S.H);
    rgb = c;
  }
  S.rgb = rgb;
  const st = statsOf(S.surfaces.ndsm);
  S.probeA = S.probeB = null;

  $('layer').querySelector('[value=rgb]').disabled = !rgb;
  for (const sel of ['layer', 'layer2d']) {
    const o = $(sel).querySelector('[value=error]');
    const o2 = $(sel).querySelector('[value=reference]');
    o.disabled = o2.disabled = !S.gt;
  }
  if (!rgb && $('layer').value === 'rgb') $('layer').value = 'height';

  let nb = 0, nv = 0;
  for (const v of S.surfaces.ndsm) { if (isFinite(v)) { nv++; if (v < 1) nb++; } }
  const frac = nb / Math.max(nv, 1);
  $('info').innerHTML =
    `<b>${S.W} × ${S.H}</b> px @ <b>${S.gsd.toFixed(3)} m</b>/px ` +
    `(${(S.W * S.gsd).toFixed(0)} × ${(S.H * S.gsd).toFixed(0)} m)<br>` +
    `height ${fmt(st.min)} … ${fmt(st.max)} m · mean ${fmt(st.mean)} m<br>` +
    `${(frac * 100).toFixed(1)} % of pixels below 1 m` +
    (meta.product ? `<br>product <b>${meta.product}</b>` : '') +
    (meta.scene?.georeferenced ? ' · georeferenced' : ' · no spatial metadata') +
    (S.surfaces.dsm ? `<br>absolute DSM ${fmt(meta.dsm_min_m, 1)} … ${fmt(meta.dsm_max_m, 1)} m` +
      ` <span class="dim">${meta.vertical_datum || ''}</span>` : '') +
    (sun ? `<br>sun ${fmt(sun.azimuth_deg, 0)}° / ${fmt(sun.elevation_deg, 0)}° ` +
      `<span class="dim">(${sun.source})</span>` +
      (isFinite(sun.shadow_iou) ? ` · shadow IoU ${fmt(sun.shadow_iou, 2)}` : '') +
      (sun.scale_check ? ` · shadow scale ×${fmt(sun.scale_check.best_scale, 2)}` : '') : '');
  if (frac < 0.15) {
    warn(`only ${(frac * 100).toFixed(1)} % of this scene is below 1 m. A nadir ` +
      `urban scene is normally 40-70 %. The model may be reading texture as ` +
      `terrain on this imagery — check the declared GSD.`);
  }

  const cmp = compareToReference();
  $('metrics').innerHTML = cmp
    ? `<table><tr><td>RMSE</td><td><b>${fmt(cmp.rmse)} m</b></td></tr>` +
      `<tr><td>MAE</td><td>${fmt(cmp.mae)} m</td></tr>` +
      `<tr><td>bias</td><td>${fmt(cmp.bias)} m</td></tr>` +
      `<tr><td>Pearson r</td><td>${fmt(cmp.r, 3)}</td></tr>` +
      `<tr><td>pixels</td><td>${cmp.n.toLocaleString()}</td></tr></table>`
    : '<span class="dim">load <code>gt_ndsm_m.npy</code> alongside to validate ' +
      'against a reference surface</span>';

  buildMesh();
  setMode('orbit');
  reportProbe();
  draw2d();
}

function refreshTexture() {
  if (!mesh) return;
  mesh.material.map?.dispose();
  const t = new THREE.CanvasTexture(layerCanvas($('layer').value));
  t.encoding = THREE.sRGBEncoding;
  t.anisotropy = renderer.capabilities.getMaxAnisotropy();
  mesh.material.map = t;
  mesh.material.wireframe = $('wire').checked;
  mesh.material.needsUpdate = true;
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
function boot() {
  initThree();
  $('files').addEventListener('change', (e) => loadFromFiles(e.target.files));
  $('vex').addEventListener('input', applyExaggeration);
  $('quality').addEventListener('change', () => S.height && buildMesh());
  $('layer').addEventListener('change', refreshTexture);
  $('surface')?.addEventListener('change', () => {
    if (!S.raw) return;
    useSurface($('surface').value);
    buildMesh(); reportProbe(); draw2d();
  });
  $('layer2d').addEventListener('change', draw2d);
  $('contours').addEventListener('change', draw2d);
  $('wire').addEventListener('change', refreshTexture);
  for (const id of ['sunAz', 'sunEl']) {
    $(id).addEventListener('input', () => {
      updateSun();
      const live = (v) => v === 'hillshade' || v === 'shadow_cast' || v === 'shadow_agree';
      if (live($('layer').value)) refreshTexture();
      if (live($('layer2d').value)) draw2d();
    });
  }
  $('shadows').addEventListener('change', updateSun);
  for (const b of document.querySelectorAll('[data-mode]')) {
    b.addEventListener('click', () => setMode(b.dataset.mode));
  }
  $('reset').addEventListener('click', resetCamera);
  $('clearProbe').addEventListener('click', () => {
    S.probeA = S.probeB = null; redrawProbe(); reportProbe();
  });
  $('split').addEventListener('click', () => {
    document.body.classList.toggle('only3d');
    setTimeout(resize, 60);
  });

  const q = new URLSearchParams(location.search);
  if (q.get('result')) loadFromUrl(q.get('result')).catch((e) => warn('load failed: ' + e));
  setMode('orbit');
}

document.addEventListener('DOMContentLoaded', boot);
