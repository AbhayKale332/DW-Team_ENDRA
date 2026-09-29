// Pre-sample the DEM cells for a bundled sample so it can be anchored to an absolute DSM offline.
//
//   node scripts/cache-dem.mjs [public/samples/buildings_large_campus]
//
// Mirrors src/lib/dem (Terrain Tiles z=12, 30 m anchor cells, 3x3 sub-samples per cell). Writes dem_cells.json.
import { readFile, writeFile } from 'node:fs/promises';
import { inflateSync } from 'node:zlib';
import { join } from 'node:path';
import proj4 from 'proj4';

const dir = process.argv[2] ?? 'public/samples/buildings_large_campus';
const Z = 12;
const T = 256;
const meta = JSON.parse(await readFile(join(dir, 'meta.json'), 'utf8'));
const { transform, crs_epsg: epsg, gsd_m: gsd, width: W, height: H } = meta.scene;
if (!transform || !epsg) throw new Error('meta.json has no georeferencing');
const src = epsg >= 32601 && epsg <= 32660 ? `+proj=utm +zone=${epsg - 32600} +datum=WGS84 +units=m +no_defs` : epsg >= 32701 && epsg <= 32760 ? `+proj=utm +zone=${epsg - 32700} +south +datum=WGS84 +units=m +no_defs` : null;
if (!src) throw new Error(`EPSG:${epsg} unsupported by this script`);
const [a, b, c, d, e, f] = transform;
const toLonLat = (col, row) => {
  // col,row are pixel-centre indices; pixelToMap adds 0.5
  const x = a * (col + 0.5) + b * (row + 0.5) + c;
  const y = d * (col + 0.5) + e * (row + 0.5) + f;
  return proj4(src, 'EPSG:4326', [x, y]);
};

const tileXY = (lon, lat) => {
  const n = 2 ** Z;
  const phi = (lat * Math.PI) / 180;
  return [((lon + 180) / 360) * n, ((1 - Math.log(Math.tan(phi) + 1 / Math.cos(phi)) / Math.PI) / 2) * n];
};

// minimal PNG decoder: 8-bit, non-interlaced, RGB/RGBA (what Terrarium tiles are)
function decodePng(buf) {
  let p = 8;
  let w = 0, h = 0, ct = 0;
  const idat = [];
  while (p < buf.length) {
    const len = buf.readUInt32BE(p);
    const type = buf.toString('ascii', p + 4, p + 8);
    const body = buf.subarray(p + 8, p + 8 + len);
    if (type === 'IHDR') {
      w = body.readUInt32BE(0);
      h = body.readUInt32BE(4);
      ct = body[9];
      if (body[8] !== 8 || body[12] !== 0) throw new Error('unsupported PNG');
    } else if (type === 'IDAT') idat.push(body);
    p += 12 + len;
  }
  const bpp = ct === 6 ? 4 : ct === 2 ? 3 : 0;
  if (!bpp) throw new Error('unsupported PNG colour type');
  const raw = inflateSync(Buffer.concat(idat));
  const stride = w * bpp;
  const out = Buffer.alloc(h * stride);
  for (let y = 0; y < h; y++) {
    const ft = raw[y * (stride + 1)];
    for (let x = 0; x < stride; x++) {
      const v = raw[y * (stride + 1) + 1 + x];
      const left = x >= bpp ? out[y * stride + x - bpp] : 0;
      const up = y ? out[(y - 1) * stride + x] : 0;
      const ul = y && x >= bpp ? out[(y - 1) * stride + x - bpp] : 0;
      let r;
      if (ft === 0) r = v;
      else if (ft === 1) r = v + left;
      else if (ft === 2) r = v + up;
      else if (ft === 3) r = v + ((left + up) >> 1);
      else {
        const pa = Math.abs(up - ul), pb = Math.abs(left - ul), pc = Math.abs(left + up - 2 * ul);
        r = v + (pa <= pb && pa <= pc ? left : pb <= pc ? up : ul);
      }
      out[y * stride + x] = r & 255;
    }
  }
  const elev = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) elev[i] = out[i * bpp] * 256 + out[i * bpp + 1] + out[i * bpp + 2] / 256 - 32768;
  return elev;
}

// tiles covering the scene
const corners = [[-0.5, -0.5], [W - 0.5, -0.5], [-0.5, H - 0.5], [W - 0.5, H - 0.5]].map(([cc, rr]) => toLonLat(cc - 0.5, rr - 0.5));
const txs = corners.map(([lo, la]) => tileXY(lo, la));
const x0 = Math.floor(Math.min(...txs.map((t) => t[0]))) ;
const x1 = Math.floor(Math.max(...txs.map((t) => t[0])));
const y0 = Math.floor(Math.min(...txs.map((t) => t[1])));
const y1 = Math.floor(Math.max(...txs.map((t) => t[1])));
const cols = x1 - x0 + 1, rows = y1 - y0 + 1;
const MW = cols * T;
const mosaic = new Float32Array(MW * rows * T);
for (let ty = 0; ty < rows; ty++) {
  for (let tx = 0; tx < cols; tx++) {
    const url = `https://s3.amazonaws.com/elevation-tiles-prod/terrarium/${Z}/${x0 + tx}/${y0 + ty}.png`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
    const tile = decodePng(Buffer.from(await r.arrayBuffer()));
    for (let y = 0; y < T; y++) mosaic.set(tile.subarray(y * T, (y + 1) * T), (ty * T + y) * MW + tx * T);
  }
}
const MH = rows * T;
const sample = (lon, lat) => {
  const [tx, ty] = tileXY(lon, lat);
  const px = (tx - x0) * T - 0.5;
  const py = (ty - y0) * T - 0.5;
  if (px < -0.5 || py < -0.5 || px > MW - 0.5 || py > MH - 0.5) return NaN;
  const cx = Math.min(Math.max(px, 0), MW - 1), cy = Math.min(Math.max(py, 0), MH - 1);
  const ix = Math.min(Math.floor(cx), MW - 2), iy = Math.min(Math.floor(cy), MH - 2);
  const fx = cx - ix, fy = cy - iy;
  const v = (i, j) => mosaic[j * MW + i];
  return (1 - fy) * ((1 - fx) * v(ix, iy) + fx * v(ix + 1, iy)) + fy * ((1 - fx) * v(ix, iy + 1) + fx * v(ix + 1, iy + 1));
};

const k = Math.max(1, Math.round(30 / gsd));
const nc = Math.ceil(W / k), nr = Math.ceil(H / k);
const cells = [];
for (let r = 0; r < nr; r++) {
  for (let cc = 0; cc < nc; cc++) {
    const ya = r * k, yb = Math.min(H, ya + k), xa = cc * k, xb = Math.min(W, xa + k);
    let s = 0, n = 0;
    for (const fy of [1 / 6, 0.5, 5 / 6]) {
      for (const fx of [1 / 6, 0.5, 5 / 6]) {
        const [lon, lat] = toLonLat(xa + fx * (xb - xa) - 0.5, ya + fy * (yb - ya) - 0.5);
        const v = sample(lon, lat);
        if (Number.isFinite(v)) { s += v; n++; }
      }
    }
    cells.push(n >= 3 ? Math.round((s / n) * 100) / 100 : null);
  }
}
const out = {
  version: 1,
  rows: nr,
  cols: nc,
  cellPx: k,
  cells,
  source: 'AWS Terrain Tiles (SRTM-based mosaic)',
  datum: 'EGM96',
  tileZoom: Z,
  fetchedAt: new Date().toISOString(),
  notes: ['Multi-source mosaic (SRTM, NED, ETOPO ...), Web-Mercator resampled; not native SRTM GL1.', 'Orthometric heights (EGM96-like); no geoid conversion applied.'],
};
await writeFile(join(dir, 'dem_cells.json'), JSON.stringify(out));
const fin = cells.filter((v) => v !== null);
console.log(`${nr}x${nc} cells (k=${k} px), ${fin.length} valid, DEM ${Math.min(...fin)}..${Math.max(...fin)} m, tiles ${cols}x${rows} -> ${join(dir, 'dem_cells.json')}`);
