import { describe, expect, it } from 'vitest';
import type { Georef } from '@/domain/types';
import { lonLatAt, proj4ForEpsg } from '../georef';
import { lonLatToTile } from '../tiles';
import { sampleDemCells } from './index';
import { decodeTerrarium, mosaicSampler } from './sources';

describe('Terrarium decoding', () => {
  it('decodes known values', () => {
    expect(decodeTerrarium(128, 0, 0)).toBe(0);
    expect(decodeTerrarium(128, 100, 128)).toBeCloseTo(100.5, 5);
    expect(decodeTerrarium(127, 246, 0)).toBe(-10);
  });

  it('samples a mosaic bilinearly in Mercator tile space', () => {
    const z = 12;
    const [tx, ty] = lonLatToTile(87.0, 20.3, z);
    const x0 = Math.floor(tx) - 1;
    const y0 = Math.floor(ty) - 1;
    const cols = 3;
    const rows = 3;
    const W = cols * 256;
    const data = new Float32Array(W * rows * 256);
    // elevation = 10 m per pixel to the east: a ramp we can invert exactly
    for (let y = 0; y < rows * 256; y++) for (let x = 0; x < W; x++) data[y * W + x] = x * 10;
    const s = mosaicSampler(z, x0, y0, cols, rows, data);
    const expected = ((tx - x0) * 256 - 0.5) * 10;
    expect(s(87.0, 20.3)).toBeCloseTo(expected, 3);
    expect(Number.isNaN(s(80, 20.3))).toBe(true);
  });
});

describe('sampleDemCells', () => {
  it('asks the DEM for the geographic position of each cell centre', () => {
    const georef: Georef = { epsg: 32645, proj4: proj4ForEpsg(32645), transform: [0.6, 0, 374684.4, 0, -0.6, 2250121.9] };
    const W = 100;
    const H = 60;
    const k = 50;
    const dem = (lon: number, lat: number) => lon * 1000 + lat;
    const g = sampleDemCells(georef, W, H, k, dem);
    expect([g.rows, g.cols]).toEqual([2, 2]);
    // cell (0,0): pixels [0,50) x [0,50); its centre is pixel index 24.5
    const ll = lonLatAt(georef, 24.5, 24.5)!;
    expect(g.data[0]).toBeCloseTo(dem(ll[0], ll[1]), 2);
    // the partial bottom row of cells is centred on its own 10 px extent (rows 50..59)
    const ll2 = lonLatAt(georef, 24.5, 54.5)!;
    expect(g.data[2]).toBeCloseTo(dem(ll2[0], ll2[1]), 2);
  });
});
