import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { decodeGray8Png, encodeGray8Png } from './png';
import { classNamesFromMeta, classPaletteRGBA, classShares } from '@/theme/classes';

const buf = (p: string) => {
  const b = readFileSync(p);
  return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength) as ArrayBuffer;
};

/** jsdom's Blob has no arrayBuffer(); browsers do. */
const blobBytes = (b: Blob) =>
  new Promise<ArrayBuffer>((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(r.result as ArrayBuffer);
    r.onerror = () => rej(r.error);
    r.readAsArrayBuffer(b);
  });

/** The pattern both fixtures encode (see the generator in the commit that added them). */
function expected(w: number, h: number) {
  const out = new Uint8Array(w * h);
  for (let r = 0; r < h; r++) for (let c = 0; c < w; c++) out[r * w + c] = (r * 31 + c * 17 + ((r * c) % 5)) % 256;
  return out;
}

describe('8-bit greyscale PNG (seg.png)', () => {
  it('decodes a Pillow-written PNG exactly', () => {
    const png = decodeGray8Png(buf('src/lib/__fixtures__/gray8_pil.png'));
    expect([png.width, png.height]).toEqual([29, 13]);
    expect(png.data).toEqual(expected(29, 13));
  });

  it('undoes every row filter (None, Sub, Up, Average, Paeth)', () => {
    const png = decodeGray8Png(buf('src/lib/__fixtures__/gray8_filters.png'));
    expect(png.data).toEqual(expected(29, 13));
  });

  it('round-trips through the project encoder', async () => {
    const data = expected(31, 7);
    const blob = encodeGray8Png(data, 31, 7);
    const back = decodeGray8Png(await blobBytes(blob));
    expect(back).toEqual({ data, width: 31, height: 7 });
  });

  it('rejects anything that is not 8-bit greyscale', () => {
    expect(() => decodeGray8Png(new Uint8Array([1, 2, 3]).buffer)).toThrow(/Not a PNG/);
  });
});

describe('class palette', () => {
  it('maps meta.json classes to names by id and colours by name', () => {
    const names = classNamesFromMeta({ '0': 'other', '3': 'building', '6': 'tree', x: 'junk' });
    expect(names[3]).toBe('building');
    expect(names[6]).toBe('tree');
    const lut = classPaletteRGBA(names);
    expect(Array.from(lut.subarray(6 * 4, 6 * 4 + 4))).toEqual([0x2e, 0x7d, 0x32, 255]);
    expect(Array.from(lut.subarray(3 * 4, 3 * 4 + 4))).toEqual([0xe0, 0x5a, 0x47, 255]);
  });

  it('reports class shares largest first', () => {
    const shares = classShares({ data: new Uint8Array([6, 6, 6, 3]), width: 2, height: 2, names: [] });
    expect(shares).toEqual([
      { id: 6, share: 0.75 },
      { id: 3, share: 0.25 },
    ]);
  });
});
