/** Minimal NumPy .npy reader/writer (format v1/v2, C order). Ported from the v5 viewer. */

export interface NpyArray {
  data: Float32Array;
  shape: number[];
}

export function parseNpy(buffer: ArrayBuffer): NpyArray {
  const b = new Uint8Array(buffer);
  const magic = String.fromCharCode(...b.slice(1, 6));
  if (b[0] !== 0x93 || magic !== 'NUMPY') throw new Error('Not a .npy file');
  const major = b[6];
  const dv = new DataView(buffer);
  const hlenBytes = major === 1 ? 2 : 4;
  const hlen = major === 1 ? dv.getUint16(8, true) : dv.getUint32(8, true);
  const start = 8 + hlenBytes;
  const header = new TextDecoder().decode(b.slice(start, start + hlen));
  const descr = /'descr':\s*'([^']+)'/.exec(header)?.[1];
  const fortran = /'fortran_order':\s*(True|False)/.exec(header)?.[1] === 'True';
  const shapeStr = /'shape':\s*\(([^)]*)\)/.exec(header)?.[1];
  if (!descr || shapeStr === undefined) throw new Error('Malformed .npy header');
  if (fortran) throw new Error('Fortran-order .npy is not supported');
  const shape = shapeStr
    .split(',')
    .map((s) => s.trim())
    .filter(Boolean)
    .map(Number);
  const off = start + hlen;
  const n = shape.reduce((a, c) => a * c, 1);
  if (descr.startsWith('>')) throw new Error('Big-endian .npy is not supported');
  const kind = descr.replace(/^[<>|=]/, '');
  // Copy into a fresh, aligned buffer so the result is transferable and independent.
  const slice = buffer.slice(off);
  let data: Float32Array;
  switch (kind) {
    case 'f4':
      data = new Float32Array(slice, 0, n);
      break;
    case 'f8':
      data = Float32Array.from(new Float64Array(slice, 0, n));
      break;
    case 'u1':
      data = Float32Array.from(new Uint8Array(slice, 0, n));
      break;
    case 'u2':
      data = Float32Array.from(new Uint16Array(slice, 0, n));
      break;
    case 'i2':
      data = Float32Array.from(new Int16Array(slice, 0, n));
      break;
    case 'i4':
      data = Float32Array.from(new Int32Array(slice, 0, n));
      break;
    default:
      throw new Error(`Unsupported .npy dtype ${descr}`);
  }
  return { data, shape };
}

/** Write a little-endian float32 C-order .npy (format v1). */
export function writeNpyF32(data: Float32Array, shape: number[]): Uint8Array {
  let header = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shape.join(', ')}${shape.length === 1 ? ',' : ''}), }`;
  // Pad so that magic(6)+ver(2)+len(2)+header is a multiple of 64, ending with \n.
  const pre = 10;
  const total = Math.ceil((pre + header.length + 1) / 64) * 64;
  header = header.padEnd(total - pre - 1, ' ') + '\n';
  const out = new Uint8Array(total + data.byteLength);
  out.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0);
  new DataView(out.buffer).setUint16(8, header.length, true);
  out.set(new TextEncoder().encode(header), pre);
  out.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), total);
  return out;
}
