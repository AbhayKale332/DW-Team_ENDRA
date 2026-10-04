import { MessageChannel } from 'node:worker_threads';
import { expose, transfer } from 'comlink';
import { expect, it, vi } from 'vitest';
import type { Scene } from '@/domain/types';
import type { ObjectSurfaceInput } from '@/lib/objectSurface';
import { loadObjectSurface, objectSurfaceIfReady } from './objectSurfaceClient';

it('transfers worker copies while keeping scene grids usable for rendering and exports', async () => {
  const { port1, port2 } = new MessageChannel();
  const post = port1.postMessage.bind(port1);
  let sent: ArrayBuffer[] = [];
  vi.spyOn(port1, 'postMessage').mockImplementation((message, buffers = []) => {
    if (message.type === 'APPLY') sent = buffers.filter((item): item is ArrayBuffer => item instanceof ArrayBuffer);
    post(message, buffers);
  });
  vi.stubGlobal('Worker', function () { return port1; });
  expose({ objectSurface: (input: ObjectSurfaceInput) => transfer(input.heights.data, [input.heights.data.buffer]) },
    port2 as unknown as Parameters<typeof expose>[1]);
  const grid = () => ({ width: 2, height: 2, data: new Float32Array([1, 2, 3, 4]) });
  const scene = {
    heights: grid(), classes: { width: 2, height: 2, data: new Uint8Array([3, 3, 3, 3]) },
    ndsm: grid(), terrain: grid(), gsd: 0.5, product: 'nDSM',
    objects: { buildings: [{ poly: [[0, 0], [1, 0], [1, 1]] }], trees: [], water: [] },
  } as unknown as Scene;
  const kinds = { buildings: true, trees: false, water: false };
  try {
    const result = await loadObjectSurface(scene, kinds);
    expect(sent).toHaveLength(4);
    expect(sent.every((buffer) => buffer.byteLength === 0)).toBe(true);
    expect([...result]).toEqual([1, 2, 3, 4]);
    expect([...scene.heights.data]).toEqual([1, 2, 3, 4]);
    expect([...scene.classes!.data]).toEqual([3, 3, 3, 3]);
    expect([...scene.ndsm!.data]).toEqual([1, 2, 3, 4]);
    expect([...scene.terrain!.data]).toEqual([1, 2, 3, 4]);
    expect(objectSurfaceIfReady(scene, kinds)).toBe(result);
    expect(await loadObjectSurface(scene, kinds)).toBe(result);
  } finally {
    port1.close(); port2.close();
    vi.restoreAllMocks(); vi.unstubAllGlobals();
  }
});
