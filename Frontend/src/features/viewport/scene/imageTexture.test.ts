import { afterEach, describe, expect, it, vi } from 'vitest';
import { loadImageTexture } from './imageTexture';

const bitmap = (width = 64, height = 32) => ({ width, height, close: vi.fn() });
afterEach(() => vi.unstubAllGlobals());

describe('image texture bitmap ownership', () => {
  it('keeps the bitmap usable until disposal, then closes it', async () => {
    const image = bitmap();
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValue(image));
    const texture = await loadImageTexture(new Blob(), 256, 4);
    expect(image.close).not.toHaveBeenCalled();
    expect(texture.image).toBe(image);
    texture.dispose();
    expect(image.close).toHaveBeenCalledOnce();
  });

  it('releases both the original and resized images at their respective lifetimes', async () => {
    const original = bitmap(1024, 512);
    const resized = bitmap(256, 128);
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValueOnce(original).mockResolvedValueOnce(resized));
    const texture = await loadImageTexture(new Blob(), 256, 4);
    expect(original.close).toHaveBeenCalledOnce();
    expect(resized.close).not.toHaveBeenCalled();
    texture.dispose();
    expect(resized.close).toHaveBeenCalledOnce();
  });

  it('releases the original bitmap if resizing fails', async () => {
    const original = bitmap(1024, 512);
    vi.stubGlobal('createImageBitmap', vi.fn().mockResolvedValueOnce(original).mockRejectedValueOnce(new Error('decode failed')));
    await expect(loadImageTexture(new Blob(), 256, 4)).rejects.toThrow('decode failed');
    expect(original.close).toHaveBeenCalledOnce();
  });
});
