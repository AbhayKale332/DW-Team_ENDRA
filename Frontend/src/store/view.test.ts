import { beforeEach, describe, expect, it } from 'vitest';
import type { SceneObjects } from '@/domain/types';
import { activeObjectKinds, anyObjectKind, DEFAULT_OBJECT_KINDS, objectKindsKey, toggleAllObjects, toggleObjectKind, useView } from './view';

const objects = (n: { trees: number; buildings: number; water: number }): SceneObjects => ({
  version: 1,
  width: 10,
  height: 10,
  gsd: 0.5,
  trees: Array.from({ length: n.trees }, () => ({ x: 1, y: 1, h: 5, r: 1 })),
  buildings: Array.from({ length: n.buildings }, () => ({ poly: [[0, 0], [1, 0], [1, 1]], h: 5, hMax: 5, areaM2: 1 })),
  water: Array.from({ length: n.water }, () => ({ poly: [[0, 0], [1, 0], [1, 1]], areaM2: 1 })),
  truncated: { trees: false, buildings: false, water: false },
});

describe('3D object classes', () => {
  beforeEach(() => useView.getState().reset());

  it('defaults to buildings only', () => {
    expect(useView.getState().objectKinds).toEqual({ buildings: true, trees: false, water: false });
    expect(DEFAULT_OBJECT_KINDS.buildings).toBe(true);
  });

  it('toggles one class at a time', () => {
    toggleObjectKind('trees');
    expect(useView.getState().objectKinds).toEqual({ buildings: true, trees: true, water: false });
    toggleObjectKind('buildings');
    expect(useView.getState().objectKinds).toEqual({ buildings: false, trees: true, water: false });
  });

  it('O turns everything off, then restores the previous selection', () => {
    toggleObjectKind('water');
    const before = useView.getState().objectKinds;
    toggleAllObjects();
    expect(anyObjectKind(useView.getState().objectKinds)).toBe(false);
    toggleAllObjects();
    expect(useView.getState().objectKinds).toEqual(before);
  });

  it('a class the scene does not have is never active', () => {
    const k = activeObjectKinds({ buildings: true, trees: true, water: true }, objects({ trees: 3, buildings: 0, water: 1 }));
    expect(k).toEqual({ buildings: false, trees: true, water: true });
    expect(anyObjectKind(activeObjectKinds(DEFAULT_OBJECT_KINDS, null))).toBe(false);
    expect(objectKindsKey(k)).toBe('b0t1w1');
  });
});
