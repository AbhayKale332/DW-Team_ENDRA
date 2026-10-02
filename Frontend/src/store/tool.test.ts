import { beforeEach, describe, expect, it } from 'vitest';
import type { Scene } from '@/domain/types';
import { useScene } from './scene';
import { clearMeasure, finishMeasure, startMeasure, stopMeasure, undoMeasurePoint, useTool } from './tool';

const P = (col: number, row: number) => ({ col, row });
const add = (...pts: Array<[number, number]>) => pts.forEach(([c, r]) => useTool.getState().addPoint(P(c, r)));
const state = () => useTool.getState();

describe('measure tool state', () => {
  beforeEach(() => {
    useScene.setState({ scene: { id: 'a' } as Scene });
    stopMeasure();
  });

  it('builds a line, undoes, and finishes only with enough points', () => {
    startMeasure('length');
    add([0, 0]);
    expect(finishMeasure()).toBe(false);
    add([5, 0], [5, 0], [5, 5]); // the repeated click is one point
    expect(state().measure).toHaveLength(3);
    undoMeasurePoint();
    expect(state().measure).toEqual([P(0, 0), P(5, 0)]);
    expect(finishMeasure()).toBe(true);
    expect(state().measureDone).toBe(true);
    // the next click starts a new line
    add([9, 9]);
    expect(state()).toMatchObject({ measure: [P(9, 9)], measureDone: false });
  });

  it('closes a polygon from three corners', () => {
    startMeasure('area');
    add([0, 0], [4, 0]);
    expect(finishMeasure()).toBe(false);
    add([4, 4]);
    expect(finishMeasure()).toBe(true);
  });

  it('takes a height pair and starts over on the third click', () => {
    startMeasure('height');
    add([1, 1]);
    expect(state().measureDone).toBe(false);
    expect(finishMeasure()).toBe(false);
    add([2, 2]);
    expect(state().measureDone).toBe(true);
    add([3, 3]);
    expect(state()).toMatchObject({ measure: [P(3, 3)], measureDone: false });
  });

  it('clears on a mode switch, Esc and another scene, but not on picking the same mode', () => {
    startMeasure('length');
    add([0, 0], [1, 1]);
    startMeasure('length');
    expect(state().measure).toHaveLength(2);
    startMeasure('area');
    expect(state().measure).toHaveLength(0);
    add([0, 0]);
    clearMeasure();
    expect(state().measure).toHaveLength(0);
    add([0, 0]);
    useScene.setState({ scene: { id: 'b' } as Scene });
    expect(state()).toMatchObject({ tool: 'measure', measure: [] });
    useScene.setState({ scene: null });
    expect(state().tool).toBe('none');
  });
});
