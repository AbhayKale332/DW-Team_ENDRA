import { create } from 'zustand';

export type Tool = 'none' | 'probe' | 'measure' | 'profile';

export const TOOL_LABELS: Record<Tool, string> = {
  none: 'Navigate',
  probe: 'Probe height',
  measure: 'Measure distance & slope',
  profile: 'Elevation profile',
};

/** A point on the height grid in fractional pixel coordinates (pixel centres at integers). */
export interface GridPoint {
  col: number;
  row: number;
}

interface ToolState {
  tool: Tool;
  probe: GridPoint | null;
  measure: GridPoint[];
  profile: GridPoint[];
  hover: GridPoint | null;
  /** Pointer position over the canvas (CSS px from its top-left) for the hover card; null when off the terrain. */
  hoverScreen: { x: number; y: number } | null;
  /** 0..1 position along the profile highlighted from the chart. */
  profileCursor: number | null;
  set: (p: Partial<Omit<ToolState, 'set' | 'clear' | 'addPoint'>>) => void;
  addPoint: (p: GridPoint) => void;
  clear: () => void;
}

export const useTool = create<ToolState>()((set, get) => ({
  tool: 'none',
  probe: null,
  measure: [],
  profile: [],
  hover: null,
  hoverScreen: null,
  profileCursor: null,
  set: (p) => set(p),
  addPoint: (p) => {
    const { tool, measure, profile } = get();
    if (tool === 'probe') set({ probe: p });
    else if (tool === 'measure') set({ measure: measure.length >= 2 ? [p] : [...measure, p] });
    else if (tool === 'profile') set({ profile: [...profile, p].slice(-24) });
  },
  clear: () => set({ probe: null, measure: [], profile: [], profileCursor: null }),
}));
