import { create } from 'zustand';

export type AnchorStatus = 'idle' | 'running' | 'done' | 'error' | 'blocked';

interface AnchorState {
  status: AnchorStatus;
  /** Progress line while running, or the reason it failed / cannot run. */
  message: string | null;
  /** Scene the status belongs to, so a stale result is never shown against a newer scene. */
  sceneId: string | null;
  set: (p: Partial<Omit<AnchorState, 'set'>>) => void;
}

export const useAnchor = create<AnchorState>()((set) => ({ status: 'idle', message: null, sceneId: null, set: (p) => set(p) }));
