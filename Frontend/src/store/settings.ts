import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_SPACE_ID, IS_DESKTOP, type ProviderId } from '@/api/registry';
import type { BasemapId } from '@/lib/tiles';

export type Quality = 'fast' | 'balanced' | 'full';

interface SettingsState {
  provider: ProviderId;
  spaceId: string;
  defaultTta: boolean;
  quality: Quality;
  showStatusBar: boolean;
  postFx: boolean;
  /** Fetch a DEM after a georeferenced result and rebuild it as an absolute DSM (sends the scene's bounding box to the tile server). */
  autoAnchor: boolean;
  /** Tiles shown around georeferenced scenes. */
  basemapProvider: BasemapId;
  set: (p: Partial<Omit<SettingsState, 'set'>>) => void;
}

/** Persisted user preferences (localStorage). The Hugging Face token lives only on the server (.env HF_TOKEN). */
export const useSettings = create<SettingsState>()(
  persist(
    (set) => ({
      provider: IS_DESKTOP ? 'depthwizard-serve' : 'gradio-space',
      spaceId: DEFAULT_SPACE_ID,
      defaultTta: false,
      quality: 'balanced',
      showStatusBar: true,
      postFx: true,
      autoAnchor: !IS_DESKTOP,
      basemapProvider: 'satellite',
      set: (p) => set(p),
    }),
    { name: 'dw.settings', version: 1, partialize: ({ set: _s, ...rest }) => rest },
  ),
);
