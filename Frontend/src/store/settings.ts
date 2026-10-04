import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { DEFAULT_SPACE_ID, IS_DESKTOP, type ProviderId } from '@/api/registry';
import type { BasemapId } from '@/lib/tiles';

export type Quality = 'fast' | 'balanced' | 'full';

interface SettingsState {
  provider: ProviderId;
  desktopModelKey?: string | null;
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
      provider: 'gradio-space',
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

/** Apply a desktop default once per model selection; preserve explicit preferences afterwards. */
export async function configureDesktopInference() {
  if (!IS_DESKTOP) return;
  const response = await fetch('./desktop-api/config');
  if (!response.ok) throw new Error('Desktop inference configuration is unavailable');
  const { modelKey } = await response.json() as { modelKey: string | null };
  if (modelKey !== null && typeof modelKey !== 'string') throw new Error('Invalid desktop model selection');
  const settings = useSettings.getState();
  if (settings.desktopModelKey !== modelKey) {
    settings.set({ desktopModelKey: modelKey, provider: modelKey ? 'depthwizard-serve' : 'gradio-space' });
  }
}
