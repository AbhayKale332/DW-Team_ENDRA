import { create } from 'zustand';

export type InspectorTab = 'layers' | 'validation' | 'info';
export type Dialog = 'shortcuts' | 'model' | 'about' | 'settings' | null;

interface UiState {
  projectOpen: boolean;
  inspectorOpen: boolean;
  inspectorTab: InspectorTab;
  dialog: Dialog;
  tourOpen: boolean;
  set: (p: Partial<Omit<UiState, 'set' | 'openInspector'>>) => void;
  openInspector: (tab: InspectorTab) => void;
}

export const useUi = create<UiState>()((set) => ({
  projectOpen: true,
  inspectorOpen: true,
  inspectorTab: 'layers',
  dialog: null,
  tourOpen: false,
  set: (p) => set(p),
  openInspector: (tab) => set({ inspectorOpen: true, inspectorTab: tab }),
}));
