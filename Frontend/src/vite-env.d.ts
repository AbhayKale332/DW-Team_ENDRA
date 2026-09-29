/// <reference types="vite/client" />

declare const __APP_VERSION__: string;

interface ImportMetaEnv {
  readonly VITE_SPACE_ID?: string;
  readonly VITE_PROVIDER?: string;
}
