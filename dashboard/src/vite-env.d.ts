/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** "http" = the demo backend (EXP-200-final test period); anything else = the synthetic mock */
  readonly VITE_DATA_SOURCE?: string;
  readonly VITE_API_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
