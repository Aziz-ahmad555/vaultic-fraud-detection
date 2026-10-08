/**
 * The ONE place the data source is chosen. To switch to the FastAPI backend, replace the
 * line below with `export const api: Api = createHttpApi(import.meta.env.VITE_API_URL)`
 * (an implementation of the same Api interface); no page or component changes.
 */
import { createMockApi } from "./mock/mockApi";
import type { Api } from "./types";

export const api: Api = createMockApi();
export const DATA_SOURCE: "mock" | "http" = "mock";
