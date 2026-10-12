/**
 * The ONE place the data source is chosen. Default: the synthetic mock. With
 * VITE_DATA_SOURCE=http (and optionally VITE_API_URL, default http://127.0.0.1:8000) the
 * dashboard reads the demo backend (services/api/demo_server.py): the EXP-200-final test
 * period, labelled "IEEE-CIS test period, final run". No page or component changes.
 */
import { createHttpApi } from "./http/httpApi";
import { createMockApi } from "./mock/mockApi";
import type { Api } from "./types";

export const DATA_SOURCE: "mock" | "http" = import.meta.env.VITE_DATA_SOURCE === "http" ? "http" : "mock";

export const api: Api =
  DATA_SOURCE === "http" ? createHttpApi(import.meta.env.VITE_API_URL || undefined) : createMockApi();
