import { useEffect, useState } from "react";

export interface DataState<T> {
  data: T | null;
  error: Error | null;
  loading: boolean;
}

/** Load async data; re-runs when deps change, ignores stale responses. */
export function useData<T>(load: () => Promise<T>, deps: unknown[]): DataState<T> & { reload: () => void } {
  const [state, setState] = useState<DataState<T>>({ data: null, error: null, loading: true });
  const [nonce, setNonce] = useState(0);
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    load()
      .then((data) => live && setState({ data, error: null, loading: false }))
      .catch((error: unknown) =>
        live && setState({ data: null, error: error instanceof Error ? error : new Error(String(error)), loading: false }),
      );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, nonce]);
  return { ...state, reload: () => setNonce((n) => n + 1) };
}
