import { lazy, Suspense, type ReactNode } from "react";
import { BrowserRouter, Link, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { Empty, Loading } from "./components/States";
import { Overview } from "./pages/Overview";
import { AppProvider } from "./state/app";

const pages: Record<string, () => Promise<{ default: () => ReactNode }>> = import.meta.glob("./pages/*.tsx") as never;
const page = (name: string) =>
  pages[`./pages/${name}.tsx`]
    ? lazy(async () => {
        const m = (await pages[`./pages/${name}.tsx`]()) as unknown as Record<string, () => ReactNode>;
        return { default: m[name] };
      })
    : () => (
        <Empty title="This page is not built yet">
          It is planned in docs/DESIGN.md and will be added in a later commit.
        </Empty>
      );

const Live = page("Live");
const Case = page("Case");
const Network = page("Network");
const Queue = page("Queue");
const Drift = page("Drift");
const Lab = page("Lab");
const Simulator = page("Simulator");
const Settings = page("Settings");

export function App() {
  return (
    <AppProvider>
      <BrowserRouter>
        <Shell>
          <Suspense fallback={<Loading what="the page" />}>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/live" element={<Live />} />
              <Route path="/cases" element={<Case />} />
              <Route path="/cases/:id" element={<Case />} />
              <Route path="/network" element={<Network />} />
              <Route path="/queue" element={<Queue />} />
              <Route path="/drift" element={<Drift />} />
              <Route path="/lab" element={<Lab />} />
              <Route path="/simulate" element={<Simulator />} />
              <Route path="/settings" element={<Settings />} />
              <Route
                path="*"
                element={
                  <Empty title="There is no page at this address" action={<Link to="/">Go to the overview</Link>}>
                    Check the address, or use Ctrl+K to jump to a page.
                  </Empty>
                }
              />
            </Routes>
          </Suspense>
        </Shell>
      </BrowserRouter>
    </AppProvider>
  );
}
