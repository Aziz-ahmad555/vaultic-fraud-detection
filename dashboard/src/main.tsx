import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "@fontsource/public-sans/400.css";
import "@fontsource/public-sans/600.css";
import "@fontsource/newsreader/400.css";
import "./theme/tokens.css";
import "./theme/base.css";
import "./theme/components.css";
import "./theme/pages.css";
import { App } from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
