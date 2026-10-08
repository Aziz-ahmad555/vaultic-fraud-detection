import cytoscape, { type Core, type ElementDefinition } from "cytoscape";
import { useEffect, useMemo, useRef } from "react";
import type { Network } from "../api/types";
import { useApp } from "../state/app";

const SHAPE: Record<string, string> = {
  uid: "ellipse",
  card: "round-rectangle",
  device: "diamond",
  email: "triangle",
  address: "hexagon",
};

function cssVar(name: string) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim() || "#888";
}

/**
 * Entity network (Cytoscape). Edges use the relational view colour; nodes are shaped by
 * entity type; ring members get a thick risk-hue border; nodes with known fraud are filled in
 * the risk hue. Only edges with time <= `until` are drawn (the time slider).
 */
export function NetworkGraph({ network, until, focus, onSelect, mini = false, label }: {
  network: Network;
  until: number;
  focus?: string | null;
  onSelect?: (id: string) => void;
  mini?: boolean;
  label: string;
}) {
  const { theme } = useApp();
  const box = useRef<HTMLDivElement>(null);
  const cy = useRef<Core | null>(null);

  const elements = useMemo<ElementDefinition[]>(() => {
    const edges = network.edges.filter((e) => e.time <= until);
    const live = new Set(edges.flatMap((e) => [e.source, e.target]));
    const nodes = network.nodes.filter((n) => live.has(n.id) || n.id === focus);
    return [
      ...nodes.map((n) => ({
        data: {
          id: n.id,
          label: mini ? "" : n.type === "uid" ? n.label : `${n.type}: ${n.label}`,
          type: n.type,
          fraud: n.knownFraud,
          ring: n.ring ?? 0,
          focus: n.id === focus ? 1 : 0,
        },
      })),
      ...edges.map((e) => ({ data: { id: e.id, source: e.source, target: e.target } })),
    ];
  }, [network, until, focus, mini]);

  useEffect(() => {
    if (!box.current) return;
    const ink = cssVar("--ink");
    const sheet = cssVar("--sheet");
    const graphite = cssVar("--graphite");
    const rel = cssVar("--view-relational");
    const risk3 = cssVar("--risk-3");
    const risk5 = cssVar("--risk-5");
    cy.current?.destroy();
    const instance = cytoscape({
      container: box.current,
      elements,
      minZoom: 0.2,
      maxZoom: 3,
      style: [
        {
          selector: "node",
          style: {
            shape: ((el: cytoscape.NodeSingular) => SHAPE[el.data("type") as string] ?? "ellipse") as never,
            width: mini ? 14 : 20,
            height: mini ? 14 : 20,
            "background-color": sheet,
            "border-width": 1.5,
            "border-color": graphite,
            label: "data(label)",
            color: ink,
            "font-family": "Public Sans, sans-serif",
            "font-size": 10,
            "text-valign": "bottom",
            "text-margin-y": 3,
          },
        },
        { selector: "node[fraud > 0]", style: { "background-color": risk3 } },
        { selector: "node[ring > 0]", style: { "border-width": 3.5, "border-color": risk5 } },
        { selector: "node[focus = 1]", style: { "border-width": 4, "border-color": ink, width: 26, height: 26 } },
        { selector: "edge", style: { width: 1.5, "line-color": rel, "curve-style": "haystack", opacity: 0.85 } },
        { selector: ":selected", style: { "border-color": ink, "border-width": 4 } },
      ],
      layout: { name: "cose", animate: false, randomize: false, nodeRepulsion: () => (mini ? 3000 : 7000), idealEdgeLength: () => (mini ? 30 : 55) } as never,
    });
    instance.on("tap", "node", (e) => onSelect?.(e.target.id()));
    cy.current = instance;
    return () => instance.destroy();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [elements, theme, mini]);

  return <div ref={box} className={mini ? "network-mini" : "network"} role="img" aria-label={label} />;
}
