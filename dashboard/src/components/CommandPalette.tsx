import { useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api } from "../api";
import { useApp } from "../state/app";
import { PAGES } from "./Shell";

interface Command {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

/** Ctrl+K: jump to a page, open a case by id, or run an action. */
export function CommandPalette() {
  const { paletteOpen, setPaletteOpen, theme, setTheme, role, setRole, asOf, toast, setHelpOpen } = useApp();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [index, setIndex] = useState(0);
  const ref = useRef<HTMLDialogElement>(null);
  const input = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const d = ref.current;
    if (!d) return;
    if (paletteOpen && !d.open) {
      d.showModal?.();
      setQuery("");
      setIndex(0);
      window.setTimeout(() => input.current?.focus(), 0);
    }
    if (!paletteOpen && d.open) d.close?.();
  }, [paletteOpen]);

  const close = () => setPaletteOpen(false);

  const commands = useMemo<Command[]>(() => {
    const go = (path: string) => () => {
      navigate(path);
      close();
    };
    const list: Command[] = PAGES.map((p) => ({ id: p.path, label: `Go to ${p.label.toLowerCase()}`, hint: `G ${p.key.toUpperCase()}`, run: go(p.path) }));
    list.push(
      { id: "theme", label: theme === "light" ? "Switch to dark theme" : "Switch to light theme", run: () => { setTheme(theme === "light" ? "dark" : "light"); close(); } },
      { id: "role", label: role === "analyst" ? "Switch role to admin" : "Switch role to analyst", run: () => { setRole(role === "analyst" ? "admin" : "analyst"); toast(`Role set to ${role === "analyst" ? "admin" : "analyst"}`); close(); } },
      { id: "help", label: "Show keyboard shortcuts", hint: "?", run: () => { close(); setHelpOpen(true); } },
    );
    const id = Number(query.replace(/\D/g, ""));
    if (query.trim() && Number.isFinite(id) && id > 0) {
      list.unshift({
        id: `case-${id}`,
        label: `Open case ${id}`,
        run: async () => {
          const t = await api.transaction(id, asOf);
          close();
          if (t) navigate(`/cases/${id}`);
          else toast(`No transaction ${id} at this as-of time. Check the id or move the as-of date forward.`, "error");
        },
      } as Command);
    }
    const q = query.trim().toLowerCase();
    return q && !/^\d+$/.test(q) ? list.filter((c) => c.label.toLowerCase().includes(q)) : list;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [query, theme, role, asOf]);

  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIndex((i) => Math.min(i + 1, commands.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setIndex((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      e.preventDefault();
      commands[index]?.run();
    }
  };

  return (
    <dialog ref={ref} className="palette" onClose={close} aria-label="Command palette">
      <input
        ref={input}
        type="search"
        placeholder="Type a page, an action or a transaction id"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setIndex(0);
        }}
        onKeyDown={onKeyDown}
        aria-controls="palette-list"
        aria-activedescendant={commands[index] ? `cmd-${commands[index].id}` : undefined}
      />
      <ul id="palette-list" role="listbox">
        {commands.length === 0 && <li className="muted small palette-empty">Nothing matches. Try a page name or a transaction id.</li>}
        {commands.map((c, i) => (
          <li
            key={c.id}
            id={`cmd-${c.id}`}
            role="option"
            aria-selected={i === index}
            onMouseEnter={() => setIndex(i)}
            onClick={() => c.run()}
          >
            <span>{c.label}</span>
            {c.hint && <kbd>{c.hint}</kbd>}
          </li>
        ))}
      </ul>
    </dialog>
  );
}
