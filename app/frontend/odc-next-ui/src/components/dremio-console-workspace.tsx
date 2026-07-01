"use client";

import { ExternalLink, RefreshCw } from "lucide-react";
import { useState } from "react";

const CONSOLE_URL =
  process.env.NEXT_PUBLIC_DREMIO_CONSOLE_URL ?? "https://dremio.luumtran.dev";

export function DremioConsoleWorkspace() {
  const [reloadKey, setReloadKey] = useState(0);

  return (
    <div className="flex h-[85vh] min-h-[640px] flex-col rounded-xl border border-[var(--line)] bg-[var(--surface)] shadow-[var(--panel-shadow)]">
      <div className="flex flex-col gap-3 border-b border-[var(--line)] px-5 py-4 lg:flex-row lg:items-center lg:justify-between">
        <div className="min-w-0">
          <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
            Analytics Console
          </p>
          <h3 className="mt-1 truncate font-display text-lg font-semibold tracking-tight text-[var(--ink-strong)]">
            Dremio Console
          </h3>
          <p className="mt-0.5 truncate font-data text-xs text-[var(--muted)]">{CONSOLE_URL}</p>
        </div>

        <div className="flex shrink-0 items-center gap-2">
          <button
            type="button"
            onClick={() => setReloadKey((value) => value + 1)}
            className="inline-flex h-10 items-center justify-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 text-sm font-medium text-[var(--ink-strong)] transition hover:border-[var(--accent)]"
          >
            <RefreshCw size={16} />
            <span>Reload</span>
          </button>
          <a
            href={CONSOLE_URL}
            target="_blank"
            rel="noreferrer"
            className="inline-flex h-10 items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-3 text-sm font-semibold text-[var(--background)] transition hover:bg-[var(--accent-strong)]"
          >
            <ExternalLink size={16} />
            <span>Open in new tab</span>
          </a>
        </div>
      </div>

      <div className="relative flex-1 overflow-hidden rounded-b-xl bg-[var(--surface-elevated)]">
        <iframe
          key={reloadKey}
          src={CONSOLE_URL}
          title="Dremio Console"
          className="h-full w-full border-0"
        />
      </div>
    </div>
  );
}
