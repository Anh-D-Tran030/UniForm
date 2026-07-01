"use client";

import { Activity, BarChart3, Database, Loader2, RefreshCw, SearchCheck } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type DashboardMetrics = {
  average_query_latency_ms: number;
  average_returned_match_count: number;
  average_top1_similarity_score: number;
  failed_queries: number;
  recent_events: Array<Record<string, unknown>>;
  selected_rank_distribution: {
    top_1: number;
    top_2: number;
    top_3: number;
    top_4_plus: number;
  };
  storage_failure_count: number;
  storage_success_count: number;
  successful_queries: number;
  total_queries: number;
  total_storage_events: number;
};

function StatCard({
  icon: Icon,
  label,
  value,
}: {
  icon: typeof Activity;
  label: string;
  value: string;
}) {
  return (
    <div className="group relative overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)] transition hover:border-[var(--line-strong)]">
      <div className="pointer-events-none absolute -right-8 -top-8 size-24 rounded-full bg-[radial-gradient(circle,var(--glow),transparent_70%)] opacity-0 transition group-hover:opacity-100" />
      <div className="flex items-center justify-between gap-4">
        <p className="font-data text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">{label}</p>
        <div className="flex size-10 items-center justify-center rounded-lg border border-[var(--line-strong)] bg-[var(--accent-soft)] text-[var(--accent)]">
          <Icon size={18} />
        </div>
      </div>
      <p className="mt-5 font-display text-[2.1rem] font-semibold tracking-tight text-[var(--ink-strong)]">{value}</p>
    </div>
  );
}

export function DashboardWorkspace() {
  const [metrics, setMetrics] = useState<DashboardMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  async function loadMetrics() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/metrics/dashboard", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to load metrics");
      }
      setMetrics(payload as DashboardMetrics);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "Failed to load metrics";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    let active = true;

    async function loadInitialMetrics() {
      try {
        const response = await fetch("/api/metrics/dashboard", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail ?? "Failed to load metrics");
        }
        if (active) {
          setMetrics(payload as DashboardMetrics);
        }
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : "Failed to load metrics";
        if (active) {
          setError(message);
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadInitialMetrics();

    return () => {
      active = false;
    };
  }, []);

  const rankRows = useMemo(() => {
    const distribution = metrics?.selected_rank_distribution ?? {
      top_1: 0,
      top_2: 0,
      top_3: 0,
      top_4_plus: 0,
    };
    const rows = [
      ["Top 1", distribution.top_1],
      ["Top 2", distribution.top_2],
      ["Top 3", distribution.top_3],
      ["Top 4+", distribution.top_4_plus],
    ] as const;
    const max = Math.max(1, ...rows.map(([, value]) => value));
    return rows.map(([label, value]) => ({ label, value, width: `${Math.round((value / max) * 100)}%` }));
  }, [metrics]);

  if (loading && !metrics) {
    return (
      <div className="flex min-h-[620px] items-center justify-center rounded-xl border border-[var(--line)] bg-[var(--surface)]">
        <Loader2 size={24} className="animate-spin text-[var(--accent)]" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
        <div>
          <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
            Runtime Telemetry
          </p>
          <h3 className="mt-2 font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">Prototype health snapshot</h3>
        </div>
        <button
          type="button"
          onClick={() => void loadMetrics()}
          className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface)] px-4 text-sm font-medium text-[var(--ink-strong)] transition hover:border-[var(--accent)]"
        >
          <RefreshCw size={17} />
          <span>Refresh</span>
        </button>
      </div>

      {error ? (
        <div className="rounded-lg border border-[var(--accent-line)] bg-[var(--danger-soft)] px-4 py-3 text-sm text-[var(--danger-text)]">
          {error}
        </div>
      ) : null}

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <StatCard icon={SearchCheck} label="Total Queries" value={String(metrics?.total_queries ?? 0)} />
        <StatCard icon={Activity} label="Avg Query Latency" value={`${metrics?.average_query_latency_ms ?? 0} ms`} />
        <StatCard icon={BarChart3} label="Avg Top-1 Score" value={`${Math.round((metrics?.average_top1_similarity_score ?? 0) * 100)}%`} />
        <StatCard icon={Database} label="Stored Forms" value={String(metrics?.storage_success_count ?? 0)} />
      </div>

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_420px]">
        <section className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)]">
          <div className="flex items-center justify-between gap-4 border-b border-[var(--line)] pb-4">
            <div>
              <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
                Selected Rank Distribution
              </p>
              <h3 className="mt-2 font-display text-xl font-semibold tracking-tight text-[var(--ink-strong)]">Top-k chosen by user</h3>
            </div>
            <p className="text-sm text-[var(--muted)]">
              {metrics?.successful_queries ?? 0} successful / {metrics?.failed_queries ?? 0} failed
            </p>
          </div>

          <div className="mt-6 space-y-5">
            {rankRows.map((row) => (
              <div key={row.label}>
                <div className="mb-2 flex items-center justify-between text-sm">
                  <span className="font-data font-medium text-[var(--ink-strong)]">{row.label}</span>
                  <span className="font-data text-[var(--muted)]">{row.value}</span>
                </div>
                <div className="h-3 overflow-hidden rounded-full border border-[var(--line)] bg-[var(--surface-subtle)]">
                  <div
                    className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),var(--accent-strong))] shadow-[0_0_10px_-1px_var(--glow-strong)]"
                    style={{ width: row.width }}
                  />
                </div>
              </div>
            ))}
          </div>

          <div className="mt-8 grid gap-4 md:grid-cols-3">
            <div className="rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] p-4">
              <p className="font-data text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Avg Matches</p>
              <p className="mt-3 font-display text-2xl font-semibold text-[var(--ink-strong)]">
                {metrics?.average_returned_match_count ?? 0}
              </p>
            </div>
            <div className="rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] p-4">
              <p className="font-data text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Store Success</p>
              <p className="mt-3 font-display text-2xl font-semibold text-[var(--primary)]">{metrics?.storage_success_count ?? 0}</p>
            </div>
            <div className="rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] p-4">
              <p className="font-data text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Store Failures</p>
              <p className="mt-3 font-display text-2xl font-semibold text-[var(--danger-text)]">{metrics?.storage_failure_count ?? 0}</p>
            </div>
          </div>
        </section>

        <aside className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)]">
          <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
            Recent Events
          </p>
          <div className="mt-5 max-h-[520px] space-y-3 overflow-y-auto pr-1">
            {metrics?.recent_events.length ? (
              metrics.recent_events.map((event, index) => (
                <div key={index} className="rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-sm font-semibold capitalize text-[var(--ink-strong)]">
                      {String(event.event_type ?? "event")}
                    </span>
                    <span className="text-xs text-[var(--muted)]">
                      {typeof event.timestamp === "string" ? new Date(event.timestamp).toLocaleTimeString() : "--"}
                    </span>
                  </div>
                  <pre className="mt-2 max-h-24 overflow-auto whitespace-pre-wrap font-data text-[11px] leading-4 text-[var(--muted)]">
                    {JSON.stringify(event, null, 2)}
                  </pre>
                </div>
              ))
            ) : (
              <div className="flex h-[360px] items-center justify-center px-6 text-center text-sm text-[var(--muted)]">
                Run a query, choose a match, or store to MinIO to populate telemetry.
              </div>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}
