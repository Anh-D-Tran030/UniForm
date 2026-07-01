"use client";

import { AlertCircle, CheckCircle2, ClipboardList, Loader2, RefreshCw } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

type UploadedForm = {
  bronze_path: string | null;
  created_at: string | null;
  run_id: string;
  silver_path: string | null;
  source_file_name: string | null;
  status: "pending" | "processed";
  template_id: string;
};

type Filter = "all" | "pending" | "processed";

function formatDate(value: string | null) {
  if (!value) {
    return "--";
  }
  return new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(value));
}

export function UploadedFormsWorkspace() {
  const [forms, setForms] = useState<UploadedForm[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedForm, setSelectedForm] = useState<UploadedForm | null>(null);
  const [jsonPreview, setJsonPreview] = useState("");
  const [jsonLoading, setJsonLoading] = useState(false);
  const [jsonError, setJsonError] = useState<string | null>(null);

  async function loadForms() {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch("/api/storage/forms", { cache: "no-store" });
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to load uploaded forms");
      }
      setForms((payload.forms ?? []) as UploadedForm[]);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "Failed to load uploaded forms";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  async function selectForm(form: UploadedForm) {
    setSelectedForm(form);
    setJsonPreview("");
    setJsonError(null);

    if (!form.silver_path) {
      return;
    }

    setJsonLoading(true);
    try {
      const response = await fetch(
        `/api/storage/forms/${encodeURIComponent(form.template_id)}/${encodeURIComponent(form.run_id)}/json`,
        { cache: "no-store" },
      );
      const payload = await response.json();
      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to load stored JSON");
      }
      setJsonPreview(JSON.stringify(payload, null, 2));
    } catch (previewError) {
      const message = previewError instanceof Error ? previewError.message : "Failed to load stored JSON";
      setJsonError(message);
    } finally {
      setJsonLoading(false);
    }
  }

  useEffect(() => {
    let active = true;

    async function loadInitialForms() {
      try {
        const response = await fetch("/api/storage/forms", { cache: "no-store" });
        const payload = await response.json();
        if (!response.ok) {
          throw new Error(payload.detail ?? "Failed to load uploaded forms");
        }
        if (active) {
          setForms((payload.forms ?? []) as UploadedForm[]);
        }
      } catch (loadError) {
        const message = loadError instanceof Error ? loadError.message : "Failed to load uploaded forms";
        if (active) {
          setError(message);
        }
      } finally {
        if (active) {
          setLoading(false);
        }
      }
    }

    void loadInitialForms();

    return () => {
      active = false;
    };
  }, []);

  const filteredForms = useMemo(() => {
    if (filter === "all") {
      return forms;
    }
    return forms.filter((form) => form.status === filter);
  }, [filter, forms]);

  const processedCount = forms.filter((form) => form.status === "processed").length;
  const pendingCount = forms.filter((form) => form.status === "pending").length;

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1fr)_460px]">
      <section className="min-w-0 rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)]">
        <div className="flex flex-col gap-4 border-b border-[var(--line)] pb-5 lg:flex-row lg:items-center lg:justify-between">
          <div>
            <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
              MinIO Artifacts
            </p>
            <h3 className="mt-2 font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">
              {forms.length} uploaded forms
            </h3>
          </div>
          <button
            type="button"
            onClick={() => void loadForms()}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-4 text-sm font-medium text-[var(--ink-strong)] transition hover:border-[var(--accent)]"
          >
            <RefreshCw size={17} />
            <span>Refresh</span>
          </button>
        </div>

        <div className="mt-5 flex flex-wrap gap-2">
          {[
            ["all", `All ${forms.length}`],
            ["processed", `Processed ${processedCount}`],
            ["pending", `Pending ${pendingCount}`],
          ].map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setFilter(value as Filter)}
              className={`h-10 rounded-lg px-4 text-sm font-medium transition ${
                filter === value
                  ? "bg-[var(--accent)] text-[var(--background)]"
                  : "border border-[var(--line)] bg-[var(--surface-subtle)] text-[var(--ink-strong)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {error ? (
          <div className="mt-5 flex gap-3 rounded-lg border border-[var(--accent-line)] bg-[var(--danger-soft)] px-4 py-3 text-sm text-[var(--danger-text)]">
            <AlertCircle size={18} className="shrink-0" />
            <span>{error}</span>
          </div>
        ) : null}

        <div className="mt-5 overflow-hidden rounded-xl border border-[var(--line)]">
          <div className="overflow-x-auto">
            <div className="min-w-[560px]">
          <div className="grid grid-cols-[1.1fr_1fr_100px_1fr] gap-4 border-b border-[var(--line)] bg-[var(--surface-subtle)] px-4 py-3 font-data text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">
            <span>Run</span>
            <span>Template</span>
            <span>Status</span>
            <span>Created</span>
          </div>

          {loading ? (
            <div className="flex h-[340px] items-center justify-center">
              <Loader2 size={24} className="animate-spin text-[var(--accent)]" />
            </div>
          ) : filteredForms.length ? (
            <div className="max-h-[620px] overflow-y-auto">
              {filteredForms.map((form) => {
                const selected = selectedForm?.run_id === form.run_id && selectedForm.template_id === form.template_id;
                return (
                  <button
                    key={`${form.template_id}-${form.run_id}`}
                    type="button"
                    onClick={() => void selectForm(form)}
                    className={`grid w-full grid-cols-[1.1fr_1fr_100px_1fr] gap-4 border-b border-[var(--line)] px-4 py-4 text-left text-sm transition last:border-b-0 ${
                      selected ? "bg-[var(--accent-soft)]" : "bg-[var(--surface-elevated)] hover:bg-[var(--surface-subtle)]"
                    }`}
                  >
                    <span className="min-w-0">
                      <span className="block truncate font-semibold text-[var(--ink-strong)]">{form.run_id}</span>
                      <span className="mt-1 block truncate text-xs text-[var(--muted)]">
                        {form.source_file_name ?? form.bronze_path ?? "No source name"}
                      </span>
                    </span>
                    <span className="truncate text-[var(--ink-strong)]">{form.template_id}</span>
                    <span>
                      <span
                        className={`inline-flex h-7 items-center gap-1.5 rounded-full px-2.5 font-data text-xs font-semibold ${
                          form.status === "processed"
                            ? "bg-[var(--success-soft)] text-[var(--primary)]"
                            : "bg-[var(--warning-soft)] text-[var(--text-main)]"
                        }`}
                      >
                        <span
                          className={`size-1.5 rounded-full ${
                            form.status === "processed"
                              ? "bg-[var(--accent)] shadow-[0_0_8px_var(--glow-strong)]"
                              : "bg-[var(--muted)]"
                          }`}
                        />
                        {form.status}
                      </span>
                    </span>
                    <span className="text-[var(--muted)]">{formatDate(form.created_at)}</span>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="flex h-[340px] flex-col items-center justify-center px-6 text-center">
              <ClipboardList size={28} className="text-[var(--muted)]" />
              <p className="mt-4 text-sm font-medium text-[var(--ink-strong)]">No uploaded forms found</p>
              <p className="mt-2 text-sm text-[var(--muted)]">Store an extraction to MinIO to populate this list.</p>
            </div>
          )}
            </div>
          </div>
        </div>
      </section>

      <aside className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)]">
        <div className="flex items-center justify-between gap-4">
          <div>
            <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
              Silver JSON
            </p>
            <h3 className="mt-2 font-display text-xl font-semibold tracking-tight text-[var(--ink-strong)]">
              {selectedForm ? selectedForm.run_id : "Select a form"}
            </h3>
          </div>
          {selectedForm?.status === "processed" ? (
            <CheckCircle2 size={22} className="text-[var(--primary)]" />
          ) : null}
        </div>

        {selectedForm ? (
          <div className="mt-5 space-y-3 text-sm">
            <div className="rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] p-3">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Bronze</p>
              <p className="mt-1 break-all text-[var(--ink-strong)]">{selectedForm.bronze_path ?? "--"}</p>
            </div>
            <div className="rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] p-3">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-[var(--muted)]">Silver</p>
              <p className="mt-1 break-all text-[var(--ink-strong)]">{selectedForm.silver_path ?? "--"}</p>
            </div>
          </div>
        ) : null}

        <div className="mt-5 min-h-[520px] rounded-xl border border-[var(--line)] bg-[var(--surface-subtle)] p-4">
          {jsonLoading ? (
            <div className="flex h-[480px] items-center justify-center">
              <Loader2 size={22} className="animate-spin text-[var(--accent)]" />
            </div>
          ) : jsonError ? (
            <div className="rounded-lg border border-[var(--accent-line)] bg-[var(--danger-soft)] px-4 py-3 text-sm text-[var(--danger-text)]">
              {jsonError}
            </div>
          ) : jsonPreview ? (
            <pre className="max-h-[540px] overflow-auto whitespace-pre-wrap font-data text-xs leading-5 text-[var(--ink-strong)]">
              {jsonPreview}
            </pre>
          ) : (
            <div className="flex h-[480px] items-center justify-center px-6 text-center text-sm text-[var(--muted)]">
              Processed forms show their stored KVP JSON here.
            </div>
          )}
        </div>
      </aside>
    </div>
  );
}
