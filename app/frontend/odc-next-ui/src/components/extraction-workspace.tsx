"use client";

import Link from "next/link";
import { AlertCircle, ChevronDown, FileSearch, Loader2, Plus, Save, Trash2 } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { KeyValueOverlayViewer, type KeyValueOverlayPair } from "@/components/key-value-overlay-viewer";

type KeyValuePair = {
  key: string;
  key_bbox: number[];
  score: number;
  value: string;
  value_bbox: number[];
};

type KeyValueResponse = {
  key_values: KeyValuePair[];
};

type ReviewKeyValuePair = {
  key: string;
  key_bbox?: number[];
  score?: number;
  source?: "custom";
  value: string;
  value_bbox?: number[];
};

type ExtractionJobResponse = {
  error: string | null;
  kvp_payload: KeyValueResponse | null;
  status: "accepted" | "running" | "kvp_ready" | "failed" | "expired";
};

type MappingRow = {
  id: string;
  key: string;
  key_bbox?: number[];
  score?: number;
  value: string;
  value_bbox?: number[];
  valueMode: "custom" | "detected";
};

type ExtractionWorkspaceProps = {
  fileName: string | null;
  initialRunId: string | null;
  queryPath: string | null;
  score: number;
  templateId: string;
  templateImagePath: string | null;
  templateName: string;
  uploadId: string | null;
};

function imageUrl(imagePath: string | null) {
  if (!imagePath) {
    return null;
  }

  return `/api/image?path=${encodeURIComponent(imagePath)}`;
}

function uploadImageUrl(uploadId: string | null) {
  if (!uploadId) {
    return null;
  }

  return `/api/uploads/${encodeURIComponent(uploadId)}/image`;
}

function pairIdFor(item: KeyValuePair, index: number) {
  return `${index}-${item.key.trim()}-${item.value.trim()}`;
}

function makeRow(key = "", value = ""): MappingRow {
  return {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    key,
    key_bbox: undefined,
    score: undefined,
    value,
    value_bbox: undefined,
    valueMode: "custom",
  };
}

function newRunId() {
  return crypto.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2, 10)}`;
}

function buildReviewPayload(runId: string, templateId: string, fileName: string | null, pairs: ReviewKeyValuePair[]) {
  return {
    run_id: runId,
    template_id: templateId || "unknown",
    source_file_name: fileName,
    key_values: pairs,
  };
}

function rowsToReviewPairs(rows: MappingRow[]) {
  return rows
    .map((row) => {
      const key = row.key.trim();
      const value = row.value.trim();
      if (!key && !value) {
        return null;
      }

      const pair: ReviewKeyValuePair = { key, value };
      if (row.valueMode === "custom") {
        pair.source = "custom";
      } else {
        if (typeof row.score === "number") {
          pair.score = row.score;
        }
        if (row.key_bbox) {
          pair.key_bbox = row.key_bbox;
        }
        if (row.value_bbox) {
          pair.value_bbox = row.value_bbox;
        }
      }

      return pair;
    })
    .filter((pair): pair is ReviewKeyValuePair => Boolean(pair));
}

export function ExtractionWorkspace({
  fileName,
  initialRunId,
  queryPath,
  score,
  templateId,
  templateImagePath,
  templateName,
  uploadId,
}: ExtractionWorkspaceProps) {
  const [pairs, setPairs] = useState<KeyValuePair[]>([]);
  const [loading, setLoading] = useState(Boolean(queryPath || uploadId));
  const [error, setError] = useState<string | null>(null);
  const [rows, setRows] = useState<MappingRow[]>([]);
  const rowsRef = useRef<MappingRow[]>([]);
  const [runId, setRunId] = useState(initialRunId ?? "");
  const runIdRef = useRef(runId);
  const [kvpJsonText, setKvpJsonText] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [activePairId, setActivePairId] = useState<string | null>(null);
  const [pinnedPairId, setPinnedPairId] = useState<string | null>(null);
  const [storing, setStoring] = useState(false);
  const [storageResult, setStorageResult] = useState<Record<string, unknown> | null>(null);

  useEffect(() => {
    if (!queryPath && !uploadId) {
      return;
    }

    let cancelled = false;

    function applyKeyValuePayload(payload: KeyValueResponse, activeRunId: string) {
      const nextPairs = payload.key_values ?? [];
      setPairs(nextPairs);
      setKvpJsonText(JSON.stringify(buildReviewPayload(activeRunId, templateId, fileName, nextPairs), null, 2));
      setJsonError(null);
      setStorageResult(null);
      setActivePairId(null);
      setPinnedPairId(null);
      const nextRows = nextPairs.length
        ? nextPairs.slice(0, 8).map((item) => ({
            id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
            key: item.key,
            key_bbox: item.key_bbox,
            score: item.score,
            value: item.value,
            value_bbox: item.value_bbox,
            valueMode: "detected" as const,
          }))
        : [makeRow()];
      rowsRef.current = nextRows;
      setRows(nextRows);
    }

    async function loadPrecomputedKeyValues(activeRunId: string) {
      if (!activeRunId) {
        return null;
      }

      for (let attempt = 0; attempt < 90; attempt += 1) {
        const response = await fetch(`/api/extraction/jobs/${encodeURIComponent(activeRunId)}`, {
          cache: "no-store",
        });

        if (response.status === 404) {
          return null;
        }

        const payload = (await response.json()) as ExtractionJobResponse | { detail?: string };
        if (!response.ok) {
          return null;
        }

        if ("status" in payload && payload.status === "kvp_ready" && payload.kvp_payload) {
          return payload.kvp_payload;
        }

        if ("status" in payload && (payload.status === "failed" || payload.status === "expired")) {
          return null;
        }

        await new Promise((resolve) => setTimeout(resolve, 1000));

        if (cancelled) {
          return null;
        }
      }

      return null;
    }

    async function loadKeyValues() {
      setLoading(true);
      setError(null);
      const activeRunId = runIdRef.current || initialRunId || newRunId();
      runIdRef.current = activeRunId;
      setRunId(activeRunId);

      try {
        const precomputedPayload = uploadId ? await loadPrecomputedKeyValues(activeRunId) : null;

        if (precomputedPayload) {
          if (!cancelled) {
            applyKeyValuePayload(precomputedPayload, activeRunId);
          }
          return;
        }

        const response = await fetch("/api/kv/key-values", {
          method: "POST",
          headers: {
            "content-type": "application/json",
          },
          body: JSON.stringify({ imagePath: queryPath, uploadId }),
        });
        const payload = (await response.json()) as KeyValueResponse | { detail?: string };

        if (!response.ok) {
          throw new Error("detail" in payload ? payload.detail ?? "Failed to load key-values" : "Failed to load key-values");
        }

        if (cancelled) {
          return;
        }

        applyKeyValuePayload("key_values" in payload ? payload : { key_values: [] }, activeRunId);
      } catch (loadError) {
        if (cancelled) {
          return;
        }

        const message = loadError instanceof Error ? loadError.message : "Failed to load key-values";
        setError(message);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void loadKeyValues();

    return () => {
      cancelled = true;
    };
  }, [fileName, initialRunId, queryPath, templateId, uploadId]);

  const overlayPairs = useMemo<KeyValueOverlayPair[]>(
    () =>
      pairs.map((item, index) => ({
        ...item,
        pairId: pairIdFor(item, index),
      })),
    [pairs],
  );

  const valueOptions = useMemo(() => {
    const seen = new Set<string>();

    return overlayPairs.filter((item) => {
      const key = item.value.trim().toLowerCase();
      if (!key || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, [overlayPairs]);

  const keySuggestions = useMemo(() => {
    const seen = new Set<string>();

    return overlayPairs.filter((item) => {
      const key = item.key.trim().toLowerCase();
      if (!key || seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
  }, [overlayPairs]);

  const similarityScore = Math.max(0, Math.min(100, Math.round(score * 100)));
  const displayTemplateName = templateName || templateId;
  const sourceImageUrl = queryPath ? imageUrl(queryPath) : uploadImageUrl(uploadId);

  function updateRow(id: string, patch: Partial<MappingRow>) {
    const nextRows = rowsRef.current.map((row) => (row.id === id ? { ...row, ...patch } : row));
    rowsRef.current = nextRows;
    setRows(nextRows);
    setKvpJsonText(JSON.stringify(buildReviewPayload(runId, templateId, fileName, rowsToReviewPairs(nextRows)), null, 2));
    setJsonError(null);
    setStorageResult(null);
  }

  function addRow(prefill = "", key_bbox?: number[]) {
    const row = makeRow(prefill, "");
    row.key_bbox = key_bbox;
    const nextRows = [...rowsRef.current, row];
    rowsRef.current = nextRows;
    setRows(nextRows);
    setKvpJsonText(JSON.stringify(buildReviewPayload(runId, templateId, fileName, rowsToReviewPairs(nextRows)), null, 2));
    setJsonError(null);
    setStorageResult(null);
  }

  function removeRow(id: string) {
    const nextRows = rowsRef.current.length > 1 ? rowsRef.current.filter((row) => row.id !== id) : rowsRef.current;
    rowsRef.current = nextRows;
    setRows(nextRows);
    setKvpJsonText(JSON.stringify(buildReviewPayload(runId, templateId, fileName, rowsToReviewPairs(nextRows)), null, 2));
    setJsonError(null);
    setStorageResult(null);
  }

  async function storeToMinio() {
    setJsonError(null);
    setError(null);
    setStorageResult(null);

    try {
      JSON.parse(kvpJsonText);
    } catch (parseError) {
      const message = parseError instanceof Error ? parseError.message : "Invalid JSON";
      setJsonError(message);
      return;
    }

    const formData = new FormData();
    formData.append("run_id", runId);
    formData.append("template_id", templateId || "unknown");
    formData.append("kvp_json", kvpJsonText);
    if (uploadId) {
      formData.append("upload_id", uploadId);
    } else if (queryPath) {
      formData.append("image_path", queryPath);
    }

    setStoring(true);
    try {
      const response = await fetch("/api/storage/ingest", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to store to MinIO");
      }

      setStorageResult(payload);
    } catch (storeError) {
      const message = storeError instanceof Error ? storeError.message : "Failed to store to MinIO";
      setError(message);
    } finally {
      setStoring(false);
    }
  }

  if (!queryPath && !uploadId) {
    return (
      <div className="flex min-h-[620px] flex-col items-center justify-center rounded-xl border border-dashed border-[var(--line)] bg-[var(--surface)] px-8 text-center shadow-[var(--panel-shadow)]">
        <div className="flex size-16 items-center justify-center rounded-2xl bg-[var(--accent-soft)] text-[var(--accent)]">
          <FileSearch size={28} />
        </div>
        <h3 className="mt-6 font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">No image selected</h3>
        <p className="mt-3 max-w-md text-sm leading-6 text-[var(--muted)]">
          Start from Upload & Match, pick the best template, and continue into extraction.
        </p>
        <Link
          href="/upload"
          className="mt-8 inline-flex h-11 items-center justify-center rounded-lg bg-[var(--accent)] px-5 text-sm font-semibold text-[var(--background)] shadow-sm transition hover:bg-[var(--accent-strong)]"
        >
          Go to Upload & Match
        </Link>
      </div>
    );
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.25fr)_420px]">
      <section className="min-w-0 rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)]">
        <div className="flex flex-col gap-4 border-b border-[var(--line)] pb-4 md:flex-row md:items-center md:justify-between">
          <div>
            <h3 className="font-display text-xl font-semibold tracking-tight text-[var(--ink-strong)]">{displayTemplateName}</h3>
            <p className="mt-1 font-data text-sm text-[var(--muted)]">
              {similarityScore}% similarity · {pairs.length} extracted pairs
            </p>
          </div>
        </div>

        <div className="mt-5">
          <div className="overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface-subtle)]">
            <KeyValueOverlayViewer
              activePairId={activePairId}
              imageSrc={sourceImageUrl}
              loading={loading}
              onActivePairChange={setActivePairId}
              onPinnedPairChange={setPinnedPairId}
              pairs={overlayPairs}
              pinnedPairId={pinnedPairId}
            />
          </div>
        </div>
      </section>

      <aside className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)]">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">Field Mapping</h3>
          </div>
          <button
            type="button"
            onClick={() => void storeToMinio()}
            disabled={loading || storing || !kvpJsonText.trim()}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-4 text-sm font-semibold text-[var(--background)] shadow-sm transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-[var(--line-strong)]"
          >
            {storing ? <Loader2 size={17} className="animate-spin" /> : <Save size={17} />}
            <span>{storing ? "Saving..." : "Save mapping"}</span>
          </button>
        </div>

        {error ? (
          <div className="mt-5 rounded-lg border border-[var(--accent-line)] bg-[var(--danger-soft)] px-4 py-3 text-sm text-[var(--danger-text)]">
            {error}
          </div>
        ) : null}

        {loading ? (
          <div className="mt-6 flex min-h-[420px] items-center justify-center rounded-xl border border-dashed border-[var(--line)] bg-[var(--surface-subtle)]">
            <Loader2 size={22} className="animate-spin text-[var(--accent)]" />
          </div>
        ) : (
          <>
            <div className="mt-6 flex flex-wrap gap-2">
              {keySuggestions.slice(0, 8).map((item) => (
                <button
                  key={`${item.key}-${item.key_bbox.join("-")}`}
                  type="button"
                  onClick={() => addRow(item.key, item.key_bbox)}
                  className="inline-flex items-center rounded-full border border-[var(--line)] bg-[var(--surface-subtle)] px-3 py-2 text-xs font-medium text-[var(--ink-strong)] transition hover:border-[var(--accent)] hover:text-[var(--accent-strong)]"
                >
                  <Plus size={13} className="mr-1.5" />
                  {item.key}
                </button>
              ))}
            </div>

            <div className="mt-6 space-y-4">
              {rows.map((row) => (
                <div key={row.id} className="rounded-xl border border-[var(--line)] bg-[var(--surface-subtle)] p-4">
                  <div className="flex items-start justify-between gap-3">
                    <div className="min-w-0 flex-1">
                      <label className="mb-2 block font-data text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
                        Key
                      </label>
                      <input
                        value={row.key}
                        onChange={(event) => updateRow(row.id, { key: event.target.value })}
                        placeholder="Field name"
                        className="h-11 w-full rounded-lg border border-[var(--line)] bg-[var(--surface-elevated)] px-3 text-sm text-[var(--ink-strong)] outline-none focus:border-[var(--accent)]"
                      />
                    </div>
                    <button
                      type="button"
                      onClick={() => removeRow(row.id)}
                      className="mt-7 inline-flex size-10 items-center justify-center rounded-lg border border-[var(--accent-line)] bg-[var(--surface-elevated)] text-[var(--danger-text)] transition hover:bg-[var(--danger-soft)]"
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>

                  <div className="mt-4">
                    <label className="mb-2 block font-data text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--muted)]">
                      Value
                    </label>
                    <div className="grid gap-2 md:grid-cols-[128px_minmax(0,1fr)]">
                      <div className="relative">
                        <select
                          value={row.valueMode}
                          onChange={(event) => {
                            const nextMode = event.target.value as MappingRow["valueMode"];
                            if (nextMode === "custom") {
                              updateRow(row.id, {
                                score: undefined,
                                valueMode: "custom",
                                value_bbox: undefined,
                              });
                              return;
                            }

                            const matchedPair = pairs.find((item) => item.value === row.value);
                            updateRow(row.id, {
                              key_bbox: matchedPair?.key_bbox ?? row.key_bbox,
                              score: matchedPair?.score,
                              valueMode: "detected",
                              value_bbox: matchedPair?.value_bbox,
                            });
                          }}
                          className="h-11 w-full appearance-none rounded-lg border border-[var(--line)] bg-[var(--surface-elevated)] px-3 pr-9 text-sm text-[var(--ink-strong)] outline-none focus:border-[var(--accent)]"
                        >
                          <option value="detected">Detected</option>
                          <option value="custom">Custom</option>
                        </select>
                        <ChevronDown
                          size={16}
                          className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[var(--muted)]"
                        />
                      </div>

                      {row.valueMode === "custom" ? (
                        <input
                          value={row.value}
                          onChange={(event) =>
                            updateRow(row.id, {
                              score: undefined,
                              value: event.target.value,
                              value_bbox: undefined,
                            })
                          }
                          placeholder="Enter custom value"
                          className="h-11 w-full rounded-lg border border-[var(--line)] bg-[var(--surface-elevated)] px-3 text-sm text-[var(--ink-strong)] outline-none focus:border-[var(--accent)]"
                        />
                      ) : (
                        <div className="relative">
                          <select
                            value={row.value}
                            onChange={(event) => {
                              const nextValue = event.target.value;
                              const matchedPair = pairs.find((item) => item.value === nextValue);
                              updateRow(row.id, {
                                key_bbox: matchedPair?.key_bbox ?? row.key_bbox,
                                score: matchedPair?.score,
                                value: nextValue,
                                value_bbox: matchedPair?.value_bbox,
                              });
                            }}
                            className="h-11 w-full appearance-none rounded-lg border border-[var(--line)] bg-[var(--surface-elevated)] px-3 pr-10 text-sm text-[var(--ink-strong)] outline-none focus:border-[var(--accent)]"
                          >
                            <option value="">Select identified value</option>
                            {valueOptions.map((option) => (
                              <option
                                key={`${option.value}-${option.value_bbox.join("-")}`}
                                value={option.value}
                              >
                                {option.value}
                              </option>
                            ))}
                          </select>
                          <ChevronDown
                            size={16}
                            className="pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 text-[var(--muted)]"
                          />
                        </div>
                      )}
                    </div>
                  </div>

                  {row.valueMode === "custom" ? (
                    <p className="mt-3 text-xs text-[var(--muted)]">Custom value will be saved without a detected bbox.</p>
                  ) : typeof row.score === "number" ? (
                    <p className="mt-3 text-xs text-[var(--muted)]">
                      Pair confidence: {Math.round(row.score * 100)}%
                    </p>
                  ) : null}
                </div>
              ))}
            </div>

            <button
              type="button"
              onClick={() => addRow()}
              className="mt-5 inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-4 text-sm font-medium text-[var(--ink-strong)] transition hover:border-[var(--accent)]"
            >
              <Plus size={17} />
              <span>Add Field</span>
            </button>

            <div className="mt-6 rounded-xl border border-[var(--line)] bg-[var(--surface-subtle)] p-4">
              <label className="mb-2 block font-data text-[11px] font-semibold uppercase tracking-[0.16em] text-[var(--accent)]">
                KVP JSON
              </label>
              <textarea
                value={kvpJsonText}
                onChange={(event) => {
                  setKvpJsonText(event.target.value);
                  setJsonError(null);
                  setStorageResult(null);
                }}
                className="min-h-[260px] w-full resize-y rounded-lg border border-[var(--line)] bg-[var(--surface-elevated)] px-3 py-3 font-data text-xs leading-5 text-[var(--ink-strong)] outline-none focus:border-[var(--accent)]"
                placeholder="Extract key-values to populate editable JSON."
              />
              {jsonError ? (
                <p className="mt-2 text-sm text-[var(--danger-text)]">Invalid JSON: {jsonError}</p>
              ) : null}
            </div>

            {storageResult ? (
              <div className="mt-5 rounded-xl border border-[var(--accent-line)] bg-[var(--success-soft)] px-4 py-3 text-sm text-[var(--text-main)]">
                <p className="font-semibold">Saved</p>
                <pre className="mt-2 overflow-x-auto whitespace-pre-wrap font-data text-xs">
                  {JSON.stringify(storageResult, null, 2)}
                </pre>
              </div>
            ) : null}

            {!pairs.length ? (
              <div className="mt-5 flex gap-3 rounded-xl border border-[var(--accent-line)] bg-[var(--warning-soft)] px-4 py-3 text-sm text-[var(--text-main)]">
                <AlertCircle size={18} className="mt-0.5 shrink-0" />
                <p>No key-value pairs were returned yet. Try another file or revisit the template match.</p>
              </div>
            ) : null}
          </>
        )}
      </aside>
    </div>
  );
}
