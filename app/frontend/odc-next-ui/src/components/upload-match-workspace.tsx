"use client";

import { useRouter } from "next/navigation";
import { CloudUpload, Expand, FileText, Loader2, Search } from "lucide-react";
import { type ChangeEvent, type DragEvent, useEffect, useState } from "react";

type Match = {
  cosine_similarity: number;
  display_name: string | null;
  id: number;
  image_path: string | null;
  template_id: string;
  word_count: string;
};

type QueryResponse = {
  matches: Match[];
  ok: boolean;
  precompute_status?: string;
  query_event_id: string;
  query_image_path: string;
  query_word_count: number;
  run_id?: string;
  upload_id: string;
  uploaded_file_name: string;
  uploaded_mime_type: string;
  uploaded_size: number;
};

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;
const ACCEPTED_IMAGE_TYPES = new Set(["image/jpeg", "image/png", "image/tiff"]);
const ACCEPTED_IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg", ".tif", ".tiff"];

function isAcceptedImage(file: File) {
  const lowerName = file.name.toLowerCase();
  return (
    ACCEPTED_IMAGE_TYPES.has(file.type) ||
    ACCEPTED_IMAGE_EXTENSIONS.some((extension) => lowerName.endsWith(extension))
  );
}

function imageUrl(imagePath: string | null) {
  if (!imagePath) {
    return null;
  }

  return `/api/image?path=${encodeURIComponent(imagePath)}`;
}

export function UploadMatchWorkspace() {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [loading, setLoading] = useState(false);
  const [topK, setTopK] = useState(5);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [queryPreview, setQueryPreview] = useState<string | null>(null);
  const [selectedTemplateId, setSelectedTemplateId] = useState<string | null>(null);
  const [expandedMatch, setExpandedMatch] = useState<Match | null>(null);

  useEffect(() => {
    return () => {
      if (queryPreview) {
        URL.revokeObjectURL(queryPreview);
      }
    };
  }, [queryPreview]);

  function handleFile(selectedFile: File | null) {
    setError(null);

    if (selectedFile && !isAcceptedImage(selectedFile)) {
      setFile(null);
      setResult(null);
      setSelectedTemplateId(null);
      setError("Unsupported file type. Upload a PNG, JPG, JPEG, TIF, or TIFF image.");
      return;
    }

    if (selectedFile && selectedFile.size > MAX_UPLOAD_BYTES) {
      setFile(null);
      setResult(null);
      setSelectedTemplateId(null);
      setError("File is too large. Upload an image up to 50MB.");
      return;
    }

    setQueryPreview((currentPreview) => {
      if (currentPreview) {
        URL.revokeObjectURL(currentPreview);
      }

      if (!selectedFile) {
        return null;
      }

      return URL.createObjectURL(selectedFile);
    });
    setFile(selectedFile);
    setResult(null);
    setSelectedTemplateId(null);
  }

  function onInputChange(event: ChangeEvent<HTMLInputElement>) {
    handleFile(event.target.files?.[0] ?? null);
  }

  function onDrop(event: DragEvent<HTMLLabelElement>) {
    event.preventDefault();
    setDragActive(false);
    handleFile(event.dataTransfer.files?.[0] ?? null);
  }

  async function handleSubmit() {
    if (!file) {
      return;
    }

    setLoading(true);
    setError(null);

    const formData = new FormData();
    formData.append("image", file);

    try {
      const response = await fetch(`/api/query?top_k=${topK}`, {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to query templates");
      }

      const queryResult = payload as QueryResponse;
      setResult(queryResult);
      setSelectedTemplateId(queryResult.matches[0]?.template_id ?? null);
    } catch (submitError) {
      const message =
        submitError instanceof Error ? submitError.message : "Failed to query templates";
      setError(message);
    } finally {
      setLoading(false);
    }
  }

  const selectedMatch =
    result?.matches.find((match) => match.template_id === selectedTemplateId) ?? result?.matches[0] ?? null;

  async function continueWithSelectedMatch() {
    if (!result?.query_image_path || !selectedMatch) {
      return;
    }

    const selectedRank = Math.max(
      1,
      result.matches.findIndex((match) => match.template_id === selectedMatch.template_id) + 1,
    );

    await fetch("/api/metrics/selection", {
      method: "POST",
      headers: {
        "content-type": "application/json",
      },
      body: JSON.stringify({
        query_event_id: result.query_event_id,
        selected_rank: selectedRank,
        selected_template_id: selectedMatch.template_id,
      }),
    }).catch(() => null);

    const params = new URLSearchParams({
      queryPath: result.query_image_path,
      templateId: selectedMatch.template_id,
      templateName: selectedMatch.display_name || selectedMatch.template_id,
      templateImagePath: selectedMatch.image_path || "",
      score: String(selectedMatch.cosine_similarity),
      uploadId: result.upload_id,
      fileName: result.uploaded_file_name,
    });
    if (result.run_id) {
      params.set("runId", result.run_id);
    }

    router.push(`/extraction?${params.toString()}`);
  }

  return (
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_380px]">
      <section className="min-w-0">
        <div className="rounded-xl border border-dashed border-[var(--line-strong)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)] sm:p-6">
          <div className="flex flex-col gap-5 lg:flex-row lg:items-start lg:justify-between">
            <div>
              <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
                Upload Source Images
              </p>
              <h3 className="mt-2 font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">
                Drop files to workspace
              </h3>
              <p className="mt-2 max-w-xl text-sm leading-6 text-[var(--muted)]">
                PNG, JPG, and TIFF files up to 50MB per artifact.
              </p>
            </div>

            <label className="flex items-center gap-2 text-sm font-medium text-[var(--muted)]">
              <span>Matches</span>
              <select
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value))}
                className="h-10 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 text-[var(--ink-strong)] outline-none"
              >
                {[3, 5, 10].map((value) => (
                  <option key={value} value={value}>
                    Top {value}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <label
            onDragOver={(event) => {
              event.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={() => setDragActive(false)}
            onDrop={onDrop}
            className={`mt-6 flex min-h-[420px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed px-6 py-10 text-center transition ${
              dragActive
                ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                : "border-[var(--line)] bg-[var(--surface-subtle)]"
            }`}
          >
            <input
              type="file"
              accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
              onChange={onInputChange}
              className="hidden"
            />
            <div className="flex size-16 items-center justify-center rounded-2xl border border-[var(--line-strong)] bg-[var(--surface-elevated)] text-[var(--accent)] shadow-[0_0_24px_-6px_var(--glow-strong)]">
              <CloudUpload size={28} />
            </div>
            <h4 className="mt-6 font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">
              {file ? file.name : "Select forms for ingestion"}
            </h4>
            <p className="mt-3 max-w-md text-sm leading-6 text-[var(--muted)]">
              Drag and drop an image here or browse from local storage.
            </p>

            <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
              <span className="inline-flex h-11 items-center rounded-lg bg-[var(--accent)] px-5 text-sm font-semibold text-[var(--background)] shadow-sm">
                Browse Local Files
              </span>
              <span className="font-data text-sm font-medium uppercase tracking-[0.18em] text-[var(--muted-light)]">
                Or drop here
              </span>
            </div>

            {queryPreview ? (
              <div className="mt-8 w-full max-w-[540px] overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface-elevated)]">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  alt="Query preview"
                  src={queryPreview}
                  className="h-[240px] w-full object-contain"
                />
              </div>
            ) : null}
          </label>

          <div className="mt-6 flex flex-col gap-4 border-t border-[var(--line)] pt-5 md:flex-row md:items-center md:justify-between">
            <div className="grid gap-3 text-sm text-[var(--muted)] sm:grid-cols-3 sm:gap-6">
              <div>
                <p className="font-data text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted-light)]">
                  Supported Formats
                </p>
                <p className="mt-2 font-medium text-[var(--ink-strong)]">PNG, JPG, TIFF</p>
              </div>
              <div>
                <p className="font-data text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted-light)]">
                  Match Mode
                </p>
                <p className="mt-2 font-medium text-[var(--ink-strong)]">Template retrieval</p>
              </div>
              <div>
                <p className="font-data text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted-light)]">
                  Word Count
                </p>
                <p className="mt-2 font-medium text-[var(--ink-strong)]">
                  {result?.query_word_count ?? "--"}
                </p>
              </div>
            </div>

            <button
              type="button"
              onClick={handleSubmit}
              disabled={!file || loading}
              className="inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-5 text-sm font-semibold text-[var(--background)] shadow-sm transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-[var(--line-strong)]"
            >
              {loading ? <Loader2 size={18} className="animate-spin" /> : <Search size={18} />}
              <span>{loading ? "Matching..." : "Run Match"}</span>
            </button>
          </div>

          {error ? (
            <div className="mt-5 rounded-lg border border-[var(--accent-line)] bg-[var(--danger-soft)] px-4 py-3 text-sm text-[var(--danger-text)]">
              {error}
            </div>
          ) : null}
        </div>
      </section>

      <aside className="flex min-h-[640px] flex-col rounded-xl border border-[var(--line)] bg-[var(--surface)] p-4 shadow-[var(--panel-shadow)]">
        <div>
          <div>
            <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
              Matching Templates
            </p>
            <h3 className="mt-2 font-display text-xl font-semibold tracking-tight text-[var(--ink-strong)]">
              {result ? `${result.matches.length} found` : "Waiting for image"}
            </h3>
          </div>
        </div>

        <div className="mt-5 flex-1 space-y-3 overflow-y-auto pr-1">
          {result?.matches?.length ? (
            result.matches.map((match) => {
              const score = Math.max(0, Math.min(100, Math.round(match.cosine_similarity * 100)));
              const label = match.display_name || match.template_id;
              const isSelected = match.template_id === selectedMatch?.template_id;

              return (
                <div
                  key={`${match.id}-${match.template_id}`}
                  onClick={() => setSelectedTemplateId(match.template_id)}
                  className={`cursor-pointer rounded-xl border p-3 transition ${
                    isSelected
                      ? "border-[var(--accent)] bg-[var(--accent-soft)]"
                      : "border-[var(--line)] bg-[var(--surface-subtle)]"
                  }`}
                >
                  <div className="flex gap-3">
                    <div className="flex h-16 w-14 shrink-0 items-center justify-center overflow-hidden rounded-lg border border-[var(--line)] bg-[var(--surface-elevated)]">
                      {match.image_path ? (
                        // eslint-disable-next-line @next/next/no-img-element
                        <img
                          alt={label}
                          src={imageUrl(match.image_path) ?? ""}
                          className="h-full w-full object-cover"
                        />
                      ) : (
                        <FileText size={18} className="text-[var(--muted)]" />
                      )}
                    </div>
                    <div className="min-w-0 flex-1">
                      <div className="flex items-start justify-between gap-3">
                        <div className="min-w-0">
                          <p className="truncate text-sm font-semibold text-[var(--ink-strong)]">
                            {label}
                          </p>
                          <p className="mt-1 font-data text-[11px] font-semibold uppercase tracking-[0.14em] text-[var(--muted-light)]">
                            {match.template_id}
                          </p>
                        </div>
                        <div className="flex items-center gap-2">
                          <button
                            type="button"
                            onClick={(event) => {
                              event.stopPropagation();
                              setExpandedMatch(match);
                            }}
                            className="inline-flex size-8 items-center justify-center rounded-full border border-[var(--line)] bg-[var(--surface-elevated)] text-[var(--muted)] transition hover:text-[var(--accent-strong)]"
                          >
                            <Expand size={14} />
                          </button>
                          <span className="rounded-full border border-[var(--line-strong)] bg-[var(--surface-elevated)] px-2 py-1 font-data text-[11px] font-semibold text-[var(--accent-strong)]">
                            {score}%
                          </span>
                        </div>
                      </div>
                      <div className="mt-4 h-2 overflow-hidden rounded-full bg-[var(--surface-elevated)]">
                        <div
                          className="h-full rounded-full bg-[linear-gradient(90deg,var(--accent),var(--accent-strong))] shadow-[0_0_10px_-1px_var(--glow-strong)]"
                          style={{ width: `${score}%` }}
                        />
                      </div>
                      <p className="mt-3 text-xs text-[var(--muted)]">
                        {match.word_count} OCR tokens
                      </p>
                    </div>
                  </div>
                </div>
              );
            })
          ) : (
            <div className="flex h-full min-h-[280px] flex-col items-center justify-center rounded-xl border border-dashed border-[var(--line)] bg-[var(--surface-subtle)] px-6 text-center">
              <div className="flex size-14 items-center justify-center rounded-2xl bg-[var(--surface-elevated)] text-[var(--accent)] shadow-sm">
                <FileText size={24} />
              </div>
              <h4 className="mt-5 text-lg font-semibold text-[var(--ink-strong)]">
                No matches yet
              </h4>
              <p className="mt-2 max-w-xs text-sm leading-6 text-[var(--muted)]">
                Upload an image and run a match to inspect the nearest templates.
              </p>
            </div>
          )}
        </div>

        <div className="mt-5 border-t border-[var(--line)] pt-4">
          <button
            type="button"
            onClick={() => void continueWithSelectedMatch()}
            disabled={!selectedMatch || !result}
            className="inline-flex h-12 w-full items-center justify-center rounded-lg bg-[var(--accent)] px-4 text-sm font-semibold text-[var(--background)] shadow-sm transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-[var(--line-strong)]"
          >
            {selectedMatch ? `Continue with ${selectedMatch.display_name || selectedMatch.template_id}` : "Continue"}
          </button>
        </div>
      </aside>

      {expandedMatch ? (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-950/55 px-5 py-8">
          <div className="w-full max-w-4xl rounded-2xl bg-[var(--surface)] p-5 shadow-2xl">
            <div className="flex items-center justify-between gap-4 border-b border-[var(--line)] pb-4">
              <div>
                <p className="font-display text-lg font-semibold tracking-tight text-[var(--ink-strong)]">
                  {expandedMatch.display_name || expandedMatch.template_id}
                </p>
                <p className="mt-1 font-data text-sm text-[var(--muted)]">{expandedMatch.template_id}</p>
              </div>
              <button
                type="button"
                onClick={() => setExpandedMatch(null)}
                className="inline-flex h-10 items-center justify-center rounded-lg border border-[var(--line)] px-4 text-sm font-medium text-[var(--ink-strong)]"
              >
                Close
              </button>
            </div>

            <div className="mt-5 flex h-[70vh] items-center justify-center overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface-subtle)] p-4">
              {expandedMatch.image_path ? (
                // eslint-disable-next-line @next/next/no-img-element
                <img
                  alt={expandedMatch.display_name || expandedMatch.template_id}
                  src={imageUrl(expandedMatch.image_path) ?? ""}
                  className="h-full w-full object-contain"
                />
              ) : (
                <div className="text-sm text-[var(--muted)]">No image preview available</div>
              )}
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
