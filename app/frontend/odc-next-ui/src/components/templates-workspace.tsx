"use client";

import Image from "next/image";
import { Grid2X2, Loader2, Plus, RefreshCw, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";

type TemplateRecord = {
  display_name: string | null;
  id: number;
  image_path: string | null;
  template_id: string;
  word_count: string;
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

  return `/api/image?path=${encodeURIComponent(imagePath)}&cache=template`;
}

export function TemplatesWorkspace() {
  const [templates, setTemplates] = useState<TemplateRecord[]>([]);
  const [templateId, setTemplateId] = useState("");
  const [displayName, setDisplayName] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [refreshing, setRefreshing] = useState(false);

  async function requestTemplates() {
    const response = await fetch("/api/templates", { cache: "no-store" });
    const payload = await response.json();

    if (!response.ok) {
      throw new Error(payload.detail ?? "Failed to load templates");
    }

    return (payload.templates ?? []) as TemplateRecord[];
  }

  async function loadTemplates(showSpinner = true) {
    if (showSpinner) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }

    try {
      setTemplates(await requestTemplates());
      setError(null);
    } catch (loadError) {
      const message = loadError instanceof Error ? loadError.message : "Failed to load templates";
      setError(message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  }

  useEffect(() => {
    let cancelled = false;

    async function initialize() {
      try {
        const nextTemplates = await requestTemplates();
        if (cancelled) {
          return;
        }

        setTemplates(nextTemplates);
        setError(null);
      } catch (loadError) {
        if (cancelled) {
          return;
        }

        const message =
          loadError instanceof Error ? loadError.message : "Failed to load templates";
        setError(message);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void initialize();

    return () => {
      cancelled = true;
    };
  }, []);

  async function addTemplate() {
    if (!file || !templateId.trim()) {
      return;
    }

    setSaving(true);
    setError(null);

    const formData = new FormData();
    formData.append("template_id", templateId.trim());
    formData.append("display_name", displayName.trim());
    formData.append("image", file);

    try {
      const response = await fetch("/api/embed", {
        method: "POST",
        body: formData,
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to add template");
      }

      setTemplateId("");
      setDisplayName("");
      setFile(null);
      await loadTemplates(false);
    } catch (saveError) {
      const message = saveError instanceof Error ? saveError.message : "Failed to add template";
      setError(message);
    } finally {
      setSaving(false);
    }
  }

  function handleTemplateFile(selectedFile: File | null) {
    setError(null);

    if (selectedFile && !isAcceptedImage(selectedFile)) {
      setFile(null);
      setError("Unsupported file type. Upload a PNG, JPG, JPEG, TIF, or TIFF image.");
      return;
    }

    if (selectedFile && selectedFile.size > MAX_UPLOAD_BYTES) {
      setFile(null);
      setError("File is too large. Upload an image up to 50MB.");
      return;
    }

    setFile(selectedFile);
  }

  async function deleteTemplate(id: string) {
    try {
      const response = await fetch(`/api/templates/${encodeURIComponent(id)}`, {
        method: "DELETE",
      });
      const payload = await response.json();

      if (!response.ok) {
        throw new Error(payload.detail ?? "Failed to delete template");
      }

      setTemplates((current) => current.filter((item) => item.template_id !== id));
    } catch (deleteError) {
      const message =
        deleteError instanceof Error ? deleteError.message : "Failed to delete template";
      setError(message);
    }
  }

  return (
    <div className="grid gap-4 2xl:grid-cols-[380px_minmax(0,1fr)]">
      <section className="rounded-xl border border-[var(--line)] bg-[var(--surface)] p-5 shadow-[var(--panel-shadow)]">
        <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
          Add Template
        </p>
        <h3 className="mt-2 font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">Register a new form</h3>

        <div className="mt-6 space-y-4">
          <div>
            <label className="mb-2 block text-sm font-medium text-[var(--ink-strong)]">
              Template ID
            </label>
            <input
              value={templateId}
              onChange={(event) => setTemplateId(event.target.value)}
              placeholder="t_39"
              className="h-11 w-full rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 text-sm text-[var(--ink-strong)] outline-none placeholder:text-[var(--muted-light)] focus:border-[var(--accent)]"
            />
          </div>
          <div>
            <label className="mb-2 block text-sm font-medium text-[var(--ink-strong)]">
              Display Name
            </label>
            <input
              value={displayName}
              onChange={(event) => setDisplayName(event.target.value)}
              placeholder="Template 39"
              className="h-11 w-full rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 text-sm text-[var(--ink-strong)] outline-none placeholder:text-[var(--muted-light)] focus:border-[var(--accent)]"
            />
          </div>
          <div>
            <label className="mb-2 block text-sm font-medium text-[var(--ink-strong)]">
              Template File
            </label>
            <label className="flex min-h-[180px] cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-[var(--line-strong)] bg-[var(--surface-subtle)] px-4 text-center transition hover:border-[var(--accent)] hover:bg-[var(--accent-soft)]">
              <input
                type="file"
                accept=".png,.jpg,.jpeg,.tif,.tiff,image/png,image/jpeg,image/tiff"
                onChange={(event) => handleTemplateFile(event.target.files?.[0] ?? null)}
                className="hidden"
              />
              <div className="rounded-xl bg-[var(--surface-elevated)] px-4 py-2 text-sm font-medium text-[var(--accent-strong)] shadow-sm">
                {file ? file.name : "Browse Local Files"}
              </div>
            </label>
          </div>
        </div>

        <button
          type="button"
          onClick={addTemplate}
          disabled={!file || !templateId.trim() || saving}
          className="mt-6 inline-flex h-12 w-full items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-4 text-sm font-semibold text-[var(--background)] shadow-sm transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:bg-[var(--line-strong)]"
        >
          {saving ? <Loader2 size={18} className="animate-spin" /> : <Plus size={18} />}
          <span>{saving ? "Saving..." : "Embed Template"}</span>
        </button>

        {error ? (
          <div className="mt-4 rounded-lg border border-[var(--accent-line)] bg-[var(--danger-soft)] px-4 py-3 text-sm text-[var(--danger-text)]">
            {error}
          </div>
        ) : null}
      </section>

      <section className="min-w-0 rounded-xl border border-[var(--line)] bg-[var(--surface)] p-5 shadow-[var(--panel-shadow)]">
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
              Saved Templates
            </p>
            <h3 className="mt-2 font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">
              {templates.length} records
            </h3>
          </div>
          <button
            type="button"
            onClick={() => void loadTemplates(false)}
            disabled={refreshing}
            className="inline-flex h-11 items-center justify-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-4 text-sm font-medium text-[var(--ink-strong)] transition hover:border-[var(--accent)]"
          >
            {refreshing ? <Loader2 size={17} className="animate-spin" /> : <RefreshCw size={17} />}
            <span>Refresh</span>
          </button>
        </div>

        {loading ? (
          <div className="mt-6 flex min-h-[420px] items-center justify-center rounded-xl border border-dashed border-[var(--line)] bg-[var(--surface-subtle)]">
            <Loader2 size={22} className="animate-spin text-[var(--accent)]" />
          </div>
        ) : templates.length ? (
          <div className="mt-6 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
            {templates.map((template) => {
              const preview = imageUrl(template.image_path);
              const label = template.display_name || template.template_id;

              return (
                <article
                  key={template.id}
                  className="overflow-hidden rounded-xl border border-[var(--line)] bg-[var(--surface-subtle)]"
                >
                  <div className="relative flex h-[220px] items-center justify-center bg-[var(--surface-elevated)]">
                    {preview ? (
                      <Image
                        alt={label}
                        src={preview}
                        fill
                        sizes="(max-width: 768px) 100vw, (max-width: 1280px) 50vw, 33vw"
                        className="object-cover"
                        loading="lazy"
                        quality={60}
                      />
                    ) : (
                      <Grid2X2 size={24} className="text-[var(--muted)]" />
                    )}
                  </div>

                  <div className="space-y-4 p-4">
                    <div>
                      <p className="truncate text-base font-semibold text-[var(--ink-strong)]">
                        {label}
                      </p>
                      <p className="mt-1 font-data text-xs font-semibold uppercase tracking-[0.16em] text-[var(--muted-light)]">
                        {template.template_id}
                      </p>
                    </div>

                    <div className="flex items-center justify-between text-sm text-[var(--muted)]">
                      <span>OCR tokens</span>
                      <span className="font-medium text-[var(--ink-strong)]">{template.word_count}</span>
                    </div>

                    <button
                      type="button"
                      onClick={() => void deleteTemplate(template.template_id)}
                      className="inline-flex h-10 w-full items-center justify-center gap-2 rounded-lg border border-[var(--accent-line)] bg-[var(--surface-elevated)] text-sm font-medium text-[var(--danger-text)] transition hover:bg-[var(--danger-soft)]"
                    >
                      <Trash2 size={16} />
                      <span>Delete</span>
                    </button>
                  </div>
                </article>
              );
            })}
          </div>
        ) : (
          <div className="mt-6 flex min-h-[420px] flex-col items-center justify-center rounded-xl border border-dashed border-[var(--line)] bg-[var(--surface-subtle)] px-6 text-center">
            <div className="flex size-14 items-center justify-center rounded-2xl bg-[var(--surface-elevated)] text-[var(--accent)] shadow-sm">
              <Grid2X2 size={24} />
            </div>
            <h4 className="mt-5 text-lg font-semibold text-[var(--ink-strong)]">No templates loaded</h4>
            <p className="mt-2 max-w-xs text-sm leading-6 text-[var(--muted)]">
              Add a template record to make it available during matching.
            </p>
          </div>
        )}
      </section>
    </div>
  );
}
