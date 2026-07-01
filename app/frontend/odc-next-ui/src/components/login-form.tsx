"use client";

import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  CheckCircle2,
  FileJson,
  Loader2,
  Lock,
  LogIn,
  ScanLine,
  User,
} from "lucide-react";
import { useState } from "react";

export function LoginForm() {
  const searchParams = useSearchParams();
  const next = searchParams.get("next") || "/upload";

  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        setError(typeof data?.detail === "string" ? data.detail : "Unable to sign in.");
        setSubmitting(false);
        return;
      }

      // Use a hard navigation so middleware re-evaluates with the new cookie.
      const target = next.startsWith("/") ? next : "/upload";
      window.location.assign(target);
    } catch {
      setError("Network error. Please try again.");
      setSubmitting(false);
    }
  }

  return (
    <div className="grid min-h-screen bg-[var(--surface-subtle)] text-[var(--ink)] lg:grid-cols-[1.05fr_0.95fr]">
      {/* Brand / atmosphere panel */}
      <div className="relative hidden overflow-hidden border-r border-[var(--line)] bg-[var(--surface)] lg:flex lg:flex-col lg:justify-between lg:p-12">
        <div className="atmosphere pointer-events-none absolute inset-0" />
        <div className="grid-field pointer-events-none absolute inset-0" />

        <Link href="/" className="relative flex items-center gap-2.5">
          <span className="font-display text-lg font-semibold tracking-tight text-[var(--ink-strong)]">
            UniForm
          </span>
          <span className="rounded-full border border-[var(--line-strong)] bg-[var(--accent-soft)] px-2 py-0.5 font-data text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent-strong)]">
            Beta
          </span>
        </Link>

        <div className="relative max-w-md">
          <h2 className="font-display text-4xl font-semibold leading-[1.1] tracking-tight text-[var(--ink-strong)]">
            From scan to{" "}
            <span className="text-[var(--accent)]">structured data</span>.
          </h2>
          <div className="mt-8 rounded-2xl border border-[var(--line-strong)] bg-[color-mix(in_srgb,var(--surface-subtle)_88%,transparent)] p-4 shadow-[var(--panel-shadow)] backdrop-blur">
            <div className="flex items-center justify-between border-b border-[var(--line)] pb-2.5">
              <span className="flex items-center gap-2 font-data text-xs font-semibold text-[var(--ink-strong)]">
                <ScanLine size={14} className="text-[var(--accent)]" />
                claim_2207.png
              </span>
              <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--success-soft)] px-2 py-0.5 text-[10px] font-semibold text-[var(--accent-strong)]">
                <span className="size-1.5 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--glow-strong)]" />
                Recognized
              </span>
            </div>
            <div className="mt-3 flex items-center gap-2 rounded-lg border border-[var(--line-strong)] bg-[var(--accent-soft)] px-3 py-2 font-data text-[11px] font-medium text-[var(--accent-strong)]">
              <FileJson size={13} />
              Exported as structured JSON
            </div>
          </div>
          <div className="mt-6 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm text-[var(--muted)]">
            <span className="inline-flex items-center gap-2">
              <CheckCircle2 size={15} className="text-[var(--accent)]" />
              Human-verified
            </span>
            <span className="inline-flex items-center gap-2">
              <CheckCircle2 size={15} className="text-[var(--accent)]" />
              Private by design
            </span>
          </div>
        </div>

        <p className="relative font-data text-xs text-[var(--muted)]">
          Unified form ingestion · UniForm Beta
        </p>
      </div>

      {/* Form panel */}
      <div className="relative flex items-center justify-center px-5 py-12">
        <div className="atmosphere pointer-events-none absolute inset-0 lg:hidden" />
        <div className="relative w-full max-w-md">
          <Link href="/" className="mb-8 flex items-center justify-center gap-2.5 lg:hidden">
            <span className="font-display text-lg font-semibold tracking-tight text-[var(--ink-strong)]">
              UniForm
            </span>
          </Link>

          <div className="rounded-2xl border border-[var(--line-strong)] bg-[var(--surface)] p-7 shadow-[var(--panel-shadow)] sm:p-8">
            <h1 className="font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)]">
              Sign in
            </h1>
            <p className="mt-1.5 text-sm text-[var(--muted)]">
              Enter your credentials to access the workspace.
            </p>

            <form onSubmit={handleSubmit} className="mt-7 space-y-4">
            <label className="block">
              <span className="mb-1.5 block text-sm font-medium text-[var(--ink-strong)]">
                Username
              </span>
              <div className="flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 focus-within:border-[var(--accent)]">
                <User size={17} className="shrink-0 text-[var(--muted)]" />
                <input
                  type="text"
                  autoComplete="username"
                  autoFocus
                  required
                  value={username}
                  onChange={(event) => setUsername(event.target.value)}
                  className="h-11 w-full bg-transparent text-sm text-[var(--ink-strong)] outline-none placeholder:text-[var(--muted)]"
                  placeholder="admin"
                />
              </div>
            </label>

            <label className="block">
              <span className="mb-1.5 block text-sm font-medium text-[var(--ink-strong)]">
                Password
              </span>
              <div className="flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 focus-within:border-[var(--accent)]">
                <Lock size={17} className="shrink-0 text-[var(--muted)]" />
                <input
                  type="password"
                  autoComplete="current-password"
                  required
                  value={password}
                  onChange={(event) => setPassword(event.target.value)}
                  className="h-11 w-full bg-transparent text-sm text-[var(--ink-strong)] outline-none placeholder:text-[var(--muted)]"
                  placeholder="••••••••"
                />
              </div>
            </label>

            {error ? (
              <p className="rounded-lg border border-[var(--line)] bg-[var(--danger-soft)] px-3 py-2.5 text-sm text-[var(--danger-text)]">
                {error}
              </p>
            ) : null}

            <button
              type="submit"
              disabled={submitting}
              className="inline-flex h-11 w-full items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-4 text-sm font-semibold text-[var(--background)] shadow-sm transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-70"
            >
              {submitting ? (
                <>
                  <Loader2 size={17} className="animate-spin" />
                  Signing in…
                </>
              ) : (
                <>
                  <LogIn size={17} />
                  Sign in
                </>
              )}
            </button>
            </form>
          </div>

          <p className="mt-6 text-center text-xs text-[var(--muted)]">
            Access is limited to authorized accounts.
          </p>
        </div>
      </div>
    </div>
  );
}
