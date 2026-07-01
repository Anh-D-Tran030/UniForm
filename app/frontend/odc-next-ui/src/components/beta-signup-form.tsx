"use client";

import { CheckCircle2, Loader2, Mail, Send, User } from "lucide-react";
import { useState } from "react";

export function BetaSignupForm() {
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    setError(null);
    setSubmitting(true);

    try {
      const response = await fetch("/api/access/request", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name, email }),
      });
      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        setError(typeof data?.detail === "string" ? data.detail : "Something went wrong.");
        setSubmitting(false);
        return;
      }

      setDone(true);
    } catch {
      setError("Network error. Please try again.");
      setSubmitting(false);
    }
  }

  if (done) {
    return (
      <div className="flex flex-col items-center gap-3 rounded-2xl border border-[var(--line)] bg-[var(--surface-subtle)] px-6 py-10 text-center">
        <div className="flex size-12 items-center justify-center rounded-full bg-[var(--success-soft)] text-[var(--accent-strong)]">
          <CheckCircle2 size={24} />
        </div>
        <h3 className="font-display text-xl font-semibold tracking-tight text-[var(--ink-strong)]">You&apos;re on the list</h3>
        <p className="max-w-sm text-sm text-[var(--muted)]">
          Thanks, {name.split(" ")[0] || "there"}. We&apos;ll reach out at{" "}
          <span className="font-medium text-[var(--ink-strong)]">{email}</span> when your beta access
          is ready.
        </p>
      </div>
    );
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2">
        <label className="block">
          <span className="mb-1.5 block text-sm font-medium text-[var(--ink-strong)]">
            Full name
          </span>
          <div className="flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 focus-within:border-[var(--accent)]">
            <User size={17} className="shrink-0 text-[var(--muted)]" />
            <input
              type="text"
              required
              value={name}
              onChange={(event) => setName(event.target.value)}
              className="h-11 w-full bg-transparent text-sm text-[var(--ink-strong)] outline-none placeholder:text-[var(--muted)]"
              placeholder="Jane Tran"
            />
          </div>
        </label>

        <label className="block">
          <span className="mb-1.5 block text-sm font-medium text-[var(--ink-strong)]">
            Work email
          </span>
          <div className="flex items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 focus-within:border-[var(--accent)]">
            <Mail size={17} className="shrink-0 text-[var(--muted)]" />
            <input
              type="email"
              required
              value={email}
              onChange={(event) => setEmail(event.target.value)}
              className="h-11 w-full bg-transparent text-sm text-[var(--ink-strong)] outline-none placeholder:text-[var(--muted)]"
              placeholder="jane@company.com"
            />
          </div>
        </label>
      </div>

      {error ? (
        <p className="rounded-lg border border-[var(--line)] bg-[var(--danger-soft)] px-3 py-2.5 text-sm text-[var(--danger-text)]">
          {error}
        </p>
      ) : null}

      <div className="flex flex-col items-start gap-3 sm:flex-row sm:items-center sm:justify-between">
        <button
          type="submit"
          disabled={submitting}
          className="inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-6 text-sm font-semibold text-[var(--background)] shadow-[0_0_28px_-8px_var(--glow-strong)] transition hover:bg-[var(--accent-strong)] disabled:cursor-not-allowed disabled:opacity-70"
        >
          {submitting ? (
            <>
              <Loader2 size={17} className="animate-spin" />
              Submitting…
            </>
          ) : (
            <>
              <Send size={16} />
              Request beta access
            </>
          )}
        </button>
        <p className="text-xs text-[var(--muted)]">
          No spam. We&apos;ll only email you about your access.
        </p>
      </div>
    </form>
  );
}
