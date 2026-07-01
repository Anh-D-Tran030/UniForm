"use client";

import { LogOut } from "lucide-react";
import { useState } from "react";

export function LogoutButton() {
  const [busy, setBusy] = useState(false);

  async function handleLogout() {
    setBusy(true);
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch {
      // Ignore network errors — we still send the user to the login page.
    }
    window.location.assign("/login");
  }

  return (
    <button
      type="button"
      onClick={handleLogout}
      disabled={busy}
      className="flex h-11 w-full items-center gap-3 rounded-lg px-3 text-sm font-medium text-[var(--muted)] transition hover:bg-[var(--surface-subtle)] hover:text-[var(--ink-strong)] disabled:opacity-60"
    >
      <LogOut size={18} />
      <span>{busy ? "Signing out…" : "Sign out"}</span>
    </button>
  );
}
