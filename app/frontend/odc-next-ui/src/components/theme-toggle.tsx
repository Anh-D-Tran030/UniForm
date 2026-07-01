"use client";

import { SunMoon } from "lucide-react";

const storageKey = "uniform-theme";

export function ThemeToggle() {
  function toggleTheme() {
    const currentTheme = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    const nextTheme = currentTheme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = nextTheme;
    window.localStorage.setItem(storageKey, nextTheme);
  }

  return (
    <button
      type="button"
      onClick={toggleTheme}
      aria-label="Toggle light and dark mode"
      className="inline-flex h-11 items-center gap-2 rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 text-sm font-medium text-[var(--ink-strong)] transition hover:border-[var(--accent)]"
    >
      <SunMoon size={17} />
      <span>Theme</span>
    </button>
  );
}
