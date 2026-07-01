import Link from "next/link";
import {
  ArrowRight,
  Building2,
  CheckCircle2,
  ClipboardCheck,
  FileJson,
  Layers,
  ScanLine,
  ShieldCheck,
  Workflow,
} from "lucide-react";
import { ThemeToggle } from "@/components/theme-toggle";
import { BetaSignupForm } from "@/components/beta-signup-form";
import { HomeInteractions } from "@/components/home-interactions";

const features = [
  {
    icon: ScanLine,
    title: "Smart form recognition",
    body:
      "Drop in any scanned page and UniForm instantly recognizes what kind of document it is, whether invoices, claims, applications or onboarding packets, so the right workflow runs every time.",
  },
  {
    icon: FileJson,
    title: "Automated data capture",
    body:
      "Every field, label and value is pulled into clean, structured data your systems can actually use. No more retyping paperwork by hand.",
  },
  {
    icon: ClipboardCheck,
    title: "Human-verified accuracy",
    body:
      "A built-in review step lets your team confirm and correct results before anything is saved, combining automation speed with human judgment.",
  },
  {
    icon: Layers,
    title: "New form types, no rebuild",
    body:
      "Add a brand-new document layout in minutes. UniForm adapts to fresh form types without long retraining cycles or engineering tickets.",
  },
  {
    icon: ShieldCheck,
    title: "Private by design",
    body:
      "Your documents stay in your environment. UniForm is built to keep sensitive records under your control end to end.",
  },
  {
    icon: Workflow,
    title: "Built to scale",
    body:
      "From a handful of forms to a daily flood of paperwork, the same pipeline keeps pace: consistent, reliable and fast.",
  },
];

const steps = [
  {
    n: "01",
    title: "Upload",
    body: "Send in a scanned form or image. UniForm figures out which template it matches.",
  },
  {
    n: "02",
    title: "Extract",
    body: "Key fields and values are read automatically and turned into structured data.",
  },
  {
    n: "03",
    title: "Verify & store",
    body: "Your team reviews the result, makes any fixes, and saves a clean record.",
  },
];

const industries = [
  "Healthcare",
  "Banking & finance",
  "Insurance",
  "Government",
  "Logistics",
  "Legal & compliance",
];

type Stat = {
  value: string;
  label: string;
  count?: number;
  decimals?: number;
  suffix?: string;
};

const stats: Stat[] = [
  { value: "99.2%", label: "Form recognition accuracy", count: 99.2, decimals: 1, suffix: "%" },
  { value: "Minutes", label: "To onboard a new form type" },
  { value: "100%", label: "Results reviewed before storage", count: 100, decimals: 0, suffix: "%" },
  { value: "Structured", label: "Clean data out, every time" },
];

export function HomeLanding() {
  return (
    <div className="min-h-screen bg-[var(--surface-subtle)] text-[var(--ink)]">
      <HomeInteractions />
      {/* Nav */}
      <header className="sticky top-0 z-30 border-b border-[var(--line)] bg-[color-mix(in_srgb,var(--surface)_82%,transparent)] backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-6xl items-center justify-between px-5 lg:px-8">
          <div className="flex items-center gap-3">
            <span className="font-display text-lg font-semibold tracking-tight text-[var(--ink-strong)]">
              UniForm
            </span>
            <span className="rounded-full border border-[var(--line-strong)] bg-[var(--accent-soft)] px-2 py-0.5 font-data text-[10px] font-semibold uppercase tracking-[0.18em] text-[var(--accent-strong)]">
              Beta
            </span>
          </div>

          <nav className="hidden items-center gap-7 text-sm font-medium text-[var(--muted)] md:flex">
            <a href="#features" className="transition hover:text-[var(--ink-strong)]">
              Features
            </a>
            <a href="#how" className="transition hover:text-[var(--ink-strong)]">
              How it works
            </a>
            <a href="#access" className="transition hover:text-[var(--ink-strong)]">
              Request access
            </a>
          </nav>

          <div className="flex items-center gap-2.5">
            <ThemeToggle />
            <Link
              href="/upload"
              className="shine inline-flex h-11 items-center justify-center gap-1.5 rounded-lg bg-[var(--accent)] px-4 text-sm font-semibold text-[var(--background)] shadow-[0_0_24px_-6px_var(--glow-strong)] transition hover:bg-[var(--accent-strong)]"
            >
              Launch app
              <ArrowRight size={16} />
            </Link>
          </div>
        </div>
      </header>

      {/* Hero */}
      <section data-hero className="relative overflow-hidden border-b border-[var(--line)]">
        <div className="atmosphere atmosphere-animate pointer-events-none absolute inset-0 -z-10" />
        <div className="grid-field pointer-events-none absolute inset-0 -z-10" />
        <div data-hero-glow className="hero-glow pointer-events-none absolute inset-0 -z-10" />
        <div className="mx-auto grid max-w-6xl items-center gap-12 px-5 py-20 lg:grid-cols-[1.08fr_0.92fr] lg:px-8 lg:py-28">
          <div>
            <span
              className="reveal inline-flex items-center gap-2 rounded-full border border-[var(--line-strong)] bg-[color-mix(in_srgb,var(--surface)_70%,transparent)] px-3 py-1 font-data text-xs font-semibold uppercase tracking-[0.18em] text-[var(--muted)] backdrop-blur"
              style={{ ["--reveal-delay" as string]: "0ms" }}
            >
              Now in beta · Intelligent document processing
            </span>

            <h1
              className="reveal mt-6 font-display text-[2rem] font-semibold leading-[1.06] tracking-[-0.02em] text-[var(--ink-strong)] min-[420px]:text-[2.4rem] sm:text-5xl sm:leading-[1.04] lg:text-[4.1rem]"
              style={{ ["--reveal-delay" as string]: "80ms" }}
            >
              Turn paperwork into{" "}
              <span className="relative whitespace-nowrap text-[var(--accent)]">
                structured data
                <span className="absolute inset-x-0 -bottom-1 h-px bg-[linear-gradient(90deg,transparent,var(--accent),transparent)]" />
              </span>
              <br className="hidden sm:block" /> automatically.
            </h1>

            <p
              className="reveal mt-6 max-w-xl text-lg leading-relaxed text-[var(--muted)]"
              style={{ ["--reveal-delay" as string]: "160ms" }}
            >
              UniForm reads your scanned forms, recognizes the document type, and pulls every field
              into clean, usable data, with a human review step so you can trust every result.
            </p>

            <div
              className="reveal mt-9 flex flex-col gap-3 sm:flex-row"
              style={{ ["--reveal-delay" as string]: "240ms" }}
            >
              <Link
                href="/upload"
                className="shine inline-flex h-12 items-center justify-center gap-2 rounded-lg bg-[var(--accent)] px-6 text-sm font-semibold text-[var(--background)] shadow-[0_0_30px_-8px_var(--glow-strong)] transition hover:-translate-y-0.5 hover:bg-[var(--accent-strong)] hover:shadow-[0_0_40px_-6px_var(--glow-strong)]"
              >
                Get started
                <ArrowRight size={17} />
              </Link>
              <Link
                href="/dashboard"
                className="inline-flex h-12 items-center justify-center gap-2 rounded-lg border border-[var(--line-strong)] bg-[color-mix(in_srgb,var(--surface)_60%,transparent)] px-6 text-sm font-semibold text-[var(--ink-strong)] backdrop-blur transition hover:border-[var(--accent)]"
              >
                View dashboard
              </Link>
            </div>

            <div
              className="reveal mt-8 flex flex-wrap items-center gap-x-6 gap-y-2 text-sm text-[var(--muted)]"
              style={{ ["--reveal-delay" as string]: "320ms" }}
            >
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 size={16} className="text-[var(--accent)]" />
                No manual data entry
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 size={16} className="text-[var(--accent)]" />
                Human-verified
              </span>
              <span className="inline-flex items-center gap-2">
                <CheckCircle2 size={16} className="text-[var(--accent)]" />
                Private by design
              </span>
            </div>
          </div>

          {/* Hero visual: a stylized "form → data" instrument card */}
          <div
            className="reveal relative"
            style={{ ["--reveal-delay" as string]: "400ms" }}
          >
            <div className="absolute -inset-6 -z-10 rounded-[2rem] bg-[radial-gradient(60%_60%_at_70%_20%,var(--glow),transparent_70%)] blur-2xl" />
            <div
              data-tilt
              data-spotlight
              className="spotlight rounded-2xl border border-[var(--line-strong)] bg-[color-mix(in_srgb,var(--surface)_92%,transparent)] p-5 shadow-[var(--panel-shadow)] backdrop-blur"
            >
              <div className="flex items-center justify-between border-b border-[var(--line)] pb-3">
                <div className="flex items-center gap-2 font-data text-sm font-semibold text-[var(--ink-strong)]">
                  <ScanLine size={16} className="text-[var(--accent)]" />
                  invoice_4821.png
                </div>
                <span className="inline-flex items-center gap-1.5 rounded-full bg-[var(--success-soft)] px-2.5 py-1 text-[11px] font-semibold text-[var(--accent-strong)]">
                  <span className="size-1.5 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--glow-strong)]" />
                  Recognized
                </span>
              </div>

              <div className="mt-4 grid grid-cols-2 gap-3 text-sm">
                {[
                  ["Vendor", "Viet Holdings"],
                  ["Invoice #", "INV-4821"],
                  ["Date", "2026-05-22"],
                  ["Total", "$12,480.00"],
                ].map(([k, v]) => (
                  <div
                    key={k}
                    className="rounded-lg border border-[var(--line)] bg-[var(--surface-subtle)] px-3 py-2.5"
                  >
                    <p className="font-data text-[10px] uppercase tracking-[0.14em] text-[var(--muted)]">
                      {k}
                    </p>
                    <p className="mt-0.5 font-medium text-[var(--ink-strong)]">{v}</p>
                  </div>
                ))}
              </div>

              <div className="mt-4 flex items-center gap-2 rounded-lg border border-[var(--line-strong)] bg-[var(--accent-soft)] px-3 py-2.5 font-data text-xs font-medium text-[var(--accent-strong)]">
                <FileJson size={15} />
                Exported as structured JSON
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats band */}
      <section className="border-b border-[var(--line)] bg-[var(--surface)]">
        <div className="mx-auto grid max-w-6xl grid-cols-2 divide-y divide-[var(--line)] px-5 py-12 sm:divide-y-0 lg:grid-cols-4 lg:divide-x lg:px-8">
          {stats.map((s, i) => (
            <div
              key={s.label}
              data-reveal
              className="px-0 py-3 lg:px-7 lg:py-0 lg:first:pl-0"
              style={{ ["--reveal-delay" as string]: `${i * 70}ms` }}
            >
              <p className="font-display text-4xl font-semibold tracking-tight text-[var(--ink-strong)] lg:text-[2.75rem]">
                {s.count != null ? (
                  <span data-count={s.count} data-decimals={s.decimals ?? 0} data-suffix={s.suffix ?? ""}>
                    {s.value}
                  </span>
                ) : (
                  s.value
                )}
              </p>
              <p className="mt-1.5 text-sm text-[var(--muted)]">{s.label}</p>
            </div>
          ))}
        </div>
      </section>

      {/* Features */}
      <section id="features" className="mx-auto max-w-6xl px-5 py-24 lg:px-8">
        <div data-reveal className="max-w-2xl">
          <p className="font-data text-sm font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
            Why UniForm
          </p>
          <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight text-[var(--ink-strong)] sm:text-[2.6rem]">
            Everything you need to retire manual data entry
          </h2>
          <p className="mt-4 text-lg text-[var(--muted)]">
            From the moment a document arrives to the moment it&apos;s stored as clean data, UniForm
            handles the heavy lifting and keeps your team in control.
          </p>
        </div>

        <div className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => {
            const Icon = f.icon;
            return (
              <div
                key={f.title}
                data-reveal
                data-spotlight
                style={{ ["--reveal-delay" as string]: `${(i % 3) * 90}ms` }}
                className="spotlight group rounded-2xl border border-[var(--line)] bg-[var(--surface)] p-6 transition duration-300 hover:-translate-y-1 hover:border-[var(--accent)] hover:shadow-[var(--panel-shadow)]"
              >
                <div className="flex size-11 items-center justify-center rounded-xl border border-[var(--line-strong)] bg-[var(--accent-soft)] text-[var(--accent-strong)] transition duration-300 group-hover:scale-105">
                  <Icon size={20} />
                </div>
                <h3 className="mt-5 font-display text-lg font-semibold text-[var(--ink-strong)]">
                  {f.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-[var(--muted)]">{f.body}</p>
              </div>
            );
          })}
        </div>
      </section>

      {/* How it works */}
      <section id="how" className="relative overflow-hidden border-y border-[var(--line)] bg-[var(--surface)]">
        <div className="grid-field pointer-events-none absolute inset-0 -z-10 opacity-60" />
        <div className="mx-auto max-w-6xl px-5 py-24 lg:px-8">
          <div data-reveal className="max-w-2xl">
            <p className="font-data text-sm font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
              How it works
            </p>
            <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight text-[var(--ink-strong)] sm:text-[2.6rem]">
              From scan to structured data in three steps
            </h2>
          </div>

          <div className="mt-14 grid gap-5 md:grid-cols-3">
            {steps.map((s, i) => (
              <div
                key={s.n}
                data-reveal
                data-spotlight
                style={{ ["--reveal-delay" as string]: `${i * 110}ms` }}
                className="spotlight relative rounded-2xl border border-[var(--line)] bg-[var(--surface-subtle)] p-7 transition duration-300 hover:-translate-y-1 hover:border-[var(--accent)]"
              >
                <span className="font-display text-5xl font-bold leading-none text-[color-mix(in_srgb,var(--accent)_30%,transparent)]">
                  {s.n}
                </span>
                <h3 className="mt-4 font-display text-xl font-semibold text-[var(--ink-strong)]">
                  {s.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-[var(--muted)]">{s.body}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Industries */}
      <section id="industries" className="mx-auto max-w-6xl px-5 py-24 lg:px-8">
        <div className="grid items-center gap-12 lg:grid-cols-[0.9fr_1.1fr]">
          <div data-reveal>
            <p className="font-data text-sm font-semibold uppercase tracking-[0.18em] text-[var(--accent)]">
              Built for document-heavy teams
            </p>
            <h2 className="mt-3 font-display text-3xl font-semibold tracking-tight text-[var(--ink-strong)] sm:text-[2.6rem]">
              Wherever forms pile up, UniForm keeps up
            </h2>
            <p className="mt-4 text-lg text-[var(--muted)]">
              The organizations that move the most paper feel the most pain. UniForm was made for
              exactly those workflows.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            {industries.map((name, i) => (
              <div
                key={name}
                data-reveal
                style={{ ["--reveal-delay" as string]: `${(i % 3) * 80}ms` }}
                className="flex items-center gap-2.5 rounded-xl border border-[var(--line)] bg-[var(--surface)] px-4 py-4 text-sm font-medium text-[var(--ink-strong)] transition duration-300 hover:-translate-y-0.5 hover:border-[var(--accent)] hover:bg-[var(--accent-soft)]"
              >
                <Building2 size={17} className="shrink-0 text-[var(--accent)]" />
                {name}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Beta access */}
      <section id="access" className="px-5 pb-24 lg:px-8">
        <div className="relative mx-auto max-w-3xl overflow-hidden rounded-3xl border border-[var(--line-strong)] bg-[var(--surface)] px-7 py-12 shadow-[var(--panel-shadow)] sm:px-10 lg:px-12">
          <div className="atmosphere atmosphere-animate pointer-events-none absolute inset-0 -z-10" />
          <div data-reveal className="text-center">
            <span className="inline-flex items-center gap-2 rounded-full border border-[var(--line-strong)] bg-[var(--accent-soft)] px-3 py-1 font-data text-xs font-semibold uppercase tracking-[0.16em] text-[var(--accent-strong)]">
              UniForm is in beta
            </span>
            <h2 className="mx-auto mt-5 max-w-xl font-display text-3xl font-semibold tracking-tight text-[var(--ink-strong)] sm:text-[2.6rem]">
              Request your beta access
            </h2>
            <p className="mx-auto mt-4 max-w-xl text-lg text-[var(--muted)]">
              We&apos;re onboarding new teams gradually. Leave your name and email and we&apos;ll
              reach out with access.
            </p>
          </div>

          <div className="mx-auto mt-9 max-w-xl">
            <BetaSignupForm />
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-[var(--line)] bg-[var(--surface)]">
        <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-4 px-5 py-8 text-sm text-[var(--muted)] sm:flex-row lg:px-8">
          <div className="flex items-center gap-2.5">
            <span className="font-display font-semibold text-[var(--ink-strong)]">UniForm</span>
          </div>
          <p>Unified form ingestion, from scan to structured data.</p>
        </div>
      </footer>
    </div>
  );
}
