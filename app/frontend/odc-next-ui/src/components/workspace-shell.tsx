import Link from "next/link";
import { BarChart3, ClipboardList, Database, HardDrive, LayoutGrid, Upload } from "lucide-react";
import type { ReactNode } from "react";
import { ThemeToggle } from "@/components/theme-toggle";
import { LogoutButton } from "@/components/logout-button";

type SectionKey = "upload" | "templates" | "extraction" | "forms" | "dashboard" | "minio" | "dremio";

type WorkspaceShellProps = {
  children: ReactNode;
  currentSection: SectionKey;
  title: string;
};

const navigation = [
  { href: "/upload", icon: Upload, key: "upload" as const, label: "Upload & Match" },
  { href: "/templates", icon: LayoutGrid, key: "templates" as const, label: "Templates" },
  { href: "/extraction", icon: Database, key: "extraction" as const, label: "Data Extraction" },
  { href: "/forms", icon: ClipboardList, key: "forms" as const, label: "Uploaded Forms" },
  { href: "/dashboard", icon: BarChart3, key: "dashboard" as const, label: "Dashboard" },
  { href: "/minio", icon: HardDrive, key: "minio" as const, label: "MinIO Console" },
  { href: "/dremio", icon: BarChart3, key: "dremio" as const, label: "Dremio Console" },
];

export function WorkspaceShell({
  children,
  currentSection,
  title,
}: WorkspaceShellProps) {
  return (
    <div className="workspace-app min-h-screen bg-[var(--surface-subtle)] text-[var(--ink)]">
      <div className="flex min-h-screen w-full flex-col lg:flex-row">
        <aside className="relative flex w-full shrink-0 flex-col overflow-hidden border-b border-[var(--line)] bg-[var(--surface)] px-5 py-4 lg:w-[260px] lg:border-b-0 lg:border-r lg:py-6">
          <div className="relative flex items-center justify-between gap-3">
            <div className="flex items-center gap-3">
              <div>
                <p className="font-display text-xl font-semibold tracking-tight text-[var(--ink-strong)]">
                  UniForm
                </p>
                <h1 className="font-data text-[11px] font-semibold uppercase tracking-[0.18em] text-[var(--muted)]">Ingestion Workspace</h1>
              </div>
            </div>
            <div className="lg:hidden">
              <LogoutButton />
            </div>
          </div>

          <Link
            href="/upload"
            className="relative mt-8 hidden h-11 items-center justify-center rounded-lg bg-[var(--accent)] px-4 text-sm font-semibold text-[var(--background)] shadow-[0_0_24px_-8px_var(--glow-strong)] transition hover:bg-[var(--accent-strong)] lg:inline-flex"
          >
            New Processing Job
          </Link>

          <div className="relative mt-4 lg:mt-8">
            <p className="mb-2 hidden px-3 font-data text-[10px] font-semibold uppercase tracking-[0.2em] text-[var(--muted-light)] lg:block">
              Workspace
            </p>
            <nav className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1 lg:mx-0 lg:block lg:space-y-1 lg:overflow-visible lg:px-0 lg:pb-0">
              {navigation.map((item) => {
                const Icon = item.icon;
                const isActive = item.key === currentSection;

                return (
                  <Link
                    key={item.key}
                    href={item.href}
                    className={`group relative flex h-11 shrink-0 items-center gap-2 whitespace-nowrap rounded-lg px-3 text-sm font-medium transition lg:w-full lg:shrink lg:gap-3 ${
                      isActive
                        ? "bg-[var(--accent-soft)] text-[var(--accent-strong)]"
                        : "text-[var(--muted)] hover:bg-[var(--surface-subtle)] hover:text-[var(--ink-strong)]"
                    }`}
                  >
                    {isActive ? (
                      <span className="absolute inset-x-2 bottom-0 hidden h-0.5 rounded-full bg-[var(--accent)] shadow-[0_0_8px_var(--glow-strong)] lg:inset-y-2 lg:left-0 lg:right-auto lg:block lg:h-auto lg:w-0.5" />
                    ) : null}
                    <Icon size={18} />
                    <span>{item.label}</span>
                  </Link>
                );
              })}
            </nav>
          </div>

          <div className="relative mt-auto hidden border-t border-[var(--line)] pt-4 lg:block">
            <LogoutButton />
          </div>
        </aside>

        <div className="flex min-h-screen flex-1 flex-col">
          <header className="border-b border-[var(--line)] bg-[var(--surface)] px-4 py-3 lg:px-6 lg:py-4">
            <div className="flex items-center justify-between gap-4">
              <h2 className="font-display text-2xl font-semibold tracking-tight text-[var(--ink-strong)] lg:text-[1.75rem]">{title}</h2>
              <ThemeToggle />
            </div>
          </header>

          <main className="flex-1 px-4 py-4 lg:px-6 lg:py-5">{children}</main>
        </div>
      </div>
    </div>
  );
}
