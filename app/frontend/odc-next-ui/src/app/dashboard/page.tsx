import { DashboardWorkspace } from "@/components/dashboard-workspace";
import { WorkspaceShell } from "@/components/workspace-shell";

export default function DashboardPage() {
  return (
    <WorkspaceShell
      currentSection="dashboard"
      title="Performance Dashboard"
    >
      <DashboardWorkspace />
    </WorkspaceShell>
  );
}
