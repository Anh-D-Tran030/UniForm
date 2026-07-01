import { DremioConsoleWorkspace } from "@/components/dremio-console-workspace";
import { WorkspaceShell } from "@/components/workspace-shell";

export default function DremioPage() {
  return (
    <WorkspaceShell
      currentSection="dremio"
      title="Dremio Console"
    >
      <DremioConsoleWorkspace />
    </WorkspaceShell>
  );
}
