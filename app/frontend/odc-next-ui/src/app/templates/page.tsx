import { TemplatesWorkspace } from "@/components/templates-workspace";
import { WorkspaceShell } from "@/components/workspace-shell";

export default function TemplatesPage() {
  return (
    <WorkspaceShell
      currentSection="templates"
      title="Template Library"
    >
      <TemplatesWorkspace />
    </WorkspaceShell>
  );
}
