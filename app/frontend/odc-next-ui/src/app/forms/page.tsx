import { UploadedFormsWorkspace } from "@/components/uploaded-forms-workspace";
import { WorkspaceShell } from "@/components/workspace-shell";

export default function FormsPage() {
  return (
    <WorkspaceShell
      currentSection="forms"
      title="Uploaded Forms"
    >
      <UploadedFormsWorkspace />
    </WorkspaceShell>
  );
}
