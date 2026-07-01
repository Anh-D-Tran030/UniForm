import { UploadMatchWorkspace } from "@/components/upload-match-workspace";
import { WorkspaceShell } from "@/components/workspace-shell";

export default function UploadPage() {
  return (
    <WorkspaceShell
      currentSection="upload"
      title="Upload Source Images"
    >
      <UploadMatchWorkspace />
    </WorkspaceShell>
  );
}
