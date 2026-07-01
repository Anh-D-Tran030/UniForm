import { MinioConsoleWorkspace } from "@/components/minio-console-workspace";
import { WorkspaceShell } from "@/components/workspace-shell";

export default function MinioPage() {
  return (
    <WorkspaceShell
      currentSection="minio"
      title="MinIO Console"
    >
      <MinioConsoleWorkspace />
    </WorkspaceShell>
  );
}
