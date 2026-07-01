import { ExtractionWorkspace } from "@/components/extraction-workspace";
import { WorkspaceShell } from "@/components/workspace-shell";

type ExtractionPageProps = {
  searchParams: Promise<Record<string, string | string[] | undefined>>;
};

export default async function ExtractionPage({ searchParams }: ExtractionPageProps) {
  const params = await searchParams;

  return (
    <WorkspaceShell
      currentSection="extraction"
      title="Data Extraction"
    >
      <ExtractionWorkspace
        queryPath={typeof params.queryPath === "string" ? params.queryPath : null}
        uploadId={typeof params.uploadId === "string" ? params.uploadId : null}
        initialRunId={typeof params.runId === "string" ? params.runId : null}
        fileName={typeof params.fileName === "string" ? params.fileName : null}
        templateId={typeof params.templateId === "string" ? params.templateId : ""}
        templateImagePath={
          typeof params.templateImagePath === "string" ? params.templateImagePath : null
        }
        templateName={typeof params.templateName === "string" ? params.templateName : ""}
        score={typeof params.score === "string" ? Number(params.score) : 0}
      />
    </WorkspaceShell>
  );
}
