import { readExtractionJob } from "@/lib/extraction-job-store";

export const runtime = "nodejs";

type JobRouteContext = {
  params: Promise<{
    runId: string;
  }>;
};

export async function GET(_request: Request, context: JobRouteContext) {
  const { runId } = await context.params;
  const job = await readExtractionJob(runId);

  if (!job) {
    return Response.json({ detail: "Extraction job not found" }, { status: 404 });
  }

  return Response.json(job, {
    headers: { "cache-control": "no-store" },
  });
}
