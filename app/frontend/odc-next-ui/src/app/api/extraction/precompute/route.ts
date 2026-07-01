import { startExtractionPrecompute } from "@/lib/extraction-precompute";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const payload = await request.json().catch(() => ({}));
  const runId = typeof payload.run_id === "string" ? payload.run_id : "";
  const uploadId = typeof payload.upload_id === "string" ? payload.upload_id : "";

  if (!runId || !uploadId) {
    return Response.json({ detail: "Missing run_id or upload_id" }, { status: 400 });
  }

  try {
    const job = await startExtractionPrecompute({ runId, uploadId });
    return Response.json(
      {
        run_id: job.run_id,
        status: job.status,
        upload_id: job.upload_id,
      },
      { headers: { "cache-control": "no-store" } },
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : "Failed to start extraction precompute";
    return Response.json({ detail }, { status: 500 });
  }
}
