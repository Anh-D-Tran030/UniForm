import { proxyFormDataToService } from "@/lib/backend";
import { startExtractionPrecompute } from "@/lib/extraction-precompute";
import { recordQueryEvent } from "@/lib/metrics-store";
import { saveUploadFile } from "@/lib/upload-cache";
import { validateImageFile } from "@/lib/upload-validation";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const startedAt = Date.now();
  const queryEventId = crypto.randomUUID();
  const runId = crypto.randomUUID();
  const topK = Number(new URL(request.url).searchParams.get("top_k") ?? 5);
  const incoming = await request.formData();
  const image = incoming.get("image");
  const validationError = validateImageFile(image);

  if (validationError) {
    await recordQueryEvent({
      event_id: queryEventId,
      file_name: image instanceof File ? image.name : null,
      file_size: image instanceof File ? image.size : null,
      latency_ms: Date.now() - startedAt,
      match_count: 0,
      success: false,
      top_k: Number.isFinite(topK) ? topK : 5,
      top_scores: [],
    });
    return Response.json({ detail: validationError }, { status: 400 });
  }

  const file = image as File;
  const upload = await saveUploadFile(file);
  const precomputePromise = startExtractionPrecompute({
    runId,
    uploadId: upload.uploadId,
  }).catch(() => null);
  const outgoing = new FormData();
  outgoing.append("image", file, file.name);

  const response = await proxyFormDataToService("odc", "/query", outgoing, request.url);
  const payload = await response.json();
  const matches = Array.isArray(payload.matches) ? payload.matches : [];
  const topScores = matches
    .map((match: { cosine_similarity?: unknown }) => Number(match.cosine_similarity))
    .filter((score: number) => Number.isFinite(score));

  await recordQueryEvent({
    event_id: queryEventId,
    file_name: upload.filename,
    file_size: upload.size,
    latency_ms: Date.now() - startedAt,
    match_count: matches.length,
    success: response.ok,
    top_k: Number.isFinite(topK) ? topK : 5,
    top_scores: topScores,
  });

  if (!response.ok) {
    return Response.json(payload, {
      status: response.status,
      headers: { "cache-control": "no-store" },
    });
  }

  const precomputeJob = await precomputePromise;

  return Response.json(
    {
      ...payload,
      precompute_status: precomputeJob?.status ?? "failed",
      run_id: runId,
      upload_id: upload.uploadId,
      uploaded_file_name: upload.filename,
      uploaded_mime_type: upload.contentType,
      uploaded_size: upload.size,
      query_event_id: queryEventId,
    },
    {
      status: response.status,
      headers: { "cache-control": "no-store" },
    },
  );
}
