import {
  cleanupExpiredExtractionJobs,
  createExtractionJob,
  markExtractionJobFailed,
  markExtractionJobReady,
  markExtractionJobRunning,
  readExtractionJob,
} from "@/lib/extraction-job-store";
import { recordPrecomputeEvent } from "@/lib/metrics-store";
import { appendCachedUpload, readCachedUpload } from "@/lib/upload-cache";
import { serviceBaseUrl } from "@/lib/backend";

type StartExtractionPrecomputeInput = {
  runId: string;
  uploadId: string;
};

const activeJobs = new Map<string, Promise<void>>();

function isTerminalStatus(status: string) {
  return status === "kvp_ready" || status === "running";
}

async function runExtractionPrecompute({ runId, uploadId }: StartExtractionPrecomputeInput) {
  const startedAt = Date.now();
  let job = await createExtractionJob(runId, uploadId);

  try {
    job = await markExtractionJobRunning(job);

    const upload = await readCachedUpload(uploadId);
    const formData = new FormData();
    appendCachedUpload(formData, upload);
    formData.append("re_threshold", "0.05");

    const response = await fetch(`${serviceBaseUrl("kv")}/key-values`, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });
    const payload = await response.json().catch(() => ({}));

    if (!response.ok) {
      const detail =
        typeof payload === "object" && payload && "detail" in payload
          ? String(payload.detail)
          : "KVP precompute failed";
      throw new Error(detail);
    }

    await markExtractionJobReady(job, payload);
    await recordPrecomputeEvent({
      latency_ms: Date.now() - startedAt,
      run_id: runId,
      status: "kvp_ready",
      success: true,
      upload_id: uploadId,
    });
  } catch (error) {
    await markExtractionJobFailed(job, error).catch(() => null);
    await recordPrecomputeEvent({
      latency_ms: Date.now() - startedAt,
      run_id: runId,
      status: "failed",
      success: false,
      upload_id: uploadId,
    }).catch(() => null);
  } finally {
    activeJobs.delete(runId);
  }
}

export async function startExtractionPrecompute(input: StartExtractionPrecomputeInput) {
  await cleanupExpiredExtractionJobs().catch(() => null);

  const existing = await readExtractionJob(input.runId);
  if (existing && isTerminalStatus(existing.status)) {
    return existing;
  }

  const job = await createExtractionJob(input.runId, input.uploadId);
  if (!activeJobs.has(input.runId)) {
    activeJobs.set(input.runId, runExtractionPrecompute(input));
  }

  return job;
}
