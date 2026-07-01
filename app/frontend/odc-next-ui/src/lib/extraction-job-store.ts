import { mkdir, readdir, readFile, rm, stat, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

export type ExtractionJobStatus = "accepted" | "running" | "kvp_ready" | "failed" | "expired";

export type ExtractionJob = {
  created_at: string;
  error: string | null;
  finished_at: string | null;
  kvp_payload: unknown | null;
  kvp_status: "pending" | "running" | "ready" | "failed";
  run_id: string;
  started_at: string | null;
  status: ExtractionJobStatus;
  updated_at: string;
  upload_id: string;
};

const JOB_DIR = path.join(os.tmpdir(), "odc-next-ui-extraction-jobs");
const JOB_TTL_MS = 4 * 60 * 60 * 1000;

function assertSafeId(id: string) {
  if (!/^[A-Za-z0-9_-]+$/.test(id)) {
    throw new Error("Invalid extraction job id");
  }
}

function jobPath(runId: string) {
  assertSafeId(runId);
  return path.join(JOB_DIR, `${runId}.json`);
}

async function ensureJobDir() {
  await mkdir(JOB_DIR, { recursive: true });
}

export async function cleanupExpiredExtractionJobs(now = Date.now()) {
  await ensureJobDir();

  const entries = await readdir(JOB_DIR, { withFileTypes: true }).catch(() => []);
  await Promise.all(
    entries
      .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
      .map(async (entry) => {
        const fullPath = path.join(JOB_DIR, entry.name);
        const details = await stat(fullPath).catch(() => null);
        if (details && now - details.mtimeMs > JOB_TTL_MS) {
          await rm(fullPath, { force: true }).catch(() => null);
        }
      }),
  );
}

export async function readExtractionJob(runId: string): Promise<ExtractionJob | null> {
  try {
    const raw = await readFile(jobPath(runId), "utf-8");
    const parsed = JSON.parse(raw) as ExtractionJob;
    return parsed;
  } catch {
    return null;
  }
}

export async function writeExtractionJob(job: ExtractionJob) {
  await ensureJobDir();
  await writeFile(jobPath(job.run_id), JSON.stringify(job, null, 2), "utf-8");
}

export async function createExtractionJob(runId: string, uploadId: string) {
  const now = new Date().toISOString();
  const existing = await readExtractionJob(runId);

  if (existing && existing.status !== "failed" && existing.status !== "expired") {
    return existing;
  }

  const job: ExtractionJob = {
    created_at: now,
    error: null,
    finished_at: null,
    kvp_payload: null,
    kvp_status: "pending",
    run_id: runId,
    started_at: null,
    status: "accepted",
    updated_at: now,
    upload_id: uploadId,
  };

  await writeExtractionJob(job);
  return job;
}

export async function markExtractionJobRunning(job: ExtractionJob) {
  const now = new Date().toISOString();
  const nextJob: ExtractionJob = {
    ...job,
    error: null,
    kvp_status: "running",
    started_at: job.started_at ?? now,
    status: "running",
    updated_at: now,
  };

  await writeExtractionJob(nextJob);
  return nextJob;
}

export async function markExtractionJobReady(job: ExtractionJob, kvpPayload: unknown) {
  const now = new Date().toISOString();
  const nextJob: ExtractionJob = {
    ...job,
    error: null,
    finished_at: now,
    kvp_payload: kvpPayload,
    kvp_status: "ready",
    status: "kvp_ready",
    updated_at: now,
  };

  await writeExtractionJob(nextJob);
  return nextJob;
}

export async function markExtractionJobFailed(job: ExtractionJob, error: unknown) {
  const now = new Date().toISOString();
  const message = error instanceof Error ? error.message : "Extraction precompute failed";
  const nextJob: ExtractionJob = {
    ...job,
    error: message,
    finished_at: now,
    kvp_payload: null,
    kvp_status: "failed",
    status: "failed",
    updated_at: now,
  };

  await writeExtractionJob(nextJob);
  return nextJob;
}
