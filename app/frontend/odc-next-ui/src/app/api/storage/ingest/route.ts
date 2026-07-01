import { readFile } from "node:fs/promises";
import { proxyFormDataToService, serviceBaseUrl } from "@/lib/backend";
import { recordStorageEvent } from "@/lib/metrics-store";
import { appendCachedUpload, readCachedUpload } from "@/lib/upload-cache";
import { validateImageFile } from "@/lib/upload-validation";
import { resolveWorkspacePath } from "@/lib/workspace-file";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const incoming = await request.formData();
  const uploadId = incoming.get("upload_id");
  const imagePath = incoming.get("image_path");
  const image = incoming.get("image");
  const kvpJson = incoming.get("kvp_json");
  const runId = incoming.get("run_id");
  const templateId = incoming.get("template_id");
  const runIdValue = typeof runId === "string" ? runId : null;
  const templateIdValue = typeof templateId === "string" ? templateId : null;

  async function failure(detail: string, status: number) {
    await recordStorageEvent({
      run_id: runIdValue,
      success: false,
      template_id: templateIdValue,
    });
    return Response.json({ detail }, { status });
  }

  if (typeof kvpJson !== "string" || !kvpJson.trim()) {
    return failure("Missing kvp_json", 400);
  }

  try {
    JSON.parse(kvpJson);
  } catch (error) {
    const message = error instanceof Error ? error.message : "Invalid JSON";
    return failure(`Invalid kvp_json: ${message}`, 400);
  }

  const outgoing = new FormData();

  if (typeof uploadId === "string" && uploadId) {
    try {
      appendCachedUpload(outgoing, await readCachedUpload(uploadId));
    } catch {
      return failure("Uploaded image was not found", 404);
    }
  } else if (typeof imagePath === "string" && imagePath) {
    const resolvedPath = resolveWorkspacePath(imagePath);
    if (!resolvedPath) {
      return failure("Image path is not allowed", 403);
    }

    try {
      const fileBuffer = await readFile(resolvedPath);
      const filename = resolvedPath.split(/[\\/]/).pop() ?? "query.png";
      outgoing.append(
        "image",
        new Blob([new Uint8Array(fileBuffer)], { type: "image/png" }),
        filename,
      );
    } catch {
      return failure("Image not found", 404);
    }
  } else {
    const validationError = validateImageFile(image);
    if (validationError) {
      return failure(validationError, 400);
    }
    outgoing.append("image", image as File, (image as File).name);
  }

  outgoing.append("kvp_json", kvpJson);
  if (typeof runId === "string") {
    outgoing.append("run_id", runId);
  }
  if (typeof templateId === "string") {
    outgoing.append("template_id", templateId);
  }

  const response = await proxyFormDataToService("storage", "/ingest", outgoing);
  const payload = await response.json().catch(() => ({}));

  if (response.ok && typeof payload === "object" && payload) {
    const storedRunId = "run_id" in payload && typeof payload.run_id === "string" ? payload.run_id : runIdValue;
    const storedTemplateId =
      "template_id" in payload && typeof payload.template_id === "string" ? payload.template_id : templateIdValue;

    if (storedRunId && storedTemplateId) {
      const goldUrl = new URL(`${serviceBaseUrl("gold")}/transform`);
      goldUrl.searchParams.set("run_id", storedRunId);
      goldUrl.searchParams.set("template_id", storedTemplateId);

      try {
        const goldResponse = await fetch(goldUrl, {
          method: "POST",
          cache: "no-store",
          signal: AbortSignal.timeout(15000),
        });
        const goldPayload = await goldResponse.json().catch(() => ({}));
        Object.assign(payload, {
          gold_transform: goldPayload,
          gold_transform_status: goldResponse.ok ? "completed" : "failed",
        });
      } catch (error) {
        Object.assign(payload, {
          gold_transform_error: error instanceof Error ? error.message : "Gold transform failed",
          gold_transform_status: "failed",
        });
      }
    }
  }

  await recordStorageEvent({
    run_id:
      typeof payload === "object" && payload && "run_id" in payload && typeof payload.run_id === "string"
        ? payload.run_id
        : typeof runId === "string"
          ? runId
          : null,
    success: response.ok,
    template_id:
      typeof payload === "object" && payload && "template_id" in payload && typeof payload.template_id === "string"
        ? payload.template_id
        : typeof templateId === "string"
          ? templateId
          : null,
  });

  return Response.json(payload, {
    status: response.status,
    headers: { "cache-control": "no-store" },
  });
}
