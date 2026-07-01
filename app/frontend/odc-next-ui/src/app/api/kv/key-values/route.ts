import { readFile } from "node:fs/promises";
import { proxyFormDataToService } from "@/lib/backend";
import { appendCachedUpload, readCachedUpload } from "@/lib/upload-cache";
import { validateImageFile } from "@/lib/upload-validation";
import { resolveWorkspacePath } from "@/lib/workspace-file";

export const runtime = "nodejs";

function appendBufferUpload(formData: FormData, fileBuffer: Buffer, filename: string, contentType = "image/png") {
  formData.append(
    "image",
    new Blob([new Uint8Array(fileBuffer)], { type: contentType }),
    filename,
  );
}

async function formDataFromRequest(request: Request) {
  const contentType = request.headers.get("content-type") ?? "";

  if (contentType.includes("multipart/form-data")) {
    const incoming = await request.formData();
    const image = incoming.get("image");
    const validationError = validateImageFile(image);
    if (validationError) {
      return { error: Response.json({ detail: validationError }, { status: 400 }) };
    }

    const outgoing = new FormData();
    outgoing.append("image", image as File, (image as File).name);
    outgoing.append("re_threshold", "0.05");
    return { formData: outgoing };
  }

  const payload = await request.json();
  const uploadId = typeof payload.uploadId === "string" ? payload.uploadId : "";
  const imagePath = typeof payload.imagePath === "string" ? payload.imagePath : "";

  if (uploadId) {
    try {
      const upload = await readCachedUpload(uploadId);
      const outgoing = new FormData();
      appendCachedUpload(outgoing, upload);
      outgoing.append("re_threshold", "0.05");
      return { formData: outgoing };
    } catch {
      return { error: Response.json({ detail: "Uploaded image was not found" }, { status: 404 }) };
    }
  }

  if (imagePath) {
    const resolvedPath = resolveWorkspacePath(imagePath);
    if (!resolvedPath) {
      return { error: Response.json({ detail: "Image path is not allowed" }, { status: 403 }) };
    }

    try {
      const fileBuffer = await readFile(resolvedPath);
      const filename = resolvedPath.split(/[\\/]/).pop() ?? "query.png";
      const outgoing = new FormData();
      appendBufferUpload(outgoing, fileBuffer, filename);
      outgoing.append("re_threshold", "0.05");
      return { formData: outgoing };
    } catch {
      return { error: Response.json({ detail: "Image not found" }, { status: 404 }) };
    }
  }

  return { error: Response.json({ detail: "Missing upload id or image path" }, { status: 400 }) };
}

export async function POST(request: Request) {
  const resolved = await formDataFromRequest(request);
  if (resolved.error) {
    return resolved.error;
  }

  return proxyFormDataToService("kv", "/key-values", resolved.formData);
}
