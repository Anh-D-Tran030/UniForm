import { readFile } from "node:fs/promises";
import path from "node:path";
import { resolveWorkspacePath } from "@/lib/workspace-file";

const CONTENT_TYPES: Record<string, string> = {
  ".bmp": "image/bmp",
  ".gif": "image/gif",
  ".jpeg": "image/jpeg",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".tif": "image/tiff",
  ".tiff": "image/tiff",
  ".webp": "image/webp",
};

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const inputPath = searchParams.get("path");
  const cacheMode = searchParams.get("cache");

  if (!inputPath) {
    return Response.json({ detail: "Missing image path" }, { status: 400 });
  }

  const resolvedPath = resolveWorkspacePath(inputPath);
  if (!resolvedPath) {
    return Response.json({ detail: "Image path is not allowed" }, { status: 403 });
  }

  try {
    const file = await readFile(resolvedPath);
    const extension = path.extname(resolvedPath).toLowerCase();
    const cacheControl =
      cacheMode === "template"
        ? "public, max-age=86400, stale-while-revalidate=604800"
        : "no-store";

    return new Response(file, {
      headers: {
        "cache-control": cacheControl,
        "content-type": CONTENT_TYPES[extension] ?? "application/octet-stream",
      },
    });
  } catch {
    return Response.json({ detail: "Image not found" }, { status: 404 });
  }
}
