import { readCachedUpload } from "@/lib/upload-cache";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{
    uploadId: string;
  }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { uploadId } = await context.params;

  try {
    const upload = await readCachedUpload(uploadId);

    return new Response(new Uint8Array(upload.buffer), {
      headers: {
        "cache-control": "no-store",
        "content-disposition": `inline; filename="${upload.filename.replaceAll("\"", "")}"`,
        "content-type": upload.contentType,
      },
    });
  } catch {
    return Response.json({ detail: "Uploaded image was not found" }, { status: 404 });
  }
}
