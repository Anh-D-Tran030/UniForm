import { proxyFormDataToService } from "@/lib/backend";
import { validateImageFile } from "@/lib/upload-validation";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const incoming = await request.formData();
  const image = incoming.get("image");
  const validationError = validateImageFile(image);

  if (validationError) {
    return Response.json({ detail: validationError }, { status: 400 });
  }

  const outgoing = new FormData();
  outgoing.append("image", image as File, (image as File).name);

  const templateId = incoming.get("template_id");
  const displayName = incoming.get("display_name");
  if (typeof templateId === "string") {
    outgoing.append("template_id", templateId);
  }
  if (typeof displayName === "string") {
    outgoing.append("display_name", displayName);
  }

  return proxyFormDataToService("odc", "/embed", outgoing, request.url);
}
