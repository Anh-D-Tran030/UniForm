import { proxyServiceRequest } from "@/lib/backend";

export const runtime = "nodejs";

// Streams a single MinIO object (image bytes, JSON, etc.) through the storage
// service. The ?key= query param is forwarded verbatim via sourceUrl.
export async function GET(request: Request) {
  return proxyServiceRequest("storage", "/object", { method: "GET" }, request.url);
}
