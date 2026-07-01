import { proxyServiceRequest } from "@/lib/backend";

export const runtime = "nodejs";

export async function GET(request: Request) {
  return proxyServiceRequest("storage", "/objects", { method: "GET" }, request.url);
}
