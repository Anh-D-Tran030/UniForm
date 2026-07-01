import { proxyServiceRequest } from "@/lib/backend";

export const runtime = "nodejs";

export async function GET() {
  return proxyServiceRequest("storage", "/forms", { method: "GET" });
}
