import { proxyServiceRequest } from "@/lib/backend";

export const runtime = "nodejs";

export async function GET() {
  return proxyServiceRequest("gold", "/gold/summary", { method: "GET" });
}
