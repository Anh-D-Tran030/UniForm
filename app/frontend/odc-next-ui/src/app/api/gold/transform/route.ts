import { proxyServiceRequest } from "@/lib/backend";

export const runtime = "nodejs";

export async function POST(request: Request) {
  return proxyServiceRequest("gold", "/transform", { method: "POST" }, request.url);
}
