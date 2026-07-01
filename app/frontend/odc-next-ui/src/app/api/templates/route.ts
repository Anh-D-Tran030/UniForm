import { proxyRequest } from "@/lib/backend";

export async function GET() {
  return proxyRequest("/templates", { method: "GET" });
}
