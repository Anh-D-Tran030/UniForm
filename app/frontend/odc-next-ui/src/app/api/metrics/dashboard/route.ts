import { getDashboardMetrics } from "@/lib/metrics-store";

export const runtime = "nodejs";

export async function GET() {
  return Response.json(await getDashboardMetrics(), {
    headers: { "cache-control": "no-store" },
  });
}
