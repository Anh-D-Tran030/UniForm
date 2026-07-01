import { proxyServiceRequest } from "@/lib/backend";

export const runtime = "nodejs";

type RouteContext = {
  params: Promise<{
    runId: string;
    templateId: string;
  }>;
};

export async function GET(_request: Request, context: RouteContext) {
  const { runId, templateId } = await context.params;
  const path = `/forms/${encodeURIComponent(templateId)}/${encodeURIComponent(runId)}/json`;

  return proxyServiceRequest("storage", path, { method: "GET" });
}
