import { proxyRequest } from "@/lib/backend";

type RouteContext = {
  params: Promise<{
    templateId: string;
  }>;
};

export async function DELETE(_: Request, context: RouteContext) {
  const { templateId } = await context.params;
  return proxyRequest(`/templates/${encodeURIComponent(templateId)}`, {
    method: "DELETE",
  });
}
