import { recordSelectionEvent } from "@/lib/metrics-store";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const payload = await request.json().catch(() => null);

  if (!payload || typeof payload !== "object") {
    return Response.json({ detail: "Invalid selection payload" }, { status: 400 });
  }

  const selectedRank = Number("selected_rank" in payload ? payload.selected_rank : 0);
  if (!Number.isFinite(selectedRank) || selectedRank < 1) {
    return Response.json({ detail: "selected_rank must be a positive number" }, { status: 400 });
  }

  await recordSelectionEvent({
    query_event_id:
      "query_event_id" in payload && typeof payload.query_event_id === "string"
        ? payload.query_event_id
        : null,
    selected_rank: Math.floor(selectedRank),
    selected_template_id:
      "selected_template_id" in payload && typeof payload.selected_template_id === "string"
        ? payload.selected_template_id
        : null,
  });

  return Response.json({ ok: true }, { headers: { "cache-control": "no-store" } });
}
