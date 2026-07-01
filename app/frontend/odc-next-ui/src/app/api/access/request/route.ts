import { NextResponse } from "next/server";
import { serviceBaseUrl } from "@/lib/backend";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const name = (body?.name ?? "").toString().trim();
  const email = (body?.email ?? "").toString().trim();

  if (!name || !email) {
    return NextResponse.json({ detail: "Name and email are required" }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${serviceBaseUrl("auth")}/access-requests`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ name, email }),
      cache: "no-store",
    });
  } catch {
    return NextResponse.json(
      { detail: "We couldn't reach the service. Please try again shortly." },
      { status: 502 },
    );
  }

  if (!upstream.ok) {
    const data = await upstream.json().catch(() => ({}));
    const detail =
      typeof data?.detail === "string" ? data.detail : "Unable to submit your request.";
    const status = upstream.status >= 400 && upstream.status < 500 ? 400 : 502;
    return NextResponse.json({ detail }, { status });
  }

  return NextResponse.json({ ok: true });
}
