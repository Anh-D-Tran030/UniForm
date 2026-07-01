import { NextResponse } from "next/server";
import { serviceBaseUrl } from "@/lib/backend";
import { createSessionToken, SESSION_COOKIE, SESSION_MAX_AGE } from "@/lib/auth";

export const runtime = "nodejs";

export async function POST(request: Request) {
  const body = await request.json().catch(() => null);
  const username = (body?.username ?? "").toString().trim();
  const password = (body?.password ?? "").toString();

  if (!username || !password) {
    return NextResponse.json(
      { detail: "Username and password are required" },
      { status: 400 },
    );
  }

  let upstream: Response;
  try {
    upstream = await fetch(`${serviceBaseUrl("auth")}/login`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ username, password }),
      cache: "no-store",
    });
  } catch {
    return NextResponse.json(
      { detail: "The authentication service is unavailable. Is AuthService running on port 8008?" },
      { status: 502 },
    );
  }

  if (!upstream.ok) {
    const data = await upstream.json().catch(() => ({}));
    const detail =
      typeof data?.detail === "string" ? data.detail : "Invalid username or password";
    const status = upstream.status === 401 || upstream.status === 403 ? 401 : 400;
    return NextResponse.json({ detail }, { status });
  }

  const data = await upstream.json().catch(() => ({}));
  const resolvedUsername = typeof data?.username === "string" ? data.username : username;

  const token = await createSessionToken(resolvedUsername);
  const response = NextResponse.json({
    ok: true,
    username: resolvedUsername,
    display_name: data?.display_name ?? null,
  });
  response.cookies.set(SESSION_COOKIE, token, {
    httpOnly: true,
    sameSite: "lax",
    path: "/",
    maxAge: SESSION_MAX_AGE,
  });
  return response;
}
