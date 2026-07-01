// Stateless session cookie signed with HMAC-SHA256 via the Web Crypto API.
// Web Crypto is available in both the Node.js (route handler) and Edge
// (middleware) runtimes, so the same code verifies tokens in both places —
// no extra npm dependency required.

const SECRET = process.env.AUTH_SECRET ?? "uniform-dev-secret-change-me";

export const SESSION_COOKIE = "uniform_session";
export const SESSION_MAX_AGE = 60 * 60 * 8; // 8 hours, in seconds

export type SessionPayload = {
  username: string;
  exp: number; // unix seconds
};

const encoder = new TextEncoder();
const decoder = new TextDecoder();

function base64urlEncode(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

function base64urlDecode(value: string): Uint8Array {
  let normalized = value.replace(/-/g, "+").replace(/_/g, "/");
  while (normalized.length % 4) normalized += "=";
  const binary = atob(normalized);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

function toArrayBuffer(bytes: Uint8Array): ArrayBuffer {
  const buffer = new ArrayBuffer(bytes.byteLength);
  new Uint8Array(buffer).set(bytes);
  return buffer;
}

async function getKey(): Promise<CryptoKey> {
  return crypto.subtle.importKey(
    "raw",
    encoder.encode(SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign", "verify"],
  );
}

export async function createSessionToken(username: string): Promise<string> {
  const payload: SessionPayload = {
    username,
    exp: Math.floor(Date.now() / 1000) + SESSION_MAX_AGE,
  };
  const payloadPart = base64urlEncode(encoder.encode(JSON.stringify(payload)));
  const key = await getKey();
  const signature = await crypto.subtle.sign("HMAC", key, encoder.encode(payloadPart));
  return `${payloadPart}.${base64urlEncode(new Uint8Array(signature))}`;
}

export async function verifySessionToken(
  token: string | undefined | null,
): Promise<SessionPayload | null> {
  if (!token) return null;
  const [payloadPart, signaturePart] = token.split(".");
  if (!payloadPart || !signaturePart) return null;

  const key = await getKey();
  let valid = false;
  try {
    valid = await crypto.subtle.verify(
      "HMAC",
      key,
      toArrayBuffer(base64urlDecode(signaturePart)),
      encoder.encode(payloadPart),
    );
  } catch {
    return null;
  }
  if (!valid) return null;

  try {
    const payload = JSON.parse(decoder.decode(base64urlDecode(payloadPart))) as SessionPayload;
    if (typeof payload.exp !== "number" || payload.exp < Math.floor(Date.now() / 1000)) {
      return null;
    }
    return payload;
  } catch {
    return null;
  }
}
