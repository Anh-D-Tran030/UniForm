const DEFAULT_BACKEND_URL = "http://127.0.0.1:8005";
const DEFAULT_KV_BACKEND_URL = "http://127.0.0.1:8006";
const DEFAULT_STORAGE_BACKEND_URL = "http://127.0.0.1:8007";
const DEFAULT_AUTH_BACKEND_URL = "http://127.0.0.1:8008";
const DEFAULT_GOLD_BACKEND_URL = "http://127.0.0.1:8009";

export type ServiceName = "odc" | "kv" | "storage" | "auth" | "gold";

export function serviceBaseUrl(service: ServiceName) {
  if (service === "kv") {
    return (process.env.KV_BACKEND_URL ?? DEFAULT_KV_BACKEND_URL).replace(/\/$/, "");
  }
  if (service === "storage") {
    return (process.env.STORAGE_BACKEND_URL ?? DEFAULT_STORAGE_BACKEND_URL).replace(/\/$/, "");
  }
  if (service === "auth") {
    return (process.env.AUTH_BACKEND_URL ?? DEFAULT_AUTH_BACKEND_URL).replace(/\/$/, "");
  }
  if (service === "gold") {
    return (process.env.GOLD_BACKEND_URL ?? DEFAULT_GOLD_BACKEND_URL).replace(/\/$/, "");
  }

  return (process.env.ODC_BACKEND_URL ?? DEFAULT_BACKEND_URL).replace(/\/$/, "");
}

function buildBackendUrl(service: ServiceName, path: string, sourceUrl?: string) {
  const backendUrl = new URL(`${serviceBaseUrl(service)}${path}`);

  if (sourceUrl) {
    const source = new URL(sourceUrl);
    source.searchParams.forEach((value, key) => {
      backendUrl.searchParams.set(key, value);
    });
  }

  return backendUrl;
}

async function relayBackendResponse(response: Response) {
  const contentType = response.headers.get("content-type") ?? "application/json";
  const body = await response.arrayBuffer();

  return new Response(body, {
    status: response.status,
    headers: {
      "cache-control": "no-store",
      "content-type": contentType,
    },
  });
}

function backendErrorResponse(error: unknown) {
  const detail =
    error instanceof Error ? error.message : "Unable to reach the backend service";

  return Response.json({ detail }, { status: 502 });
}

export async function proxyRequest(
  path: string,
  init: RequestInit,
  sourceUrl?: string,
) {
  return proxyServiceRequest("odc", path, init, sourceUrl);
}

export async function proxyServiceRequest(
  service: ServiceName,
  path: string,
  init: RequestInit,
  sourceUrl?: string,
) {
  try {
    const response = await fetch(buildBackendUrl(service, path, sourceUrl), {
      ...init,
      cache: "no-store",
    });

    return relayBackendResponse(response);
  } catch (error) {
    return backendErrorResponse(error);
  }
}

export async function proxyMultipartRequest(
  request: Request,
  path: string,
  sourceUrl?: string,
) {
  const formData = await request.formData();

  return proxyRequest(
    path,
    {
      method: "POST",
      body: formData,
    },
    sourceUrl,
  );
}

export async function proxyFormDataToService(
  service: ServiceName,
  path: string,
  formData: FormData,
  sourceUrl?: string,
) {
  try {
    const response = await fetch(buildBackendUrl(service, path, sourceUrl), {
      method: "POST",
      body: formData,
      cache: "no-store",
    });

    return relayBackendResponse(response);
  } catch (error) {
    return backendErrorResponse(error);
  }
}
