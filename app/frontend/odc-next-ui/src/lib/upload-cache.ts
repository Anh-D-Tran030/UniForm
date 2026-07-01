import { mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { randomUUID } from "node:crypto";

type UploadMetadata = {
  contentType: string;
  filename: string;
};

export type CachedUpload = UploadMetadata & {
  buffer: Buffer;
  uploadId: string;
};

const CACHE_DIR = path.join(os.tmpdir(), "odc-next-ui-uploads");

function assertSafeUploadId(uploadId: string) {
  if (!/^[A-Za-z0-9_-]+$/.test(uploadId)) {
    throw new Error("Invalid upload id");
  }
}

function filePath(uploadId: string) {
  assertSafeUploadId(uploadId);
  return path.join(CACHE_DIR, `${uploadId}.bin`);
}

function metadataPath(uploadId: string) {
  assertSafeUploadId(uploadId);
  return path.join(CACHE_DIR, `${uploadId}.json`);
}

export async function saveUploadFile(file: File) {
  await mkdir(CACHE_DIR, { recursive: true });

  const uploadId = randomUUID();
  const buffer = Buffer.from(await file.arrayBuffer());
  const metadata: UploadMetadata = {
    contentType: file.type || "application/octet-stream",
    filename: file.name || "upload.png",
  };

  await writeFile(filePath(uploadId), buffer);
  await writeFile(metadataPath(uploadId), JSON.stringify(metadata), "utf-8");

  return {
    ...metadata,
    size: buffer.length,
    uploadId,
  };
}

export async function readCachedUpload(uploadId: string): Promise<CachedUpload> {
  const [buffer, metadataRaw] = await Promise.all([
    readFile(filePath(uploadId)),
    readFile(metadataPath(uploadId), "utf-8"),
  ]);
  const metadata = JSON.parse(metadataRaw) as UploadMetadata;

  return {
    ...metadata,
    buffer,
    uploadId,
  };
}

export function appendCachedUpload(formData: FormData, upload: CachedUpload) {
  formData.append(
    "image",
    new Blob([new Uint8Array(upload.buffer)], { type: upload.contentType }),
    upload.filename,
  );
}
