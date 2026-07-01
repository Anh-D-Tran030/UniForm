export const MAX_UPLOAD_BYTES = 50 * 1024 * 1024;

export const ALLOWED_IMAGE_EXTENSIONS = new Set([".png", ".jpg", ".jpeg", ".tif", ".tiff"]);

export const ALLOWED_IMAGE_TYPES = new Set([
  "image/jpeg",
  "image/png",
  "image/tiff",
]);

export function extensionFromName(filename: string) {
  const index = filename.lastIndexOf(".");
  return index >= 0 ? filename.slice(index).toLowerCase() : "";
}

export function isAllowedImageFile(file: File) {
  const extension = extensionFromName(file.name);
  return ALLOWED_IMAGE_TYPES.has(file.type) || ALLOWED_IMAGE_EXTENSIONS.has(extension);
}

export function uploadLimitLabel() {
  return "50MB";
}

export function validateImageFile(input: FormDataEntryValue | null) {
  if (!(input instanceof File)) {
    return "Missing image file";
  }

  if (input.size <= 0) {
    return "Uploaded image is empty";
  }

  if (input.size > MAX_UPLOAD_BYTES) {
    return `Uploaded image is too large. Maximum size is ${uploadLimitLabel()}.`;
  }

  if (!isAllowedImageFile(input)) {
    return "Unsupported file type. Upload a PNG, JPG, JPEG, TIF, or TIFF image.";
  }

  return null;
}
