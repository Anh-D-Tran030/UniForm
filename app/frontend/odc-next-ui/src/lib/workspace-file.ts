import path from "node:path";

const WORKSPACE_ROOT = path.resolve(/* turbopackIgnore: true */ process.cwd(), "..");

export function resolveWorkspacePath(input: string) {
  const candidate = path.isAbsolute(input)
    ? path.resolve(input)
    : path.resolve(WORKSPACE_ROOT, input);

  if (!candidate.startsWith(WORKSPACE_ROOT)) {
    return null;
  }

  return candidate;
}
