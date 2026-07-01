"use client";

import { Loader2 } from "lucide-react";
import { type CSSProperties, useMemo, useRef, useState } from "react";

export type KeyValueOverlayPair = {
  key: string;
  key_bbox: number[];
  pairId: string;
  score: number;
  value: string;
  value_bbox: number[];
};

type NaturalSize = {
  height: number;
  width: number;
};

type PctBox = {
  height: number;
  left: number;
  top: number;
  width: number;
};

type RenderablePair = KeyValueOverlayPair & {
  bounds: PctBox | null;
  keyBox: PctBox | null;
  valueBox: PctBox | null;
};

type KeyValueOverlayViewerProps = {
  activePairId: string | null;
  imageSrc: string | null;
  loading: boolean;
  onActivePairChange: (pairId: string | null) => void;
  onPinnedPairChange: (pairId: string | null) => void;
  pairs: KeyValueOverlayPair[];
  pinnedPairId: string | null;
};

type BoxColor = {
  border: string;
  fill: string;
  fillActive: string;
};

const KEY_COLOR: BoxColor = {
  border: "#22c55e", // green-500
  fill: "rgba(134, 239, 172, 0.30)", // light green
  fillActive: "rgba(134, 239, 172, 0.55)",
};

const VALUE_COLOR: BoxColor = {
  border: "#eab308", // yellow-500
  fill: "rgba(254, 240, 138, 0.34)", // soft yellow
  fillActive: "rgba(253, 224, 71, 0.6)",
};

// Extra breathing room (in screen px) around the blue pair bounding box so it
// doesn't hug the key/value boxes too tightly.
const BOUNDS_PAD = 8;

function clamp(value: number, min: number, max: number) {
  return Math.min(Math.max(value, min), max);
}

// Convert a [x0, y0, x1, y1] bbox in natural-image pixels into a box expressed
// as percentages of the natural image. Percentages are resolution-independent,
// so the overlay stays aligned no matter what size the image is displayed at.
function pctBox(bbox: number[] | undefined, natural: NaturalSize): PctBox | null {
  if (!bbox || bbox.length !== 4 || !natural.width || !natural.height) {
    return null;
  }

  const [left, top, right, bottom] = bbox.map(Number);
  if (![left, top, right, bottom].every(Number.isFinite)) {
    return null;
  }

  const x0 = clamp(Math.min(left, right), 0, natural.width);
  const x1 = clamp(Math.max(left, right), 0, natural.width);
  const y0 = clamp(Math.min(top, bottom), 0, natural.height);
  const y1 = clamp(Math.max(top, bottom), 0, natural.height);

  if (x1 - x0 <= 0 || y1 - y0 <= 0) {
    return null;
  }

  return {
    height: ((y1 - y0) / natural.height) * 100,
    left: (x0 / natural.width) * 100,
    top: (y0 / natural.height) * 100,
    width: ((x1 - x0) / natural.width) * 100,
  };
}

function unionBox(boxes: (PctBox | null)[]): PctBox | null {
  const present = boxes.filter((box): box is PctBox => Boolean(box));
  if (!present.length) {
    return null;
  }

  const left = Math.min(...present.map((box) => box.left));
  const top = Math.min(...present.map((box) => box.top));
  const right = Math.max(...present.map((box) => box.left + box.width));
  const bottom = Math.max(...present.map((box) => box.top + box.height));

  return { height: bottom - top, left, top, width: right - left };
}

function boxStyle(box: PctBox, active: boolean, color: BoxColor): CSSProperties {
  return {
    backgroundColor: active ? color.fillActive : color.fill,
    borderColor: color.border,
    boxShadow: active ? `0 0 0 2px ${color.border}` : undefined,
    height: `${box.height}%`,
    left: `${box.left}%`,
    minHeight: 8,
    minWidth: 8,
    top: `${box.top}%`,
    width: `${box.width}%`,
  };
}

function tooltipStyle(bounds: PctBox): CSSProperties {
  const above = bounds.top > 18;

  return {
    left: `${clamp(bounds.left, 0, 92)}%`,
    maxWidth: 260,
    top: above ? `${bounds.top}%` : `${bounds.top + bounds.height}%`,
    transform: above ? "translateY(calc(-100% - 8px))" : "translateY(8px)",
  };
}

function scorePercent(score: number) {
  if (!Number.isFinite(score)) {
    return "--";
  }

  return `${Math.max(0, Math.min(100, Math.round(score * 100)))}%`;
}

export function KeyValueOverlayViewer({
  activePairId,
  imageSrc,
  loading,
  onActivePairChange,
  onPinnedPairChange,
  pairs,
  pinnedPairId,
}: KeyValueOverlayViewerProps) {
  const imageRef = useRef<HTMLImageElement>(null);
  const [natural, setNatural] = useState<NaturalSize>({ height: 0, width: 0 });
  const visiblePairId = activePairId ?? pinnedPairId;

  const renderablePairs = useMemo<RenderablePair[]>(() => {
    if (!natural.width || !natural.height) {
      return [];
    }

    return pairs
      .map((pair) => {
        const keyBox = pctBox(pair.key_bbox, natural);
        const valueBox = pctBox(pair.value_bbox, natural);

        return {
          ...pair,
          bounds: unionBox([keyBox, valueBox]),
          keyBox,
          valueBox,
        };
      })
      .filter((pair) => pair.keyBox || pair.valueBox);
  }, [natural, pairs]);

  const visiblePair = renderablePairs.find((pair) => pair.pairId === visiblePairId) ?? null;

  if (loading) {
    return (
      <div className="flex h-[720px] items-center justify-center bg-[var(--surface-elevated)] p-4">
        <Loader2 size={28} className="animate-spin text-[var(--accent)]" />
      </div>
    );
  }

  if (!imageSrc) {
    return (
      <div className="flex h-[720px] items-center justify-center bg-[var(--surface-elevated)] p-4 text-sm text-[var(--muted)]">
        No image preview available
      </div>
    );
  }

  return (
    <div className="flex h-[720px] items-center justify-center bg-[var(--surface-elevated)] p-4">
      <div className="relative max-h-full max-w-full" onClick={() => onPinnedPairChange(null)}>
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img
          ref={imageRef}
          alt="Image preview with key-value boxes"
          src={imageSrc}
          className="block max-h-[688px] max-w-full object-contain"
          onLoad={() => {
            const image = imageRef.current;
            if (!image || !image.naturalWidth || !image.naturalHeight) {
              return;
            }

            setNatural({ height: image.naturalHeight, width: image.naturalWidth });
          }}
        />

        {natural.width && natural.height ? (
          <div className="pointer-events-none absolute inset-0">
            {renderablePairs.map((pair) => {
              const active = pair.pairId === visiblePairId;

              return (
                <div key={pair.pairId}>
                  {pair.bounds ? (
                    <div
                      className="absolute rounded-md border-2 transition"
                      style={{
                        borderColor: "var(--accent)",
                        borderStyle: active ? "solid" : "dashed",
                        height: `calc(${pair.bounds.height}% + ${BOUNDS_PAD * 2}px)`,
                        left: `calc(${pair.bounds.left}% - ${BOUNDS_PAD}px)`,
                        opacity: active ? 0.95 : 0.55,
                        top: `calc(${pair.bounds.top}% - ${BOUNDS_PAD}px)`,
                        width: `calc(${pair.bounds.width}% + ${BOUNDS_PAD * 2}px)`,
                      }}
                    />
                  ) : null}

                  {pair.keyBox ? (
                    <button
                      type="button"
                      aria-label={`Key ${pair.key}`}
                      className="pointer-events-auto absolute rounded-[3px] border transition"
                      style={boxStyle(pair.keyBox, active, KEY_COLOR)}
                      onClick={(event) => {
                        event.stopPropagation();
                        onPinnedPairChange(pinnedPairId === pair.pairId ? null : pair.pairId);
                      }}
                      onMouseEnter={() => onActivePairChange(pair.pairId)}
                      onMouseLeave={() => onActivePairChange(null)}
                      onPointerEnter={() => onActivePairChange(pair.pairId)}
                      onPointerLeave={() => onActivePairChange(null)}
                    />
                  ) : null}

                  {pair.valueBox ? (
                    <button
                      type="button"
                      aria-label={`Value ${pair.value}`}
                      className="pointer-events-auto absolute rounded-[3px] border transition"
                      style={boxStyle(pair.valueBox, active, VALUE_COLOR)}
                      onClick={(event) => {
                        event.stopPropagation();
                        onPinnedPairChange(pinnedPairId === pair.pairId ? null : pair.pairId);
                      }}
                      onMouseEnter={() => onActivePairChange(pair.pairId)}
                      onMouseLeave={() => onActivePairChange(null)}
                      onPointerEnter={() => onActivePairChange(pair.pairId)}
                      onPointerLeave={() => onActivePairChange(null)}
                    />
                  ) : null}
                </div>
              );
            })}

            {visiblePair && visiblePair.bounds ? (
              <div
                className="pointer-events-none absolute z-20 rounded-lg border border-[var(--line)] bg-[var(--surface)] px-3 py-2 text-xs text-[var(--ink-strong)] shadow-[var(--panel-shadow)]"
                style={tooltipStyle(visiblePair.bounds)}
              >
                <p className="font-semibold text-[var(--accent-strong)]">Key: {visiblePair.key || "--"}</p>
                <p className="mt-1 leading-5 text-[var(--ink-strong)]">Value: {visiblePair.value || "--"}</p>
                <p className="mt-1 text-[var(--muted)]">Score: {scorePercent(visiblePair.score)}</p>
              </div>
            ) : null}
          </div>
        ) : null}
      </div>
    </div>
  );
}
