import { mkdir, readFile, writeFile } from "node:fs/promises";
import os from "node:os";
import path from "node:path";

type QueryMetricEvent = {
  event_id: string;
  event_type: "query";
  file_name: string | null;
  file_size: number | null;
  latency_ms: number;
  match_count: number;
  success: boolean;
  timestamp: string;
  top_k: number;
  top_scores: number[];
};

type SelectionMetricEvent = {
  event_type: "selection";
  query_event_id: string | null;
  selected_rank: number;
  selected_template_id: string | null;
  timestamp: string;
};

type StorageMetricEvent = {
  event_type: "storage";
  run_id: string | null;
  success: boolean;
  template_id: string | null;
  timestamp: string;
};

type PrecomputeMetricEvent = {
  event_type: "precompute";
  latency_ms: number;
  run_id: string | null;
  status: string;
  success: boolean;
  timestamp: string;
  upload_id: string | null;
};

type MetricEvent = QueryMetricEvent | SelectionMetricEvent | StorageMetricEvent | PrecomputeMetricEvent;

type MetricsFile = {
  events: MetricEvent[];
};

const metricsDir = path.join(os.tmpdir(), "odc-next-ui-metrics");
const metricsPath = path.join(metricsDir, "metrics.json");

async function readMetricsFile(): Promise<MetricsFile> {
  try {
    const raw = await readFile(metricsPath, "utf-8");
    const parsed = JSON.parse(raw) as Partial<MetricsFile>;
    return { events: Array.isArray(parsed.events) ? (parsed.events as MetricEvent[]) : [] };
  } catch {
    return { events: [] };
  }
}

async function writeMetricsFile(data: MetricsFile) {
  await mkdir(metricsDir, { recursive: true });
  await writeFile(metricsPath, JSON.stringify(data, null, 2), "utf-8");
}

async function appendEvent(event: MetricEvent) {
  const data = await readMetricsFile();
  data.events.push(event);
  await writeMetricsFile(data);
}

export async function recordQueryEvent(event: Omit<QueryMetricEvent, "event_type" | "timestamp">) {
  await appendEvent({
    ...event,
    event_type: "query",
    timestamp: new Date().toISOString(),
  });
}

export async function recordSelectionEvent(event: Omit<SelectionMetricEvent, "event_type" | "timestamp">) {
  await appendEvent({
    ...event,
    event_type: "selection",
    timestamp: new Date().toISOString(),
  });
}

export async function recordStorageEvent(event: Omit<StorageMetricEvent, "event_type" | "timestamp">) {
  await appendEvent({
    ...event,
    event_type: "storage",
    timestamp: new Date().toISOString(),
  });
}

export async function recordPrecomputeEvent(event: Omit<PrecomputeMetricEvent, "event_type" | "timestamp">) {
  await appendEvent({
    ...event,
    event_type: "precompute",
    timestamp: new Date().toISOString(),
  });
}

function average(values: number[]) {
  if (!values.length) {
    return 0;
  }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

export async function getDashboardMetrics() {
  const { events } = await readMetricsFile();
  const queryEvents = events.filter((event): event is QueryMetricEvent => event.event_type === "query");
  const selectionEvents = events.filter((event): event is SelectionMetricEvent => event.event_type === "selection");
  const storageEvents = events.filter((event): event is StorageMetricEvent => event.event_type === "storage");
  const precomputeEvents = events.filter((event): event is PrecomputeMetricEvent => event.event_type === "precompute");
  const successfulQueries = queryEvents.filter((event) => event.success);
  const failedQueries = queryEvents.filter((event) => !event.success);
  const successfulStorage = storageEvents.filter((event) => event.success);
  const failedStorage = storageEvents.filter((event) => !event.success);
  const successfulPrecompute = precomputeEvents.filter((event) => event.success);
  const failedPrecompute = precomputeEvents.filter((event) => !event.success);

  return {
    total_queries: queryEvents.length,
    successful_queries: successfulQueries.length,
    failed_queries: failedQueries.length,
    average_query_latency_ms: Math.round(average(successfulQueries.map((event) => event.latency_ms))),
    average_returned_match_count: Number(average(successfulQueries.map((event) => event.match_count)).toFixed(1)),
    average_top1_similarity_score: Number(
      average(successfulQueries.map((event) => event.top_scores[0]).filter((score) => typeof score === "number")).toFixed(4),
    ),
    selected_rank_distribution: {
      top_1: selectionEvents.filter((event) => event.selected_rank === 1).length,
      top_2: selectionEvents.filter((event) => event.selected_rank === 2).length,
      top_3: selectionEvents.filter((event) => event.selected_rank === 3).length,
      top_4_plus: selectionEvents.filter((event) => event.selected_rank >= 4).length,
    },
    average_precompute_latency_ms: Math.round(average(successfulPrecompute.map((event) => event.latency_ms))),
    failed_precompute_count: failedPrecompute.length,
    successful_precompute_count: successfulPrecompute.length,
    storage_success_count: successfulStorage.length,
    storage_failure_count: failedStorage.length,
    total_storage_events: storageEvents.length,
    recent_events: events.slice(-12).reverse(),
  };
}
