export type NodeHealthStatus = "ok" | "observed" | "fault" | "stale" | "unknown";
export type NodeHealthSignal = { id: string; status: NodeHealthStatus;
  observedAt: string | null; detail: string };

export const nodeHealthItems = [
  { id: "camera_front", label: "정면 카메라" },
  { id: "camera_side", label: "옆면 카메라" },
  { id: "vision", label: "영상 분석" },
  { id: "mission", label: "임무 제어" },
  { id: "stm_bridge", label: "STM 통신" },
  { id: "encoder_left", label: "왼쪽 엔코더" },
  { id: "encoder_right", label: "오른쪽 엔코더" },
  { id: "cleaner_front", label: "정면 클리너" },
  { id: "cleaner_side", label: "옆면 클리너" },
  { id: "pump_front", label: "정면 워터펌프" },
  { id: "pump_side", label: "옆면 워터펌프" },
] as const;

function fresh(time: string | null | undefined, now: number) {
  const age = typeof time === "string" ? now - Date.parse(time) : NaN;
  return Number.isFinite(age) && age >= -5_000 && age < 10_000;
}

export function resolveNodeHealth(signals: NodeHealthSignal[] | undefined,
  telemetryObservedAt: string | null, now: number) {
  const byId = new Map((Array.isArray(signals) ? signals : [])
    .filter((item): item is NodeHealthSignal => !!item && typeof item.id === "string")
    .map((item) => [item.id, item]));
  return nodeHealthItems.map((item) => {
    const signal = byId.get(item.id);
    let status: NodeHealthStatus = "unknown";
    if (signal && ["ok", "observed", "fault", "stale"].includes(signal.status) && typeof signal.observedAt === "string") {
      status = fresh(telemetryObservedAt, now) && fresh(signal.observedAt, now) ? signal.status : "stale";
    }
    return { ...item, status };
  });
}
