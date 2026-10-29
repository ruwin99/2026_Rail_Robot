export const missionStages = [
  { phase: "CAPTURE_SCAN", label: "초기 구간 검사" },
  { phase: "RETURN", label: "시작점 복귀" },
  { phase: "REALTIME", label: "청소 및 윤활" },
  { phase: "RESCAN_RETURN", label: "재검사 시작점 복귀" },
  { phase: "RESCAN", label: "구간 재검사" },
  { phase: "RESCAN_DONE", label: "재검사 종료 확인" },
  { phase: "COMPLETE", label: "임무 종료 확인" },
] as const;

export function missionStageState(phase: string | undefined, fresh: boolean, index: number) {
  const current = missionStages.findIndex((stage) => stage.phase === phase);
  if (!fresh || current < 0) return "unknown";
  if (phase === "COMPLETE" || index < current) return "done";
  return index === current ? "active" : "pending";
}
