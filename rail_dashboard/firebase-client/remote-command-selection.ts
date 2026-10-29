export type VisibleCommand = {
  robotId: string; id: string; action: "START" | "E_STOP"; sentAt: number;
};

export function selectVisibleCommand(current: VisibleCommand | null, next: VisibleCommand): VisibleCommand {
  if (current?.robotId === next.robotId && current.action === "E_STOP"
    && next.action === "START" && current.sentAt >= next.sentAt) return current;
  return next;
}
