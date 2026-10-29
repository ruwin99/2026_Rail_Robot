import assert from "node:assert/strict";
import test from "node:test";
import { nodeHealthItems, resolveNodeHealth } from "../firebase-client/node-health.ts";

test("uses readable equipment names including the encoder, cleaner and water pump", () => {
  const labels = Object.fromEntries(nodeHealthItems.map((item) => [item.id, item.label]));
  assert.equal(labels.encoder_left, "왼쪽 엔코더");
  assert.equal(labels.cleaner_front, "정면 클리너");
  assert.equal(labels.pump_side, "옆면 워터펌프");
});

test("shows only fresh direct evidence as healthy and never treats missing telemetry as healthy", () => {
  const now = Date.parse("2026-09-26T08:00:00Z");
  const items = [{ id: "encoder_left", status: "observed", observedAt: "2026-09-26T07:59:58Z", detail: "shared_feedback" },
    { id: "pump_side", status: "unknown", observedAt: null, detail: "no_feedback" }];
  const fresh = resolveNodeHealth(items, "2026-09-26T07:59:59Z", now);
  assert.equal(fresh.find((item) => item.id === "encoder_left")?.status, "observed");
  assert.equal(fresh.find((item) => item.id === "pump_side")?.status, "unknown");
  assert.equal(fresh.find((item) => item.id === "cleaner_front")?.status, "unknown");
  const stale = resolveNodeHealth(items, "2026-09-26T07:59:30Z", now);
  assert.equal(stale.find((item) => item.id === "encoder_left")?.status, "stale");
  assert.equal(resolveNodeHealth(undefined, null, now).every((item) => item.status === "unknown"), true);
});

test("keeps fresh faults visible but marks old fault reports as delayed", () => {
  const now = Date.parse("2026-09-26T08:00:00Z");
  const fault = [{ id: "stm_bridge", status: "fault", observedAt: "2026-09-26T07:59:58Z", detail: "uart_error" }];
  assert.equal(resolveNodeHealth(fault, "2026-09-26T07:59:59Z", now)
    .find((item) => item.id === "stm_bridge")?.status, "fault");
  assert.equal(resolveNodeHealth(fault, "2026-09-26T07:59:59Z", now + 20_000)
    .find((item) => item.id === "stm_bridge")?.status, "stale");
});

test("ignores malformed optional health entries instead of crashing the dashboard", () => {
  const result = resolveNodeHealth([null, { id: "encoder_left", status: "ok", observedAt: 42 }],
    "2026-09-26T07:59:59Z", Date.parse("2026-09-26T08:00:00Z"));
  assert.equal(result.find((item) => item.id === "encoder_left")?.status, "unknown");
});
