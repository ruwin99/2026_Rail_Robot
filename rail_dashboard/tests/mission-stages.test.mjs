import assert from "node:assert/strict";
import test from "node:test";
import { missionStages, missionStageState } from "../firebase-client/mission-stages.ts";

test("marks earlier FSM stages done, the current stage active, and later stages pending", () => {
  assert.deepEqual(missionStages.map((_, index) => missionStageState("RESCAN", true, index)),
    ["done", "done", "done", "done", "active", "pending", "pending"]);
  assert.deepEqual(missionStages.map((_, index) => missionStageState("COMPLETE", true, index)),
    Array(missionStages.length).fill("done"));
});

test("does not claim progress for missing, stale, failed, or unknown mission states", () => {
  for (const [phase, fresh] of [[undefined, true], ["RESCAN", false], ["FAILED", true], ["UNKNOWN", true]]) {
    assert.deepEqual(missionStages.map((_, index) => missionStageState(phase, fresh, index)),
      Array(missionStages.length).fill("unknown"));
  }
});
