import { test } from "node:test";
import assert from "node:assert/strict";
import { selectVisibleCommand } from "../firebase-client/remote-command-selection.ts";

test("a late START response cannot replace a newer STOP request on the dashboard", () => {
  const stop = { robotId: "robot", id: "stop-1", action: "E_STOP", sentAt: 2000 };
  const start = { robotId: "robot", id: "start-1", action: "START", sentAt: 1000 };
  assert.deepEqual(selectVisibleCommand(stop, start), stop);
});

test("the dashboard can display a newer command after an earlier one", () => {
  const start = { robotId: "robot", id: "start-1", action: "START", sentAt: 1000 };
  const stop = { robotId: "robot", id: "stop-1", action: "E_STOP", sentAt: 2000 };
  assert.deepEqual(selectVisibleCommand(start, stop), stop);
});
