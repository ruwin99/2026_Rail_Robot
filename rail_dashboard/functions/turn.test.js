import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import test from "node:test";
import { TURN_TTL_SECONDS, TurnRequestError, authorizeViewer, turnRestIceServers } from "./turn.js";

const viewer = { uid: "viewer-1", token: { email_verified: true } };
const robot = { enabled: true, viewerUids: ["viewer-1"] };
const secret = "0123456789abcdef0123456789abcdef0123456789abcdef";

function code(fn) {
  try { fn(); } catch (error) { return error instanceof TurnRequestError ? error.code : "other"; }
  return "ok";
}

test("only a verified viewer of an enabled robot gets credentials", () => {
  assert.equal(code(() => authorizeViewer(viewer, "rail-robot-01", robot)), "ok");
  assert.equal(code(() => authorizeViewer(null, "rail-robot-01", robot)), "unauthenticated");
  assert.equal(code(() => authorizeViewer(viewer, "../robots", robot)), "invalid-argument");
  assert.equal(code(() => authorizeViewer({ ...viewer, token: {} }, "rail-robot-01", robot)),
    "permission-denied");
  assert.equal(code(() => authorizeViewer({ ...viewer, uid: "other" }, "rail-robot-01", robot)),
    "permission-denied");
  assert.equal(code(() => authorizeViewer(viewer, "rail-robot-01", { ...robot, enabled: false })),
    "permission-denied");
  assert.equal(code(() => authorizeViewer(viewer, "rail-robot-01", undefined)), "permission-denied");
});

test("credentials follow the coturn TURN REST contract", () => {
  const [server] = turnRestIceServers({ secret, host: "34.64.10.20", uid: "viewer-1", nowSeconds: 1000.7 });
  assert.equal(server.username, `${1000 + TURN_TTL_SECONDS}:viewer-1`);
  assert.equal(server.credential, createHmac("sha1", secret).update(server.username).digest("base64"));
  assert.deepEqual(server.urls, [
    "turn:34.64.10.20:3478?transport=udp",
    "turn:34.64.10.20:3478?transport=tcp",
  ]);
});

test("missing or unsafe server settings never produce credentials", () => {
  const base = { secret, host: "34.64.10.20", uid: "viewer-1", nowSeconds: 0 };
  assert.equal(code(() => turnRestIceServers({ ...base, secret: "" })), "unavailable");
  assert.equal(code(() => turnRestIceServers({ ...base, secret: "short" })), "unavailable");
  assert.equal(code(() => turnRestIceServers({ ...base, host: "" })), "unavailable");
  assert.equal(code(() => turnRestIceServers({ ...base, host: "1.2.3.4?transport=tcp" })), "unavailable");
  assert.equal(code(() => turnRestIceServers({ ...base, host: "999.1.1.1" })), "unavailable");
  assert.equal(code(() => turnRestIceServers({ ...base, host: "turn.example.com" })), "ok");
});
