import { after, before, beforeEach, test } from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { assertFails, assertSucceeds, initializeTestEnvironment } from "@firebase/rules-unit-testing";
import { collection, deleteDoc, doc, documentId, getDoc, getDocs, orderBy, query,
  serverTimestamp, setDoc, Timestamp, updateDoc, where } from "firebase/firestore";

let env;
const path = "robots/robot/commands/request_one";
const viewer = (uid = "operator") => env.authenticatedContext(uid,
  { email: "admin1@example.invalid", email_verified: true }).firestore();
const sender = () => env.authenticatedContext("sender").firestore();
const request = (ownerUid = "operator", action = "START") => ({
  schemaVersion: 1, robotId: "robot", ownerUid, action, state: "requested",
  createdAt: serverTimestamp(), acceptedAt: null, finishedAt: null, resultCode: null,
});
// 전환 기간의 예전 웹 형식: 브라우저 시계로 만든 expiresAt이 함께 있다.
const legacyRequest = (offsetMs = 10_000) => ({
  ...request(), expiresAt: Timestamp.fromMillis(Date.now() + offsetMs),
});

before(async () => {
  env = await initializeTestEnvironment({ projectId: "demo-rail-remote-control",
    firestore: { host: "127.0.0.1", port: 8087,
      rules: await readFile(new URL("../firestore.rules", import.meta.url), "utf8") } });
});
beforeEach(async () => {
  await env.clearFirestore();
  await env.withSecurityRulesDisabled(async (context) => {
    const db = context.firestore();
    await setDoc(doc(db, "robots/robot"), { enabled: true, senderId: "sender",
      viewerUids: ["operator", "viewer"], operatorUids: ["operator"], remoteControlEnabled: true });
    await setDoc(doc(db, "device_uploaders/sender"), { enabled: true });
  });
});
after(async () => { await env?.cleanup(); });

test("only an assigned operator creates a short-lived command", async () => {
  await assertSucceeds(setDoc(doc(viewer(), path), request()));
  await assertSucceeds(getDoc(doc(viewer(), path)));
  await assertSucceeds(getDoc(doc(sender(), path)));
  await assertFails(getDoc(doc(viewer("viewer"), path)));
  await assertFails(setDoc(doc(viewer("viewer"), "robots/robot/commands/other"), request("viewer")));
  await assertFails(setDoc(doc(viewer(), "robots/robot/commands/forged"), request("someone_else")));
  // 전환 기간이 끝나 브라우저 시계로 만든 expiresAt이 있는 예전 형식은 거부한다.
  await assertFails(setDoc(doc(viewer(), "robots/robot/commands/legacy"), legacyRequest()));
  await assertFails(setDoc(doc(viewer(), "robots/robot/commands/long_ttl"), legacyRequest(60_000)));
  await assertFails(setDoc(doc(viewer(), "robots/robot/commands/unknown"), request("operator", "SAFE_OFF")));
  await assertFails(setDoc(doc(viewer(), "robots/robot/commands/extra"), { ...request(), extra: true }));
});

test("command lifetime is server createdAt + 10 s, independent of the browser clock", async () => {
  await env.withSecurityRulesDisabled(async (context) => {
    const db = context.firestore();
    const base = { ...request(), createdAt: Timestamp.fromMillis(Date.now() - 30_000) };
    await setDoc(doc(db, "robots/robot/commands/stale_new"), base);
    await setDoc(doc(db, "robots/robot/commands/stale_legacy"),
      { ...base, expiresAt: Timestamp.fromMillis(Date.now() - 20_000) });
    await setDoc(doc(db, "robots/robot/commands/fresh_new"),
      { ...request(), createdAt: Timestamp.fromMillis(Date.now() - 2_000) });
  });
  for (const id of ["stale_new", "stale_legacy"]) {
    await assertFails(updateDoc(doc(sender(), "robots/robot/commands/" + id), {
      state: "accepted", acceptedAt: serverTimestamp(),
    }));
  }
  await assertSucceeds(updateDoc(doc(sender(), "robots/robot/commands/fresh_new"), {
    state: "accepted", acceptedAt: serverTimestamp(),
  }));
});

test("only sender acknowledges a command and history cannot be rewritten or deleted", async () => {
  await assertSucceeds(setDoc(doc(viewer(), path), request("operator", "E_STOP")));
  await assertFails(updateDoc(doc(viewer(), path), { state: "succeeded" }));
  await assertSucceeds(updateDoc(doc(sender(), path), { state: "accepted", acceptedAt: serverTimestamp() }));
  await assertFails(updateDoc(doc(sender(), path), { action: "START" }));
  await assertSucceeds(updateDoc(doc(sender(), path), {
    state: "succeeded", finishedAt: serverTimestamp(), resultCode: "estop_ack",
  }));
  await assertFails(updateDoc(doc(sender(), path), { state: "requested" }));
  await assertFails(deleteDoc(doc(sender(), path)));
  await assertFails(deleteDoc(doc(viewer(), path)));
});

test("remote control is denied unless robot configuration explicitly enables it", async () => {
  await env.withSecurityRulesDisabled(async (context) => {
    await updateDoc(doc(context.firestore(), "robots/robot"), { remoteControlEnabled: false });
  });
  await assertFails(setDoc(doc(viewer(), path), request()));
});

test("sender can query priority STOP commands while a viewer cannot list command history", async () => {
  await assertSucceeds(setDoc(doc(viewer(), path), request("operator", "E_STOP")));
  const commandQuery = (db) => query(collection(db, "robots/robot/commands"),
    where("action", "==", "E_STOP"), where("createdAt", ">=", Timestamp.fromMillis(Date.now() - 60_000)),
    orderBy("createdAt", "desc"), orderBy(documentId(), "desc"));
  const results = await assertSucceeds(getDocs(commandQuery(sender())));
  assert.equal(results.docs.length, 1);
  await assertFails(getDocs(commandQuery(viewer())));
});
