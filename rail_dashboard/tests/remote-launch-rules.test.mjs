import { after, before, beforeEach, test } from "node:test";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { assertFails, assertSucceeds, initializeTestEnvironment } from "@firebase/rules-unit-testing";
import { createMockUserToken } from "@firebase/util";
import { deleteDoc, doc, getDoc, serverTimestamp, setDoc, Timestamp, updateDoc } from "firebase/firestore";

let env;
const statusPath = "robots/robot/launcher/status";
const requestPath = "robots/robot/launchRequests/request_one";
const operator = (uid = "operator") => env.authenticatedContext(uid,
  { email: "admin1@example.invalid", email_verified: true }).firestore();
const sender = () => env.authenticatedContext("sender").firestore();
const launchRequest = (ownerUid = "operator") => ({
  schemaVersion: 1, robotId: "robot", ownerUid, action: "LAUNCH_MISSION", state: "requested",
  createdAt: serverTimestamp(), acceptedAt: null, finishedAt: null, resultCode: null,
});
// 전환 기간의 예전 웹 형식: 브라우저 시계로 만든 expiresAt이 함께 있다.
const legacyLaunchRequest = (offsetMs = 10_000) => ({
  ...launchRequest(), expiresAt: Timestamp.fromMillis(Date.now() + offsetMs),
});

before(async () => {
  env = await initializeTestEnvironment({ projectId: "demo-rail-launch",
    firestore: { host: "127.0.0.1", port: 8087,
      rules: await readFile(new URL("../firestore.rules", import.meta.url), "utf8") } });
});
beforeEach(async () => {
  await env.clearFirestore();
  await env.withSecurityRulesDisabled(async (context) => {
    const db = context.firestore();
    await setDoc(doc(db, "robots/robot"), { enabled: true, senderId: "sender",
      viewerUids: ["operator", "viewer"], operatorUids: ["operator"],
      remoteControlEnabled: true, remoteLaunchEnabled: true });
    await setDoc(doc(db, "device_uploaders/sender"), { enabled: true });
  });
});
after(async () => { await env?.cleanup(); });

test("only the designated sender publishes launcher status", async () => {
  await assertSucceeds(setDoc(doc(sender(), statusPath), {
    schemaVersion: 1, senderId: "sender", ready: true, missionRunning: false,
    updatedAt: serverTimestamp(),
  }));
  await assertSucceeds(getDoc(doc(operator(), statusPath)));
  await assertFails(setDoc(doc(operator(), statusPath), {
    schemaVersion: 1, senderId: "sender", ready: true, missionRunning: false,
    updatedAt: serverTimestamp(),
  }));
  await assertFails(updateDoc(doc(sender(), statusPath), { senderId: "someone_else" }));
});

test("launch needs separate opt-in and an assigned operator", async () => {
  await env.withSecurityRulesDisabled(async (context) => {
    await updateDoc(doc(context.firestore(), "robots/robot"), { remoteControlEnabled: false });
  });
  await assertSucceeds(setDoc(doc(operator(), requestPath), launchRequest()));
  await assertFails(setDoc(doc(operator("viewer"), "robots/robot/launchRequests/viewer"),
    launchRequest("viewer")));
  await assertFails(setDoc(doc(operator(), "robots/robot/launchRequests/forged"),
    launchRequest("someone_else")));
  // 전환 기간이 끝나 브라우저 시계로 만든 expiresAt이 있는 예전 형식은 거부한다.
  await assertFails(setDoc(doc(operator(), "robots/robot/launchRequests/legacy"),
    legacyLaunchRequest()));
  await assertFails(setDoc(doc(operator(), "robots/robot/launchRequests/long_ttl"),
    legacyLaunchRequest(60_000)));
  await env.withSecurityRulesDisabled(async (context) => {
    await updateDoc(doc(context.firestore(), "robots/robot"), { remoteLaunchEnabled: false });
  });
  await assertFails(setDoc(doc(operator(), "robots/robot/launchRequests/disabled"), launchRequest()));
});

test("sender alone acknowledges once and neither party rewrites completed requests", async () => {
  await assertSucceeds(setDoc(doc(operator(), requestPath), launchRequest()));
  await assertSucceeds(getDoc(doc(sender(), requestPath)));
  await assertFails(updateDoc(doc(operator(), requestPath), { state: "accepted" }));
  await assertSucceeds(updateDoc(doc(sender(), requestPath), {
    state: "accepted", acceptedAt: serverTimestamp(),
  }));
  await assertFails(updateDoc(doc(sender(), requestPath), { action: "START" }));
  await assertSucceeds(updateDoc(doc(sender(), requestPath), {
    state: "succeeded", finishedAt: serverTimestamp(), resultCode: "mission_running",
  }));
  await assertFails(updateDoc(doc(sender(), requestPath), { state: "requested" }));
  await assertFails(deleteDoc(doc(sender(), requestPath)));
});

test("launch lifetime is server createdAt + 10 s, independent of the browser clock", async () => {
  await env.withSecurityRulesDisabled(async (context) => {
    await setDoc(doc(context.firestore(), "robots/robot/launchRequests/stale"),
      { ...launchRequest(), createdAt: Timestamp.fromMillis(Date.now() - 30_000) });
  });
  await assertFails(updateDoc(doc(sender(), "robots/robot/launchRequests/stale"), {
    state: "accepted", acceptedAt: serverTimestamp(),
  }));
});

test("production Jetson launcher handles a virtual request through Firestore REST", async () => {
  await assertSucceeds(setDoc(doc(operator(), requestPath), launchRequest()));
  const python = process.env.RAIL_VIRTUAL_PYTHON || (process.platform === "win32" ? "python" : "python3");
  const probe = fileURLToPath(new URL("../../code/tests/mission_launcher_emulator_probe.py", import.meta.url));
  const output = execFileSync(python, [probe], {
    encoding: "utf8", timeout: 15_000,
    env: { ...process.env, FIRESTORE_EMULATOR_HOST: "127.0.0.1:8087",
      RAIL_VIRTUAL_PROJECT_ID: "demo-rail-launch", RAIL_VIRTUAL_ROBOT_ID: "robot",
      RAIL_VIRTUAL_SENDER_UID: "sender",
      RAIL_VIRTUAL_SENDER_TOKEN: createMockUserToken({ sub: "sender" }, "demo-rail-launch") },
  });
  if (!output.includes("mission_launcher_store_and_fake_service_ok")) {
    throw new Error("Virtual Jetson launch probe did not complete");
  }
  const result = (await assertSucceeds(getDoc(doc(operator(), requestPath)))).data();
  if (result.state !== "succeeded" || result.resultCode !== "launch_started") {
    throw new Error("Virtual Jetson launch request did not reach succeeded");
  }
});
