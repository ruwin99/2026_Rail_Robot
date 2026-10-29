import { after, before, beforeEach, test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { assertFails, assertSucceeds, initializeTestEnvironment } from "@firebase/rules-unit-testing";
import { createMockUserToken } from "@firebase/util";
import { collection, doc, documentId, getDoc, getDocs, orderBy, query,
  serverTimestamp, setDoc, Timestamp, updateDoc, where } from "firebase/firestore";

const projectId = "demo-rail-remote-virtual";
if (!projectId.startsWith("demo-")) throw new Error("Virtual tests require a demo Firebase project");

const robotId = "virtual-robot";
const operatorUid = "virtual-operator";
const senderUid = "virtual-sender";
let env;
let adminEmail;

const operator = () => env.authenticatedContext(operatorUid,
  { email: adminEmail, email_verified: true }).firestore();
const sender = () => env.authenticatedContext(senderUid).firestore();
const outsider = () => env.authenticatedContext("virtual-outsider",
  { email: adminEmail, email_verified: true }).firestore();
const commandRef = (db, id) => doc(db, "robots", robotId, "commands", id);
const requestedCommand = (action) => ({
  schemaVersion: 1, robotId, ownerUid: operatorUid, action, state: "requested",
  createdAt: serverTimestamp(), acceptedAt: null, finishedAt: null, resultCode: null,
});

before(async () => {
  const rules = await readFile(new URL("../firestore.rules", import.meta.url), "utf8");
  adminEmail = rules.match(/'([^']+@[^']+)'/)?.[1];
  assert.ok(adminEmail, "an administrator email must exist in the rules under test");
  env = await initializeTestEnvironment({ projectId,
    firestore: { host: "127.0.0.1", port: 8087, rules } });
});

beforeEach(async () => {
  await env.clearFirestore();
  await env.withSecurityRulesDisabled(async (context) => {
    const db = context.firestore();
    await setDoc(doc(db, "robots", robotId), {
      enabled: true, senderId: senderUid, viewerUids: [operatorUid],
      operatorUids: [operatorUid], remoteControlEnabled: true,
    });
    await setDoc(doc(db, "device_uploaders", senderUid), { enabled: true });
  });
});

after(async () => { await env?.cleanup(); });

async function virtualReceiver(id, services, { missionReady = true } = {}) {
  const ref = commandRef(sender(), id);
  const command = (await assertSucceeds(getDoc(ref))).data();
  if (command.action === "START" && !missionReady) {
    await assertSucceeds(updateDoc(ref, {
      state: "failed", finishedAt: serverTimestamp(), resultCode: "mission_not_ready",
    }));
    return;
  }
  await assertSucceeds(updateDoc(ref, { state: "accepted", acceptedAt: serverTimestamp() }));
  const result = command.action === "START" ? await services.start() : await services.stop();
  await assertSucceeds(updateDoc(ref, {
    state: result ? "succeeded" : "failed", finishedAt: serverTimestamp(),
    resultCode: command.action === "START"
      ? result ? "start_sent" : "start_rejected"
      : result ? "estop_ack" : "estop_rejected",
  }));
}

test("virtual operator START and E_STOP complete through Firestore and mock services", async () => {
  const calls = [];
  const services = {
    start: async () => { calls.push("mock:/mission/start"); return true; },
    stop: async () => { calls.push("mock:/stm/command E_STOP"); return true; },
  };
  await assertSucceeds(setDoc(commandRef(operator(), "start"), requestedCommand("START")));
  await virtualReceiver("start", services);
  const start = (await assertSucceeds(getDoc(commandRef(operator(), "start")))).data();
  assert.equal(start.state, "succeeded");
  assert.equal(start.resultCode, "start_sent");
  assert.ok(start.acceptedAt instanceof Timestamp);
  assert.ok(start.finishedAt instanceof Timestamp);

  await assertSucceeds(setDoc(commandRef(operator(), "stop"), requestedCommand("E_STOP")));
  const stopQuery = query(collection(sender(), "robots", robotId, "commands"),
    where("action", "==", "E_STOP"),
    where("createdAt", ">=", Timestamp.fromMillis(Date.now() - 60_000)),
    orderBy("createdAt", "desc"), orderBy(documentId(), "desc"));
  assert.deepEqual((await assertSucceeds(getDocs(stopQuery))).docs.map((item) => item.id), ["stop"]);
  await virtualReceiver("stop", services);
  const stop = (await assertSucceeds(getDoc(commandRef(operator(), "stop")))).data();
  assert.equal(stop.state, "succeeded");
  assert.equal(stop.resultCode, "estop_ack");
  assert.deepEqual(calls, ["mock:/mission/start", "mock:/stm/command E_STOP"]);
});

test("virtual setup rejects outsiders and disables requests when robot opt-in is off", async () => {
  await assertFails(setDoc(commandRef(outsider(), "outsider"), requestedCommand("START")));
  await env.withSecurityRulesDisabled(async (context) => {
    await updateDoc(doc(context.firestore(), "robots", robotId), { remoteControlEnabled: false });
  });
  await assertFails(setDoc(commandRef(operator(), "disabled"), requestedCommand("START")));
});

test("unready START never reaches the mock service and expired requests cannot be claimed", async () => {
  let serviceCalls = 0;
  const services = {
    start: async () => { serviceCalls++; return true; },
    stop: async () => { serviceCalls++; return true; },
  };
  await assertSucceeds(setDoc(commandRef(operator(), "unready"), requestedCommand("START")));
  await virtualReceiver("unready", services, { missionReady: false });
  const unready = (await assertSucceeds(getDoc(commandRef(operator(), "unready")))).data();
  assert.equal(unready.state, "failed");
  assert.equal(unready.resultCode, "mission_not_ready");

  await env.withSecurityRulesDisabled(async (context) => {
    await setDoc(commandRef(context.firestore(), "expired"), {
      schemaVersion: 1, robotId, ownerUid: operatorUid, action: "START", state: "requested",
      createdAt: Timestamp.fromMillis(Date.now() - 30_000),
      expiresAt: Timestamp.fromMillis(Date.now() - 20_000),
      acceptedAt: null, finishedAt: null, resultCode: null,
    });
  });
  await assertFails(updateDoc(commandRef(sender(), "expired"), {
    state: "accepted", acceptedAt: serverTimestamp(),
  }));
  // 새 형식(expiresAt 없음)도 서버 createdAt + 10 s가 지나면 수락할 수 없다.
  await env.withSecurityRulesDisabled(async (context) => {
    await setDoc(commandRef(context.firestore(), "expired_new"), {
      ...requestedCommand("START"), createdAt: Timestamp.fromMillis(Date.now() - 30_000),
    });
  });
  await assertFails(updateDoc(commandRef(sender(), "expired_new"), {
    state: "accepted", acceptedAt: serverTimestamp(),
  }));
  assert.equal(serviceCalls, 0);
});

test("production Jetson command store reads and acknowledges virtual START and STOP over emulator REST", async () => {
  await assertSucceeds(setDoc(commandRef(operator(), "store_start"), requestedCommand("START")));
  await assertSucceeds(setDoc(commandRef(operator(), "store_stop"), requestedCommand("E_STOP")));
  const python = process.env.RAIL_VIRTUAL_PYTHON || (process.platform === "win32" ? "python" : "python3");
  const probe = fileURLToPath(new URL("../../code/tests/remote_control_emulator_probe.py", import.meta.url));
  const output = execFileSync(python, [probe], {
    encoding: "utf8", timeout: 15_000,
    env: { ...process.env, FIRESTORE_EMULATOR_HOST: "127.0.0.1:8087",
      RAIL_VIRTUAL_PROJECT_ID: projectId, RAIL_VIRTUAL_ROBOT_ID: robotId,
      RAIL_VIRTUAL_COMMAND_ID: "store_stop", RAIL_VIRTUAL_SENDER_UID: senderUid,
      RAIL_VIRTUAL_SENDER_TOKEN: createMockUserToken({ sub: senderUid }, projectId) },
  });
  assert.match(output, /store_claim_and_finish_ok/);
  const start = (await assertSucceeds(getDoc(commandRef(operator(), "store_start")))).data();
  assert.equal(start.state, "succeeded");
  assert.equal(start.resultCode, "start_sent");
  const data = (await assertSucceeds(getDoc(commandRef(operator(), "store_stop")))).data();
  assert.equal(data.state, "succeeded");
  assert.equal(data.resultCode, "estop_ack");
});
