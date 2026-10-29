import { before, after, beforeEach, test } from 'node:test';
import { readFile } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { initializeTestEnvironment, assertFails, assertSucceeds } from '@firebase/rules-unit-testing';
import { doc, setDoc, updateDoc, getDoc, getDocs, collection, query, where, deleteDoc,
  writeBatch, runTransaction, Timestamp, serverTimestamp } from 'firebase/firestore';

let env;
const viewer = (uid = 'viewer') => env.authenticatedContext(uid,
  { email: 'admin1@example.invalid', email_verified: true }).firestore();
const sender = (uid = 'sender') => env.authenticatedContext(uid).firestore();
const sessionPath = (id) => `robots/robot/webrtcSessions/${id}`;
const lockPath = 'robots/robot/webrtcLocks/active';
const future = (ms) => Timestamp.fromMillis(Date.now() + ms);
const payload = (uid) => ({ schemaVersion: 1, robotId: 'robot', ownerUid: uid, senderId: 'sender',
  offer: { type: 'offer', sdp: 'v=0\r\n' }, answer: null, state: 'requested',
  createdAt: serverTimestamp(), viewerHeartbeatAt: serverTimestamp(), expiresAt: future(3_600_000),
  endedAt: null, errorCode: null });
const candidate = () => ({ candidate: 'candidate:test', sdpMid: '0', sdpMLineIndex: 0,
  usernameFragment: null, createdAt: serverTimestamp() });
async function claim(db, uid, id) {
  return runTransaction(db, async (tx) => {
    const lock = await tx.get(doc(db, lockPath));
    if (lock.exists() && lock.data().leaseUntil.toMillis() > Date.now()) throw new Error('busy');
    tx.set(doc(db, sessionPath(id)), payload(uid));
    tx.set(doc(db, lockPath), { sessionId: id, ownerUid: uid, leaseUntil: future(30_000) });
  });
}
async function beat(db, id, ms = 30_000) {
  const batch = writeBatch(db);
  batch.update(doc(db, sessionPath(id)), { viewerHeartbeatAt: serverTimestamp() });
  batch.update(doc(db, lockPath), { leaseUntil: future(ms) });
  return batch.commit();
}
before(async () => {
  env = await initializeTestEnvironment({ projectId: 'demo-rail-webrtc',
    firestore: { host: '127.0.0.1', port: 8087, rules: await readFile(new URL('../firestore.rules', import.meta.url), 'utf8') } });
});
beforeEach(async () => {
  await env.clearFirestore();
  await env.withSecurityRulesDisabled(async (context) => {
    const db = context.firestore();
    await setDoc(doc(db, 'robots/robot'), { enabled: true, senderId: 'sender', viewerUids: ['viewer', 'viewer2'] });
    await setDoc(doc(db, 'device_uploaders/sender'), { enabled: true });
    await setDoc(doc(db, 'device_uploaders/other-sender'), { enabled: true });
  });
});
after(async () => { await env?.cleanup(); });

test('live telemetry is sender-owned, robot-scoped and node health reaches the viewer', async () => {
  const path = 'robots/robot/telemetry/current';
  const value = { schemaVersion: 1, senderId: 'sender', updatedAt: serverTimestamp(), topics: [], mission: null,
    cameras: { side: {}, front: {} } };
  await assertSucceeds(setDoc(doc(sender(), path), value));
  const virtualNodeHealth = [
    { id: 'camera_front', status: 'ok', observedAt: '2026-09-26T08:00:00Z', detail: 'fps_received' },
    { id: 'camera_side', status: 'ok', observedAt: '2026-09-26T08:00:00Z', detail: 'fps_received' },
    { id: 'vision', status: 'ok', observedAt: '2026-09-26T08:00:00Z', detail: 'timing_received' },
    { id: 'mission', status: 'ok', observedAt: '2026-09-26T08:00:00Z', detail: 'mission_state_received' },
    { id: 'stm_bridge', status: 'ok', observedAt: '2026-09-26T08:00:00Z', detail: 'ready' },
    { id: 'encoder_left', status: 'observed', observedAt: '2026-09-26T08:00:00Z', detail: 'shared_feedback' },
    { id: 'encoder_right', status: 'observed', observedAt: '2026-09-26T08:00:00Z', detail: 'shared_feedback' },
    { id: 'cleaner_front', status: 'unknown', observedAt: null, detail: 'no_feedback' },
    { id: 'cleaner_side', status: 'unknown', observedAt: null, detail: 'no_feedback' },
    { id: 'pump_front', status: 'unknown', observedAt: null, detail: 'no_feedback' },
    { id: 'pump_side', status: 'fault', observedAt: '2026-09-26T08:00:00Z', detail: 'simulated_fault' },
  ];
  await assertSucceeds(setDoc(doc(sender(), path), { ...value, nodeHealth: virtualNodeHealth }));
  await assertFails(setDoc(doc(sender(), path), { ...value,
    nodeHealth: Array.from({ length: 17 }, () => ({ id: 'x', status: 'ok', observedAt: null, detail: 'x' })) }));
  const received = await assertSucceeds(getDoc(doc(viewer(), path)));
  assert.deepEqual(received.data().nodeHealth, virtualNodeHealth);
  for (const db of [viewer(), viewer('outsider'), sender('other-sender'), env.unauthenticatedContext().firestore()]) {
    await assertFails(setDoc(doc(db, path), value));
  }
  await assertFails(getDoc(doc(viewer('outsider'), path)));
  await assertFails(deleteDoc(doc(sender(), path)));
  await assertFails(updateDoc(doc(sender(), path), { updatedAt: future(-60_000) }));
});

test('capture results publish before mission completion, preserve identity and cannot regress', async () => {
  const captureId = 'cap_' + 'a'.repeat(32);
  const path = `robots/robot/captures/${captureId}`;
  const value = { schemaVersion: 1, senderId: 'sender', captureId, runId: 'run_' + 'a'.repeat(24),
    cameraRole: 'side', phase: 'initial', zone: 1, capturedAt: new Date().toISOString(), analyzedAt: null,
    updatedAt: serverTimestamp(), status: 'captured', rawObjectKey: `robots/robot/captures/${captureId}/raw.jpg`, processedObjectKeys: [] };
  await assertSucceeds(setDoc(doc(sender(), path), value));
  await assertSucceeds(getDoc(doc(viewer(), path)));
  await assertFails(getDoc(doc(viewer('outsider'), path)));
  await assertFails(setDoc(doc(sender('other-sender'), path), value));
  await assertFails(updateDoc(doc(viewer(), path), { status: 'processed' }));
  await assertSucceeds(updateDoc(doc(sender(), path), { status: 'processed', analyzedAt: new Date().toISOString(), updatedAt: serverTimestamp(),
    processedObjectKeys: [{ label: 'Rust', objectKey: `robots/robot/captures/${captureId}/rust.jpg` }] }));
  await assertFails(setDoc(doc(sender(), path), value));
  await assertFails(updateDoc(doc(sender(), path), { runId: 'run_' + 'b'.repeat(24), updatedAt: serverTimestamp() }));
  await assertFails(deleteDoc(doc(sender(), path)));
});

test('allowlisted admin discovers robots and owns offer; sender owns answer and ICE', async () => {
  const db = viewer(); const tx = sender();
  await assertSucceeds(getDocs(query(collection(db, 'robots'), where('enabled', '==', true), where('viewerUids', 'array-contains', 'viewer'))));
  await assertSucceeds(claim(db, 'viewer', 'one'));
  await assertSucceeds(beat(db, 'one'));
  await assertSucceeds(getDoc(doc(tx, sessionPath('one'))));
  await assertSucceeds(updateDoc(doc(tx, sessionPath('one')), { state: 'negotiating', answer: { type: 'answer', sdp: 'v=0\r\n' } }));
  await assertSucceeds(setDoc(doc(tx, `${sessionPath('one')}/senderCandidates/a`), candidate()));
  await assertSucceeds(setDoc(doc(db, `${sessionPath('one')}/viewerCandidates/a`), candidate()));
  await assertSucceeds(setDoc(doc(tx, `${sessionPath('one')}/senderCandidates/end`), { endOfCandidates: true, createdAt: serverTimestamp() }));
  await assertSucceeds(updateDoc(doc(tx, sessionPath('one')), { state: 'connected' }));
});
test('anonymous, unassigned admin and unassigned sender cannot access sessions', async () => {
  await claim(viewer(), 'viewer', 'one');
  for (const db of [env.unauthenticatedContext().firestore(), viewer('outsider'), sender('other-sender')]) {
    await assertFails(getDoc(doc(db, sessionPath('one'))));
    await assertFails(updateDoc(doc(db, sessionPath('one')), { state: 'closed', endedAt: serverTimestamp() }));
  }
  await assertFails(setDoc(doc(viewer(), 'robots/robot'), { enabled: true, senderId: 'attacker', viewerUids: ['viewer'] }));
});
test('admin viewer cannot forge answer, sender ICE, identity, future heartbeat or another owner session', async () => {
  const db = viewer(); await claim(db, 'viewer', 'one');
  for (const value of [{ answer: { type: 'answer', sdp: 'v=0' } }, { senderId: 'attacker' },
    { ownerUid: 'viewer2' }, { expiresAt: future(7_200_000) }, { viewerHeartbeatAt: future(60_000) }]) {
    await assertFails(updateDoc(doc(db, sessionPath('one')), value));
  }
  await assertFails(setDoc(doc(db, `${sessionPath('one')}/senderCandidates/a`), candidate()));
  await assertFails(getDoc(doc(viewer('viewer2'), sessionPath('one'))));
  await assertFails(updateDoc(doc(viewer('viewer2'), sessionPath('one')), { state: 'closed', endedAt: serverTimestamp() }));
  await assertFails(beat(db, 'one', 600_000));
});
test('concurrent requests grant one lease, and expired takeover rejects old renewal or deletion', async () => {
  const results = await Promise.allSettled([claim(viewer(), 'viewer', 'one'), claim(viewer('viewer2'), 'viewer2', 'two')]);
  assert.equal(results.filter((item) => item.status === 'fulfilled').length, 1);
  const winner = results[0].status === 'fulfilled' ? ['viewer', 'one'] : ['viewer2', 'two'];
  const loser = winner[0] === 'viewer' ? ['viewer2', 'two'] : ['viewer', 'one'];
  await env.withSecurityRulesDisabled(async (context) => {
    await updateDoc(doc(context.firestore(), lockPath), { leaseUntil: future(-1_000) });
  });
  await assertSucceeds(claim(viewer(loser[0]), loser[0], loser[1]));
  await assertFails(beat(viewer(winner[0]), winner[1]));
  await assertFails(deleteDoc(doc(viewer(winner[0]), lockPath)));
});
test('rules reject forced active lock takeover even when a client skips transaction checks', async () => {
  await claim(viewer(), 'viewer', 'one');
  const db = viewer('viewer2');
  const batch = writeBatch(db);
  batch.set(doc(db, sessionPath('two')), payload('viewer2'));
  batch.set(doc(db, lockPath), { sessionId: 'two', ownerUid: 'viewer2', leaseUntil: future(30_000) });
  await assertFails(batch.commit());
  await assertFails(setDoc(doc(db, sessionPath('two')), payload('viewer2')));
});
test('terminal sessions cannot revive or accept ICE; owner atomically closes and releases lock', async () => {
  const db = viewer(); await claim(db, 'viewer', 'one');
  const batch = writeBatch(db);
  batch.update(doc(db, sessionPath('one')), { state: 'closed', endedAt: serverTimestamp() });
  batch.delete(doc(db, lockPath));
  await assertSucceeds(batch.commit());
  await assertFails(updateDoc(doc(sender(), sessionPath('one')), { state: 'negotiating' }));
  await assertFails(setDoc(doc(db, `${sessionPath('one')}/viewerCandidates/a`), candidate()));
  await assertFails(claim(db, 'viewer', 'one'));
  await assertSucceeds(claim(db, 'viewer', 'new'));
});
test('sender cannot change offer or skip negotiation; candidates are immutable and bounded', async () => {
  const db = viewer(); const tx = sender(); await claim(db, 'viewer', 'one');
  await assertFails(updateDoc(doc(tx, sessionPath('one')), { offer: { type: 'offer', sdp: 'forged' } }));
  await assertFails(updateDoc(doc(tx, sessionPath('one')), { state: 'connected' }));
  await assertFails(setDoc(doc(tx, `${sessionPath('one')}/senderCandidates/a`), { ...candidate(), candidate: 'x'.repeat(9000) }));
  await assertSucceeds(setDoc(doc(tx, `${sessionPath('one')}/senderCandidates/a`), candidate()));
  await assertFails(setDoc(doc(tx, `${sessionPath('one')}/senderCandidates/a`), candidate()));
});
test('stale heartbeat blocks media signaling even when lease remains valid', async () => {
  const db = viewer(); await claim(db, 'viewer', 'one');
  await env.withSecurityRulesDisabled(async (context) => {
    await updateDoc(doc(context.firestore(), sessionPath('one')), { viewerHeartbeatAt: future(-31_000) });
  });
  await assertFails(beat(db, 'one'));
  await assertFails(setDoc(doc(sender(), `${sessionPath('one')}/senderCandidates/a`), candidate()));
  await assertSucceeds(updateDoc(doc(sender(), sessionPath('one')), { state: 'failed', endedAt: serverTimestamp(), errorCode: 'viewer_timeout' }));
});
