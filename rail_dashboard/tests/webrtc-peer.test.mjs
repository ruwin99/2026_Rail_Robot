import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import ts from 'typescript';
const source = await readFile(new URL('../firebase-client/webrtc-peer.ts', import.meta.url), 'utf8');
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } });
const { createViewerPeer } = await import(`data:text/javascript;base64,${Buffer.from(compiled.outputText).toString('base64')}`);
const flush = () => new Promise((resolve) => setImmediate(resolve));
function setup(create = async () => {}, close = async () => {}) {
  const calls = { candidates: [], remote: [], closed: 0, peerClosed: 0, unlisten: 0, failures: [], streams: [], beats: 0 };
  const pc = { connectionState: 'new', remoteDescription: null,
    addTransceiver(kind, options) { assert.equal(kind, 'video'); assert.equal(options.direction, 'recvonly'); },
    async createOffer() { return { type: 'offer', sdp: 'offer' }; },
    async setLocalDescription() { pc.onicecandidate({ candidate: { toJSON: () => ({ candidate: 'local' }) } }); },
    async setRemoteDescription(sdp) { pc.remoteDescription = sdp; },
    async addIceCandidate(value) { calls.remote.push(value); }, getReceivers: () => [],
    async getStats() { return new Map(); }, close() { calls.peerClosed++; } };
  const channel = { create, async candidate(value) { calls.candidates.push(value); }, async heartbeat() { calls.beats++; },
    async close() { calls.closed++; await close(); }, listen(answer, candidate, ended, failed) {
      Object.assign(calls, { answer, candidate, ended, failed }); return () => { calls.unlisten++; };
    } };
  const viewer = createViewerPeer(channel, { peer: pc, stream: (s) => calls.streams.push(s), status: () => {}, failed: (e) => calls.failures.push(e) });
  return { viewer, pc, calls };
}
test('queues ICE until answer, serializes remote candidates, and stops all resources', async () => {
  const { viewer, calls } = setup();
  try {
    await viewer.start(); await flush();
    assert.equal(calls.candidates[0].candidate, 'local');
    calls.candidate({ candidate: 'early' }); await flush(); assert.equal(calls.remote.length, 0);
    calls.answer({ type: 'answer', sdp: 'remote' });
    calls.candidate({ endOfCandidates: true }); await flush();
    assert.equal(calls.remote[0].candidate, 'early'); assert.equal(calls.remote[1], undefined);
  } finally { await viewer.stop(); }
  assert.equal(calls.closed, 1); assert.equal(calls.peerClosed, 1); assert.equal(calls.unlisten, 1);
  await viewer.stop(); assert.equal(calls.closed, 1);
});
test('stop waits for a pending Firestore create, then closes the late session', async () => {
  let resolve; const created = new Promise((done) => { resolve = done; });
  const { viewer, calls } = setup(() => created);
  const started = viewer.start(); await flush();
  const stopped = viewer.stop(); await flush();
  assert.equal(calls.closed, 0);
  resolve(); await started; await stopped;
  assert.equal(calls.closed, 1); assert.equal(calls.peerClosed, 1); assert.equal(calls.unlisten, 0);
  assert.equal(calls.candidates.length, 0);
});

test('a stalled signaling cleanup cannot hide a failure after media has already closed', async () => {
  let finishClose;
  const cleanup = new Promise((resolve) => { finishClose = resolve; });
  const { viewer, calls } = setup(undefined, () => cleanup);
  await viewer.start();
  calls.ended('negotiation_timeout');
  await flush();
  assert.equal(calls.peerClosed, 1);
  assert.equal(calls.failures.length, 1);
  assert.equal(calls.closed, 1);
  calls.failed(new Error('late callback'));
  assert.equal(calls.failures.length, 1);
  finishClose(); await viewer.stop();
  assert.equal(calls.closed, 1);
});
test('malformed answer and terminal sender state release peer and report failure', async () => {
  for (const action of ['bad-answer', 'ended']) {
    const { viewer, calls } = setup(); await viewer.start();
    if (action === 'ended') calls.ended(); else calls.answer({ type: 'offer', sdp: 'bad' });
    await flush(); assert.equal(calls.failures.length, 1); assert.equal(calls.closed, 1);
    assert.equal(calls.peerClosed, 1); await viewer.stop();
  }
});
test('10-second heartbeat and negotiation timeout stop a silent sender', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval'] });
  const { viewer, calls } = setup(); await viewer.start();
  t.mock.timers.tick(10_000); await flush(); assert.equal(calls.beats, 1);
  t.mock.timers.tick(15_000); await flush();
  assert.equal(calls.failures.length, 1); assert.equal(calls.peerClosed, 1);
  const beats = calls.beats; t.mock.timers.tick(40_000); await flush(); assert.equal(calls.beats, beats);
});

test('sender termination distinguishes negotiation timeout from expiry without exposing arbitrary remote text', async () => {
  for (const [reason, message] of [
    ['negotiation_timeout', '영상 연결 협상 시간이 초과되었습니다.'],
    ['heartbeat_timeout', '시청 연결 확인이 지연되어 영상을 종료했습니다.'],
    ['lock_lease_expired', '시청 연결 유지 시간이 만료되었습니다.'],
    ['session_expired', '시청 세션의 유효 시간이 만료되었습니다.'],
    ['untrusted remote diagnostic payload', '송신자가 연결을 종료했습니다.'],
  ]) {
    const { viewer, calls } = setup();
    await viewer.start(); calls.ended(reason); await flush();
    assert.equal(calls.failures[0].message, message);
    assert.equal(calls.closed, 1);
    await viewer.stop();
  }
});
test('a connected peer with no decoded video fails instead of showing a frozen live image', async (t) => {
  t.mock.timers.enable({ apis: ['setTimeout', 'setInterval', 'Date'] });
  const { viewer, calls, pc } = setup(); await viewer.start();
  pc.connectionState = 'connected'; pc.onconnectionstatechange();
  for (let i = 0; i < 3; i++) { t.mock.timers.tick(5_000); await flush(); }
  assert.equal(calls.failures.length, 1); assert.equal(calls.peerClosed, 1);
});
