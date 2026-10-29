# Dashboard / ROS 2 sender signaling contract — v1

This describes the dashboard and Jetson sender contract included in this repository.
Provision robot identities, viewer/operator permissions and optional TURN infrastructure
in your own Firebase project before use. The repository contains code and configuration
templates, not production credentials or a pre-provisioned backend.

## Provisioning (trusted operator)

Create `robots/{robotId}` through Firebase Console or a trusted Admin SDK:

```json
{
  "name": "Rail robot",
  "enabled": true,
  "senderId": "<Jetson Firebase Authentication UID>",
  "viewerUids": ["<authorized viewer Firebase Authentication UID>"],
  "previewObjectKey": "live/jetson/gui.jpg"
}
```

The IDs above are placeholders, not actual identities. The viewer must satisfy
the dashboard's existing verified-admin policy AND appear in `viewerUids`.
The sender must authenticate as `senderId` and have
`device_uploaders/{senderId}.enabled == true`. Use a device user ID token, not
an Admin SDK service-account credential (Admin SDK bypasses these rules).
Browser clients cannot provision or edit robot configuration.
Robot discovery queries `enabled == true` and `viewerUids array-contains uid`.
Until a matching robot is registered, Live View shows registration pending.

## Session and candidate paths

`robots/{robotId}/webrtcSessions/{sessionId}`

The viewer generates a new random session ID and creates a recvonly video offer.
Session keys are exactly:

| Field | Initial value / owner |
| --- | --- |
| schemaVersion | `1`, immutable |
| robotId | parent robot ID, immutable |
| ownerUid | viewer Firebase UID, immutable |
| senderId | configured sender Firebase UID, immutable |
| offer | `{type: "offer", sdp: "..."}`, immutable |
| answer | `null`, then `{type: "answer", sdp: "..."}` by sender, once |
| state | `requested`, then sender `negotiating`, then sender `connected` |
| createdAt | Firestore server timestamp, immutable |
| viewerHeartbeatAt | server timestamp; viewer updates every 10 seconds |
| expiresAt | hard deadline: creation + 1 hour, immutable |
| endedAt | `null`, then server timestamp on terminal state |
| errorCode | `null`, or sender error string of at most 128 characters |

All dates are Firestore Timestamp values, not ISO strings. All nullable fields
are present from creation. SDP strings are limited to 100,000 characters.
The sender may set `negotiating` and `answer` in one write, or set the answer
while remaining `negotiating`, then set `connected`. It cannot skip directly
from `requested` to `connected`. The viewer never writes shared `connected`;
it displays browser connection and playback state locally.

Either peer may close an active session. The viewer writes only
`{state: "closed", endedAt: serverTimestamp()}`; the sender may write `closed`
or `failed`, with `endedAt` and optional `errorCode`. No terminal session can
be revived, overwritten, or reused. Reconnection creates a new session ID.
Sender terminal writes remain allowed after expiry so it can report timeout.

ICE collections:

- `.../viewerCandidates/{candidateId}`: viewer creates, sender reads.
- `.../senderCandidates/{candidateId}`: sender creates, viewer reads.

Normal candidate: `{candidate, sdpMid, sdpMLineIndex, usernameFragment, createdAt}`.
Use explicit nulls for missing optional RTC fields. `candidate` is at most
8,192 characters; mid/username fragment at most 256; line index null or 0–255.
End marker: `{endOfCandidates: true, createdAt: serverTimestamp()}` ONLY.
Documents are immutable, create-only, and ordered by `createdAt`.
Write candidates sequentially and await the final candidate write before writing
the end marker. Queue remote ICE until the remote SDP has been applied.

## Added single-viewer lease (required)

`robots/{robotId}/webrtcLocks/active`:

```text
{ sessionId, ownerUid, leaseUntil: Firestore Timestamp }
```

Viewer creates the session AND acquires this lock atomically in one transaction.
A new session may take the lock only if it is absent or expired. The lease is
30 seconds; every 10 seconds the viewer atomically updates its session heartbeat
and renews `leaseUntil` to now + 30 seconds. Rules allow at most 40 seconds ahead
to tolerate modest clock skew; client clocks must be synchronized.

The sender can listen to this single lock document instead of scanning sessions.
Read the referenced session and verify all of the following before accepting it:

1. Robot enabled; senderId matches the authenticated sender; viewer still allowed.
2. Lock sessionId and ownerUid match the session; lease has not expired.
3. Heartbeat is less than 30 seconds old; expiresAt is in the future.
4. Session is nonterminal and schemaVersion is supported.

Continue checking these conditions while sending, including robot configuration
changes. Stop the previous stream immediately when the lock changes. Release media
and encoder resources on terminal state, timeout, expiry, permission revocation,
or ICE failure. When connectivity is unavailable, enforce the timeout locally;
do not wait indefinitely for a Firestore notification.

Viewer closure atomically marks the session closed and deletes the lock only if
it still references that session. Sender need not delete the lock; viewer closure
or lease expiry releases it. An old viewer cannot modify a replacement lock.
Browser exit cleanup is best effort; sender timeout is authoritative.

## Browser behavior

- Mount connection logic only inside the Live View menu, for an authorized user.
- One receive-only video transceiver, no microphone or browser camera request.
- STUN only initially: `stun:stun.l.google.com:19302`. No TURN credentials embedded.
- Heartbeat 10 seconds; negotiation timeout 25 seconds; disconnected grace 8 seconds.
- Connected peers with no new decoded video frames for 15 seconds fail and retry.
- At most three automatic retries after failure, with 5/10/20-second backoff;
  manual reconnect starts a fresh attempt cycle.
- Leaving the menu, changing robot, logout, or page exit closes the peer,
  listeners and timers. Sender still enforces the 30-second timeout.
- Five-second snapshot polling happens only inside Live View while video is not
  playing. Existing snapshots are explicitly labeled as fallback images.
- Optional Storage custom metadata `capturedAt` is an ISO 8601 UTC capture time.
  Until sender supplies it, UI shows Storage upload time labeled as upload time,
  not capture time. It does not infer freshness from the download time.
- `previewObjectKey` must explicitly identify the correct robot; without a robot
  assignment and this setting, the viewer does not fetch fallback images.
  The existing `live/jetson/gui.jpg` path is a legacy single-robot path, not a
  multi-robot routing mechanism. Its existing Storage policy permits all approved
  dashboard admins to read snapshots; robot-scoped WebRTC rules do not restrict
  direct legacy Storage access. Do not use this path for camera images requiring
  isolation between those trusted admins; that requires per-robot Storage rules.

## Trusted cleanup and deployment responsibilities

Firestore/Storage rules and Firebase Hosting are managed by the dashboard task.
Do not deploy competing rules from the ROS 2 task.

An expired session is rejected immediately by rules/sender checks regardless of
whether its documents have been deleted. This task does NOT deploy a scheduled
cleanup service. An operator must arrange trusted recursive cleanup before
continuous production use. Delete only terminal/expired session documents and
their candidate subcollections after verifying no active lock references them.
Do not assume Firestore TTL deletes subcollections. For a verified expired session,
the existing Firebase CLI supports recursive deletion of that exact path:

```sh
firebase firestore:delete robots/ROBOT_ID/webrtcSessions/EXPIRED_SESSION_ID --recursive --project your-firebase-project
```

Keep cleanup authority server-side. Browser/sender cannot delete session history.
Do not set a TTL policy that deletes parent sessions before candidate cleanup.

## Validation

```sh
node node_modules/typescript/bin/tsc --noEmit
node --test tests/webrtc-peer.test.mjs
firebase emulators:exec --only firestore --project demo-rail-webrtc --config firebase.emulator.json "node --test tests/webrtc-rules.test.mjs"
node node_modules/vite/bin/vite.js build --config firebase.vite.config.ts
```

Emulator tests use synthetic identities only and require Java 21+. They check
access denial, ownership, single-viewer lease contention, expired takeover,
terminal state handling and candidate constraints. Peer tests use a mock WebRTC
peer to check asynchronous lifecycle/ICE ordering. They do not establish codec
compatibility or measure Jetson FPS, encoding load, or network latency.
Real same-network Jetson-to-browser playback and remote-network/TURN testing
remain required after robot registration and matching sender implementation.

Verified locally on 2026-09-06: TypeScript check and Firebase production build
passed; 5 mocked peer lifecycle tests, 8 Firestore emulator security tests, and
15 existing dashboard/data tests passed (28 tests total). Existing HTML-render
test uses the retained server build; the deployed Firebase client was built from
the updated source. No real Jetson/browser media exchange was performed.
