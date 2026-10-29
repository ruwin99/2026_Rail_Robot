# WebRTC signaling contract v1 — dashboard alignment, 2026-09-06

Authoritative shared contract: [`rail_dashboard/WEBRTC_SIGNALING_CONTRACT.md`](../../../rail_dashboard/WEBRTC_SIGNALING_CONTRACT.md). Alignment requirements: [dashboard-webrtc-alignment.md](dashboard-webrtc-alignment.md). This document records the Jetson implementation. Dashboard files, Firebase Hosting/Firestore/Storage rules, identity provisioning and coturn are outside this change.

## Identity and authorization

Trusted operators provision `robots/{robotId}` with `enabled: true`, `senderId` equal to the dedicated Jetson Firebase Authentication UID, `viewerUids` containing authorized viewers, and the explicitly assigned `previewObjectKey`. A viewer must satisfy the dashboard's verified-admin policy AND be listed in `viewerUids`. The sender authenticates using a device-user ID token, never an Admin SDK credential; `device_uploaders/{senderId}.enabled` must be true.

Jetson local `senderId` and `senderUid` must both equal that authenticated UID. An optional local `allowedViewerUids` list adds a restriction; it never replaces remote robot authorization. The sender reads robot/lock through rules-protected REST calls, which enforce uploader enablement. It does not directly read the uploader document, for which device clients have no read grant. Authentication or permission failure stops media.

Credentials and future TURN secrets remain in an operator-managed JSON outside the repository (`chmod 600`). No credential provisioning, rules deployment or cleanup service is performed here.

## Session and single-viewer lease

Viewer creates a fresh random `robots/{robotId}/webrtcSessions/{sessionId}` AND `robots/{robotId}/webrtcLocks/active` atomically. The lock contains exactly `{sessionId, ownerUid, leaseUntil: Timestamp}`. The lease lasts 30 seconds; every 10 seconds the viewer atomically updates heartbeat and renews the lease to now + 30 seconds. The allowed upper bound is 40 seconds ahead for modest clock skew. Keep clocks synchronized.

| Session field | Type, initial value and ownership |
|---|---|
| schemaVersion | Integer `1`, immutable |
| robotId / senderId / ownerUid | Parent robot / designated sender UID / authenticated viewer UID; immutable |
| offer | `{type:"offer", sdp:string}`, viewer-owned, immutable |
| answer | Explicit `null`, then one sender-owned `{type:"answer", sdp:string}` |
| state | `requested` → sender `negotiating` → sender `connected`; either peer may close |
| createdAt | Server Timestamp, immutable |
| viewerHeartbeatAt | Server Timestamp, viewer updates every 10 seconds |
| expiresAt | Fixed creation + one-hour deadline, immutable; never renewed by heartbeat |
| endedAt | Explicit `null`, then server Timestamp on terminal transition |
| errorCode | Explicit `null`, or sanitized sender error string, at most 128 characters |

All nullable fields are present from creation; dates are native Firestore Timestamps. REST internally decodes timestamps to UTC strings without changing their Firestore type. Dashboard computes the one-hour deadline using its synchronized clock; rules permit up to creation + 3,660 seconds for server/client creation skew. Sender honors the stored fixed deadline without extending it. SDP is limited to 100,000 characters. Browser creates one recvonly video offer, H.264 video only, no microphone, data channel or robot command path.

Sender verifies robot, lock and referenced session as a coherent read before admission and continuously while active: robot enabled, designated UID match, viewer authorized, matching lock session/owner, live lease, heartbeat younger than 30 seconds, future hard expiry, nonterminal supported schema. Robot configuration revision, changed lock/owner/offer/deadline, terminal session or permission failure ends the stream. Existing local OS lock prevents duplicate sender processes on one Jetson; Firestore lock identifies the sole authorized viewer.

Sender may write negotiating and answer together, or answer while already negotiating, then connected; it cannot skip requested → connected. Viewer never writes shared connected. Terminal sessions cannot be revived/reused; reconnect creates a new ID. Sender terminal writes may report expiry after the deadline. Viewer closure atomically writes closed/endedAt and deletes only its matching lock. Jetson never deletes the lock.

## ICE

`.../viewerCandidates/{candidateId}` is viewer-created/sender-read; `.../senderCandidates/{candidateId}` is sender-created/viewer-read. Documents are immutable, create-only, ordered by `createdAt`. Normal fields are exactly:

```text
{candidate, sdpMid, sdpMLineIndex, usernameFragment, createdAt: serverTimestamp()}
```

Use explicit nulls for missing optional RTC fields. Candidate strings are at most 8,192 characters; mid and username fragment at most 256; line index is null or integer 0–255. A null index is resolved through its matching sdpMid in the accepted offer; an absent or ambiguous match is rejected. End marker contains ONLY `{endOfCandidates: true, createdAt: serverTimestamp()}`. Sender candidate and terminal writes use Firestore server timestamp transforms, never local wall time for those fields.

Write candidates sequentially and await the final candidate before the end marker. Apply remote ICE only after remote SDP. Sender retains its local bound of 128 candidates per side and fixed callback queue; these are resource limits, not a global Firestore collection-count rule.

## Lifetime, resources and cleanup

Independent watchdog enforces the earliest of heartbeat + 30 seconds, lock lease and fixed hard deadline while network calls stall. Remote changes are observed through bounded periodic reads, not instantly across an unavailable network. Frame admission stops upon local detection; native teardown can finish asynchronously and blocks replacement encoding until complete. ROS callbacks perform no Firestore I/O or encoding.

Jetson attempts only terminal updates: `closed` or `failed`, `endedAt: serverTimestamp()`, optional sanitized `errorCode`. It never deletes session history, candidates or the active lock. Failed terminal reporting must not retain media or block later admission. Trusted dashboard-side/operator cleanup must recursively remove terminal/expired sessions and candidates after confirming no active lock references them. No scheduled cleanup service is installed here; parent TTL alone does not remove subcollections.

Existing independent five-second Firebase snapshot publisher remains available. Dashboard selects video versus fallback; no snapshot cadence, GUI/camera/inference/motor/UART node or Storage policy changes here.

