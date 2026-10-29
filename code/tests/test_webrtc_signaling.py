import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
from node_runtime.webrtc_signaling import (FirestoreSignaling, fields, decode,
    session_valid, context_valid, utc, validate_candidate, SignalingError)
from node_runtime.webrtc_sender import SessionSender

CONFIG = dict(projectId="project", robotId="r", senderId="jetson", senderUid="jetson",
              apiKey="test-placeholder", refreshToken="test-placeholder")


def document(now=1000):
    return {"path": "robots/r/webrtcSessions/s", "updateTime": "v1", "data": {
        "schemaVersion": 1, "robotId": "r", "senderId": "jetson", "ownerUid": "viewer",
        "state": "requested", "createdAt": utc(1000), "viewerHeartbeatAt": utc(now),
        "expiresAt": utc(4600), "answer": None, "endedAt": None, "errorCode": None,
        "offer": {"type": "offer", "sdp": "m=video 9 RTP/SAVPF 96\na=mid:0\na=recvonly\n"}}}


def context(now=1000):
    return {"robot": {"path": "robots/r", "updateTime": "robot-v1", "data": {
        "enabled": True, "senderId": "jetson", "viewerUids": ["viewer"]}},
        "lock": {"path": "robots/r/webrtcLocks/active", "updateTime": "lock-v1", "data": {
            "sessionId": "s", "ownerUid": "viewer", "leaseUntil": utc(now + 30)}},
        "session": document(now)}


class SignalingTests(unittest.TestCase):
    def test_session_authorization_heartbeat_and_fixed_expiry(self):
        valid = lambda d, now=1000: session_valid(d, "r", "jetson", {"viewer"}, now)
        self.assertTrue(valid(document()))
        self.assertFalse(valid(document(), 1030))
        self.assertFalse(valid(document(), 999))
        self.assertTrue(valid(document(4599), 4599))
        self.assertFalse(valid(document(4600), 4600))
        for key, value in (("ownerUid", "intruder"), ("senderId", "other"), ("robotId", "other"),
                           ("state", "closed"), ("expiresAt", utc(4661)), ("schemaVersion", True)):
            changed = document()
            changed["data"][key] = value
            self.assertFalse(valid(changed), key)
        for key in ("answer", "endedAt", "errorCode"):
            changed = document()
            del changed["data"][key]
            self.assertFalse(valid(changed), key)

    def test_remote_authorization_and_matching_lock_required(self):
        valid = lambda c: context_valid(c, CONFIG, None, 1000)
        self.assertTrue(valid(context()))
        for section, key, value in (("robot", "enabled", False), ("robot", "senderId", "other"),
                ("robot", "viewerUids", []), ("lock", "sessionId", "replacement"),
                ("lock", "ownerUid", "other"), ("lock", "leaseUntil", utc(1000)),
                ("lock", "leaseUntil", utc(1041))):
            changed = context()
            changed[section]["data"][key] = value
            self.assertFalse(valid(changed), (section, key))
        for section in ("robot", "lock", "session"):
            changed = context()
            changed[section] = None
            self.assertFalse(valid(changed))
        self.assertFalse(context_valid(context(), CONFIG, set(), 1000))

    def test_identity_config_optional_local_allowlist(self):
        self.assertIsNone(FirestoreSignaling(CONFIG).allowed_uids)
        self.assertEqual(FirestoreSignaling(dict(CONFIG, allowedViewerUids=[])).allowed_uids, set())
        with self.assertRaises(ValueError):
            FirestoreSignaling(dict(CONFIG, senderId="other"))

    def test_firestore_timestamp_and_candidate_contract(self):
        value = fields({"createdAt": utc(1000), "offer": {"type": "offer", "sdp": "v=0"}})
        self.assertIn("timestampValue", value["createdAt"])
        self.assertEqual(decode(value["offer"]), {"type": "offer", "sdp": "v=0"})
        self.assertEqual(validate_candidate({"endOfCandidates": True, "createdAt": utc(1000)}), ("", 0))
        candidate = dict(candidate="candidate:" + "a" * (8192-10), sdpMid=None,
                         sdpMLineIndex=255, usernameFragment=None, createdAt=utc(1000))
        self.assertEqual(validate_candidate(candidate)[1], 255)
        candidate.update(sdpMLineIndex=None, sdpMid="0")
        self.assertEqual(validate_candidate(candidate, document()["data"]["offer"]["sdp"])[1], 0)
        for key, value in (("candidate", "candidate:" + "a" * 8193), ("sdpMLineIndex", True),
                           ("sdpMLineIndex", 256), ("sdpMid", "a" * 257)):
            with self.assertRaises(ValueError):
                validate_candidate(dict(candidate, **{key: value}))

    def test_sdp_character_limit(self):
        changed = document()
        prefix = changed["data"]["offer"]["sdp"]
        changed["data"]["offer"]["sdp"] = prefix + "é" * (100000-len(prefix))
        self.assertTrue(session_valid(changed, "r", "jetson", None, 1000))
        changed["data"]["offer"]["sdp"] += "a"
        self.assertFalse(session_valid(changed, "r", "jetson", None, 1000))

    def test_terminal_cas_uses_server_time_commit(self):
        client = FirestoreSignaling(CONFIG)
        with patch.object(client, "request", return_value={"writeResults": [{"updateTime": "v2"}],
                "commitTime": utc(1001)}) as request:
            result = client.patch(document(), {"state": "closed", "endedAt": utc(999)})
        method, endpoint, body = request.call_args.args
        self.assertEqual((method, endpoint), ("POST", ":commit"))
        write = body["writes"][0]
        self.assertEqual(write["currentDocument"], {"updateTime": "v1"})
        self.assertNotIn("endedAt", write["update"]["fields"])
        self.assertEqual(write["updateTransforms"], [{"fieldPath": "endedAt", "setToServerValue": "REQUEST_TIME"}])
        self.assertEqual(result["data"]["endedAt"], utc(1001))
        with self.assertRaises(ValueError):
            client.patch(document(), {"expiresAt": utc(9999)})

    def test_candidates_create_only_server_timestamp_and_no_delete(self):
        client = FirestoreSignaling(CONFIG)
        with patch.object(client, "request", return_value={}) as request:
            client.send_candidate(document()["path"], 0, {"endOfCandidates": True, "createdAt": utc(1000)})
        write = request.call_args.args[2]["writes"][0]
        self.assertEqual(write["currentDocument"], {"exists": False})
        self.assertEqual(write["updateTransforms"], [{"fieldPath": "createdAt", "setToServerValue": "REQUEST_TIME"}])
        self.assertEqual(set(write["update"]["fields"]), {"endOfCandidates"})
        with patch.object(client, "_auth") as auth:
            for method, body in (("DELETE", None), ("POST", {"writes": [{"delete": "anything"}]})):
                with self.assertRaises(ValueError):
                    client.request(method, ":commit", body)
            auth.assert_not_called()
        self.assertFalse(hasattr(client, "cleanup"))

    def test_context_reads_coherent_transaction_and_no_uploader_document(self):
        client = FirestoreSignaling(CONFIG)
        prefix = "projects/project/databases/(default)/documents/"
        def raw(doc):
            values = dict(doc["data"])
            viewers = values.pop("viewerUids", None)
            encoded = fields(values)
            if viewers is not None:
                encoded["viewerUids"] = {"arrayValue": {"values": [{"stringValue": uid} for uid in viewers]}}
            return {"found": {"name": prefix + doc["path"], "updateTime": doc["updateTime"], "fields": encoded}}
        c = context()
        with patch.object(client, "request", side_effect=[{"transaction": "transaction-id"},
                [raw(c["robot"]), raw(c["lock"])], [raw(c["session"])], {}]) as request:
            result = client.context()
        self.assertEqual(result["session"]["path"], document()["path"])
        self.assertTrue(context_valid(result, CONFIG, None, 1000))
        self.assertEqual(request.call_args_list[1].args[2]["transaction"], "transaction-id")
        self.assertEqual(request.call_args_list[2].args[2]["transaction"], "transaction-id")
        self.assertEqual(request.call_args.args[1], ":rollback")
        self.assertNotIn("device_uploaders", str(request.call_args_list))
        self.assertNotIn("runQuery", str(request.call_args_list))

    def test_commit_uses_documents_action_url_and_bearer_auth(self):
        client = FirestoreSignaling(CONFIG)
        client._token = "test-only-token"
        with patch.object(client, "_auth"), patch.object(client, "_raw", return_value={}) as raw:
            client.request("POST", ":commit", {"writes": []})
        request = raw.call_args.args[0]
        self.assertEqual(request.full_url,
            "https://firestore.googleapis.com/v1/projects/project/databases/(default)/documents:commit")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-only-token")

    def test_server_time_follows_firestore_responses_not_the_jetson_clock(self):
        # RTC가 없는 Jetson 시계가 3시간 뒤처져 있어도 서버 기준 시각을 쓴다.
        client = FirestoreSignaling(CONFIG)
        client._token = "test-only-token"
        self.assertIsNone(client.server_now())
        with patch("node_runtime.webrtc_signaling.time.time", return_value=1000.0):
            self.assertEqual(client.now(), 1000.0)
        with patch.object(client, "_auth"), \
                patch.object(client, "_raw", return_value=[{"readTime": utc(20000)}]), \
                patch("node_runtime.webrtc_signaling.time.monotonic", side_effect=[10.0, 10.2]):
            client.request("POST", ":runQuery", {})
        with patch("node_runtime.webrtc_signaling.time.monotonic", return_value=15.1), \
                patch("node_runtime.webrtc_signaling.time.time", return_value=9200.0):
            self.assertAlmostEqual(client.server_now(), 20005.0, places=6)
            self.assertAlmostEqual(client.now(), 20005.0, places=6)
        with patch.object(client, "_auth"), \
                patch.object(client, "_raw", return_value={"commitTime": utc(30000)}), \
                patch("node_runtime.webrtc_signaling.time.monotonic", side_effect=[20.0, 20.0]):
            client.request("POST", ":commit", {"writes": []})
        with patch("node_runtime.webrtc_signaling.time.monotonic", return_value=21.0):
            self.assertAlmostEqual(client.server_now(), 30001.0, places=6)

    def test_sender_validates_sessions_with_server_time(self):
        sender, signal = self.active_sender()
        signal.server_now.return_value = 1001.0
        # The Jetson clock is hours behind; the session is still judged current.
        with patch("node_runtime.webrtc_sender.time.time", return_value=-5000.0):
            self.assertTrue(sender._valid(context()))
        signal.server_now.return_value = 1100.0
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001.0):
            self.assertFalse(sender._valid(context()))

    def active_sender(self):
        signal = Mock(config=CONFIG, allowed_uids={"viewer"})
        with patch("node_runtime.webrtc_sender.threading.Thread.start"):
            sender = SessionSender(signal, Mock())
        sender._session = document()
        sender._context = context()
        sender._media = Mock()
        signal.context.return_value = context()
        signal.get.return_value = document()
        signal.candidates.return_value = []
        return sender, signal

    def test_idle_no_encoder_and_watchdog_closes(self):
        with patch("node_runtime.webrtc_sender.threading.Thread.start"):
            factory = Mock()
            sender = SessionSender(Mock(), factory)
        self.assertFalse(sender.push(b"jpeg", 1))
        factory.assert_not_called()
        sender, _ = self.active_sender()
        media = sender._media
        sender.watchdog()
        media.close.assert_called_once()
        self.assertIsNone(sender._media)
        self.assertFalse(sender.push(b"jpeg", 1))

    def test_lease_uses_earliest_deadline(self):
        with patch("node_runtime.webrtc_sender.time.time", return_value=1000), patch("node_runtime.webrtc_sender.time.monotonic", return_value=10):
            c = context()
            c["lock"]["data"]["leaseUntil"] = utc(1010)
            self.assertEqual(SessionSender._lease(c), 20)
            c["session"]["data"]["expiresAt"] = utc(1005)
            self.assertEqual(SessionSender._lease(c), 15)

    def test_stale_callbacks_and_queue_overflow_are_bounded(self):
        with patch("node_runtime.webrtc_sender.threading.Thread.start"):
            sender = SessionSender(None, None)
        sender._event(-1, "ice", ("ignored", 0))
        self.assertEqual(sender._events.qsize(), 0)
        for _ in range(150): sender._event(0, "ice", ("candidate:abc", 0))
        self.assertEqual(sender._events.qsize(), 132)
        self.assertEqual(sender._deadline, 0)
        self.assertTrue(sender._event_overflow)
        sender._media = Mock()
        sender.watchdog()
        self.assertEqual(sender.error, "event_queue_overflow")

    def test_active_revocation_and_fixed_deadline_changes_stop_media(self):
        for section, key, value in (("robot", "enabled", False), ("robot", "viewerUids", []),
                ("lock", "sessionId", "replacement"), ("lock", "leaseUntil", utc(999)),
                ("session", "expiresAt", utc(4601)), ("session", "state", "closed"),
                ("session", "viewerHeartbeatAt", utc(970))):
            sender, signal = self.active_sender()
            changed = context()
            changed[section]["data"][key] = value
            signal.context.return_value = changed
            media = sender._media
            with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
                sender._poll_active(document(), 0)
            media.close.assert_called_once()
            self.assertIsNone(sender._media)
            signal.candidates.assert_not_called()
        sender, signal = self.active_sender()
        signal.context.return_value["robot"]["updateTime"] = "robot-v2"
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            sender._poll_active(document(), 0)
        self.assertIsNone(sender._media)

    def test_heartbeat_cas_conflict_rechecks_context(self):
        sender, signal = self.active_sender()
        refreshed = context(1010)
        refreshed["session"]["updateTime"] = "v2"
        signal.context.side_effect = [refreshed]
        signal.patch.side_effect = [SignalingError("http_412"), refreshed["session"]]
        values = {"answer": {"type": "answer", "sdp": "answer"}}
        with patch("node_runtime.webrtc_sender.time.time", return_value=1011):
            result = sender._patch_active(document(), 0, values)
        self.assertEqual(result, refreshed["session"])
        self.assertEqual(signal.patch.call_count, 2)
        self.assertEqual(signal.patch.call_args.args, (refreshed["session"], values))
        self.assertEqual(signal.context.call_count, 1)

    def test_answer_and_negotiating_transition_are_published_atomically(self):
        signal = Mock(config=CONFIG, allowed_uids={"viewer"})
        media = Mock()
        with patch("node_runtime.webrtc_sender.threading.Thread.start"):
            sender = SessionSender(signal, lambda: media)
        signal.context.return_value = context()
        signal.candidates.return_value = []
        signal.patch.side_effect = lambda doc, values: {
            **doc, "data": dict(doc["data"], **values), "updateTime": "v2"
        }
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            sender._start(context())
            generation = sender._generation
            sender._event(generation, "answer", "v=0\r\n")
            sender._poll_active(document(), generation)
        values = signal.patch.call_args.args[1]
        self.assertEqual(values["state"], "negotiating")
        self.assertEqual(values["answer"], {"type": "answer", "sdp": "v=0\r\n"})

    def test_progress_callbacks_do_not_delay_answer_behind_context_read(self):
        sender, signal = self.active_sender()
        order = []
        signal.patch.side_effect = lambda doc, values: (
            order.append("answer") or {**doc, "data": dict(doc["data"], **values)})
        signal.context.side_effect = lambda path: order.append("context") or context()
        for state in ("setting_remote", "creating_answer", "setting_local", "connecting"):
            sender._event(0, "state", state)
        sender._event(0, "answer", "v=0\r\n")
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            sender._poll_active(document(), 0)
        self.assertEqual(order, ["answer", "context"])
        self.assertTrue(sender._events.empty())

    def test_network_callback_batch_stays_bounded(self):
        sender, signal = self.active_sender()
        for index in range(10):
            sender._event(0, "state", "connecting")
            sender._event(0, "ice", ("candidate:abc", 0))
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            sender._poll_active(document(), 0)
        signal.send_candidates.assert_called_once()
        self.assertEqual(len(signal.send_candidates.call_args.args[2]), 4)
        self.assertEqual(sender._events.qsize(), 6)

    def test_answer_precedes_already_queued_ice_and_batch_preserves_end_marker(self):
        sender, signal = self.active_sender()
        order = []
        signal.patch.side_effect = lambda doc, values: order.append("answer") or doc
        signal.send_candidates.side_effect = lambda *args: order.append("ice")
        for candidate in ("candidate:first", "candidate:second", ""):
            sender._event(0, "ice", (candidate, 0))
        sender._event(0, "answer", "v=0\r\n")
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            sender._poll_active(document(), 0)
        self.assertEqual(order, ["answer", "ice"])
        batch = signal.send_candidates.call_args.args[2]
        self.assertEqual([value.get("candidate") for value in batch], ["candidate:first", "candidate:second", None])
        self.assertTrue(batch[-1]["endOfCandidates"])
        self.assertEqual(sender._outgoing, 3)

    def test_candidate_batch_create_only_and_denied_commit_not_retried(self):
        client = FirestoreSignaling(CONFIG)
        values = [{"endOfCandidates": True, "createdAt": utc(1000)}] * 4
        with patch.object(client, "request", return_value={}) as request:
            client.send_candidates(document()["path"], 5, values)
        writes = request.call_args.args[2]["writes"]
        self.assertEqual(len(writes), 4)
        self.assertEqual([w["update"]["name"].rsplit("/", 1)[1] for w in writes], ["005", "006", "007", "008"])
        for write in writes:
            self.assertEqual(write["currentDocument"], {"exists": False})
            self.assertEqual(write["updateTransforms"], [{"fieldPath": "createdAt", "setToServerValue": "REQUEST_TIME"}])
        with patch.object(client, "request", side_effect=SignalingError("http_403")) as request:
            with self.assertRaises(SignalingError):
                client.send_candidates(document()["path"], 0, values)
            request.assert_called_once()
        for start, batch in ((0, []), (0, values + values), (127, values), (-1, values)):
            with self.assertRaises(ValueError):
                client.send_candidates(document()["path"], start, batch)

    def test_revoked_or_closed_context_on_answer_conflict_prevents_candidate_batch(self):
        for section, key, value in (("robot", "enabled", False), ("session", "state", "closed")):
            sender, signal = self.active_sender()
            changed = context()
            changed[section]["data"][key] = value
            signal.context.return_value = changed
            signal.patch.side_effect = SignalingError("http_412")
            sender._event(0, "ice", ("candidate:abc", 0))
            sender._event(0, "answer", "v=0\r\n")
            with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
                with self.assertRaises(SignalingError):
                    sender._poll_active(document(), 0)
            signal.send_candidates.assert_not_called()
            self.assertIsNone(sender._media)

    def test_watchdog_distinguishes_negotiation_from_lease_expiry(self):
        for reason in ("negotiation_timeout", "session_expired", "heartbeat_timeout", "lock_lease_expired"):
            sender, _ = self.active_sender()
            sender.state = "negotiating"
            sender._deadline = 11 if reason == "negotiation_timeout" else 9
            sender._negotiation_deadline = 9
            if reason == "session_expired":
                sender._context["session"]["data"]["expiresAt"] = utc(1005)
            elif reason == "heartbeat_timeout":
                sender._context["session"]["data"]["viewerHeartbeatAt"] = utc(970)
            elif reason == "lock_lease_expired":
                sender._context["lock"]["data"]["leaseUntil"] = utc(1005)
            with patch("node_runtime.webrtc_sender.time.monotonic", return_value=10):
                sender.watchdog()
            self.assertEqual(sender.error, reason)
            self.assertIsNone(sender._media)

    def test_negotiation_timeout_logs_candidate_kinds_without_addresses(self):
        from node_runtime.webrtc_sender import candidate_kind
        self.assertEqual(candidate_kind("candidate:1 1 UDP 2122 192.168.0.6 5000 typ host"), "host/udp")
        self.assertEqual(candidate_kind("candidate:2 1 udp 2122 ab12-cd.local 5000 typ host generation 0"), "host/udp/mdns")
        self.assertEqual(candidate_kind("candidate:3 1 udp 1686 203.0.113.9 6000 typ srflx raddr 0.0.0.0 rport 0"), "srflx/udp")
        self.assertEqual(candidate_kind(""), "end")
        sender, _ = self.active_sender()
        sender.state = "negotiating"
        sender._deadline, sender._negotiation_deadline = 11, 9
        sender._local_kinds = ["host/udp", "srflx/udp"]
        sender._remote_kinds = ["host/udp/mdns"]
        with patch("node_runtime.webrtc_sender.time.monotonic", return_value=10),                 self.assertLogs("node_runtime.webrtc_sender", "WARNING") as logs:
            sender.watchdog()
        self.assertEqual(sender.error, "negotiation_timeout")
        self.assertIn("local=host/udpx1,srflx/udpx1 remote=host/udp/mdnsx1", logs.output[0])
        self.assertNotIn("192.168", logs.output[0])

    def test_watchdog_expiry_during_context_read_cannot_resurrect(self):
        sender, signal = self.active_sender()
        media = sender._media
        def expire_in_flight(path):
            sender._end("session_expired")
            return context()
        signal.context.side_effect = expire_in_flight
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            sender._poll_active(document(), 0)
        self.assertIsNone(sender._session)
        self.assertIsNone(sender._media)
        self.assertEqual(sender._deadline, 0)
        media.close.assert_called_once()
        signal.candidates.assert_not_called()

    def test_terminal_report_failure_does_not_block_new_admission(self):
        sender, signal = self.active_sender()
        sender._media.is_closed.return_value = True
        sender._end()
        with patch.object(sender, "_tidy", side_effect=SignalingError("http_403")) as tidy, \
                patch.object(sender, "_start", side_effect=lambda _: sender._stop.set()) as start, \
                patch("node_runtime.webrtc_sender.time.time", return_value=1001), \
                patch("node_runtime.webrtc_sender.time.monotonic", return_value=100):
            sender._run()
        tidy.assert_called_once()
        start.assert_called_once_with(signal.context.return_value)
        self.assertEqual(sender._cleanup_after, 130)

    def test_cas_retry_is_bounded_and_lock_replacement_prevents_retry(self):
        sender, signal = self.active_sender()
        signal.patch.side_effect = SignalingError("http_412")
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            with self.assertRaisesRegex(SignalingError, "http_412"):
                sender._patch_active(document(), 0, {"state": "connected"})
        self.assertEqual(signal.patch.call_count, 3)
        self.assertEqual(signal.context.call_count, 2)
        sender, signal = self.active_sender()
        replacement = context()
        replacement["lock"]["data"]["sessionId"] = "other"
        signal.context.side_effect = [replacement]
        signal.patch.side_effect = SignalingError("http_412")
        with patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            with self.assertRaisesRegex(SignalingError, "session_closed_or_invalid"):
                sender._patch_active(document(), 0, {"state": "connected"})
        signal.patch.assert_called_once()
        self.assertIsNone(sender._media)

    def test_terminal_reporting_never_deletes_history(self):
        sender, signal = self.active_sender()
        sender._end("session_expired")
        sender._tidy()
        signal.patch.assert_called_once()
        self.assertEqual(signal.patch.call_args.args[1]["state"], "failed")
        self.assertIn("endedAt", signal.patch.call_args.args[1])
        signal.cleanup.assert_not_called()
        self.assertEqual(sender._cleanup, [])

    def test_authentication_failure_in_worker_closes_media(self):
        sender, signal = self.active_sender()
        media = sender._media
        signal.context.side_effect = SignalingError("http_403")
        with patch.object(sender._stop, "wait", side_effect=lambda _: sender._stop.set()):
            sender._run()
        media.close.assert_called_once()
        self.assertIsNone(sender._media)
        self.assertEqual(sender.error, "http_403")

    def test_stalled_teardown_blocks_new_encoder_then_resumes(self):
        sender, signal = self.active_sender()
        retired = sender._media
        retired.is_closed.return_value = False
        sender._end("media_failed")
        waits = []
        def wait_for_teardown(delay):
            waits.append(delay)
            if delay == 0.1:
                signal.context.assert_not_called()
                self.assertFalse(sender.push(b"jpeg", 1))
                retired.is_closed.return_value = True
        with patch.object(sender._stop, "wait", side_effect=wait_for_teardown), patch.object(sender, "_tidy"), \
                patch.object(sender, "_start", side_effect=lambda _: sender._stop.set()) as start, \
                patch("node_runtime.webrtc_sender.time.time", return_value=1001):
            sender._run()
        self.assertEqual(waits[0], 0.1)
        start.assert_called_once_with(signal.context.return_value)
        retired.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
