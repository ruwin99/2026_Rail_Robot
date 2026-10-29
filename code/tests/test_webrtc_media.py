"""Media contract checks without loading native GStreamer or opening a camera."""
import gc
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "jetson_code"))
from node_runtime.webrtc_media import (
    GStreamerWebRTCSender, WebRTCMediaConfig, h264_payload,
    pipeline_description, select_encoder,
)


def offer(payload=102, profile="42e01f"):
    return (f"v=0\r\nm=video 9 UDP/TLS/RTP/SAVPF {payload}\r\na=recvonly\r\n"
            f"a=rtpmap:{payload} H264/90000\r\n"
            f"a=fmtp:{payload} packetization-mode=1;profile-level-id={profile}\r\n")


class MediaTests(unittest.TestCase):
    def test_vertical_camera_and_legacy_dimensions_are_negotiated_without_stretching(self):
        for width, height in ((640, 720), (1280, 360), (1280, 720)):
            description = pipeline_description(WebRTCMediaConfig(width=width, height=height), "x264enc", 102)
            self.assertIn("caps=image/jpeg,framerate=12/1", description)
            self.assertIn("width=(int){640,1280},height=(int){360,720}", description)
            self.assertNotIn("videoscale", description)
        with self.assertRaises(ValueError):
            WebRTCMediaConfig(height=240)
        with self.assertRaises(ValueError):
            WebRTCMediaConfig(width=640, height=360)
        self.assertEqual((WebRTCMediaConfig().width, WebRTCMediaConfig().height), (640, 720))

    def test_negotiates_offered_h264_payload_instead_of_fixed_96(self):
        self.assertEqual(h264_payload(offer(102)), 102)
        description = pipeline_description(WebRTCMediaConfig(), "x264enc", 102)
        self.assertIn("pt=102", description)
        self.assertIn("payload=102", description)

    def test_rejects_wrong_media_direction_codec_and_unbounded_offer(self):
        for invalid in (
            offer().replace("recvonly", "sendrecv"),
            offer() + "m=audio 9 UDP/TLS/RTP/SAVPF 111\r\n",
            offer() + offer(),
            offer(profile="64001f"),
            offer().replace("packetization-mode=1", "packetization-mode=0"),
            offer().replace("H264/90000", "VP8/90000"),
            "x" * 131073,
            None,
        ):
            with self.subTest(offer=repr(invalid)[:80]):
                with self.assertRaises(ValueError):
                    h264_payload(invalid)

    def test_auto_hardware_requires_encoder_and_converter(self):
        config = WebRTCMediaConfig()
        self.assertEqual(select_encoder(config, {"nvv4l2h264enc", "nvvidconv", "x264enc"}.__contains__),
                         "nvv4l2h264enc")
        self.assertEqual(select_encoder(config, {"nvv4l2h264enc", "x264enc"}.__contains__), "x264enc")

    def test_software_fallback_is_explicitly_configurable_or_disabled(self):
        config = WebRTCMediaConfig(software_encoder="openh264enc")
        self.assertEqual(select_encoder(config, {"openh264enc"}.__contains__), "openh264enc")
        for config in (WebRTCMediaConfig(software_encoder="disabled"),
                       WebRTCMediaConfig(encoder="nvv4l2h264enc")):
            with self.assertRaises(RuntimeError):
                select_encoder(config, {"x264enc"}.__contains__)

    def test_pipeline_has_no_backpressure_or_unbounded_media_queue(self):
        for encoder in ("x264enc", "openh264enc", "nvv4l2h264enc"):
            with self.subTest(encoder=encoder):
                description = pipeline_description(WebRTCMediaConfig(), encoder, 102)
                self.assertIn("block=false max-buffers=1 leaky-type=downstream", description)
                self.assertIn("max-size-buffers=1 max-size-bytes=0 max-size-time=0 leaky=downstream", description)
                self.assertNotIn("max-size-buffers=4", description)

    def test_leaky_queue_cannot_discard_fragments_after_rtp_packetization(self):
        # One H264 frame commonly spans many RTP packets: dropping individual
        # fragments makes almost every browser frame undecodable under load.
        for encoder in ("x264enc", "openh264enc", "nvv4l2h264enc"):
            with self.subTest(encoder=encoder):
                description = pipeline_description(WebRTCMediaConfig(), encoder, 102)
                before_pay, after_pay = description.split("rtph264pay", 1)
                self.assertIn("queue max-size-buffers=1", before_pay)
                self.assertNotIn("queue", after_pay)
                self.assertNotIn("leaky", after_pay)
                self.assertTrue(after_pay.endswith("! peer."))

    def test_latest_frame_replaces_old_frame_and_rejects_reordering(self):
        sender = GStreamerWebRTCSender()
        self.assertFalse(sender.push_jpeg(b"not-active", 0))
        sender._thread = object()  # Active session without a native worker.
        for stamp in range(100):
            self.assertTrue(sender.push_jpeg(bytes([stamp]), stamp))
        self.assertEqual(sender.stats()["pending_frames"], 1)
        self.assertEqual(sender.stats()["replaced"], 99)
        self.assertEqual(sender._latest[:2], (bytes([99]), 99))
        self.assertFalse(sender.push_jpeg(b"old", 98))
        self.assertFalse(sender.push_jpeg(b"duplicate", 99))
        self.assertFalse(sender.push_jpeg(b"", 100))
        self.assertFalse(sender.push_jpeg(b"bad-time", -1))
        self.assertEqual(sender._latest[:2], (bytes([99]), 99))

    def test_ice_queue_is_bounded_and_close_discards_pending_work(self):
        sender = GStreamerWebRTCSender()
        self.assertFalse(sender.add_ice("candidate:inactive", 0))
        sender._thread = object()
        for index in range(128):
            self.assertTrue(sender.add_ice(f"candidate:{index}", 0))
        self.assertFalse(sender.add_ice("candidate:overflow", 0))
        self.assertEqual(sender.stats()["pending_ice"], 128)
        sender.push_jpeg(b"jpeg", 1)
        sender.close()
        self.assertEqual(sender.stats()["pending_ice"], 0)
        self.assertEqual(sender.stats()["pending_frames"], 0)
        self.assertFalse(sender.add_ice("candidate:closed", 0))
        self.assertFalse(sender.push_jpeg(b"jpeg", 2))

    def test_gi_boxed_answer_owner_survives_serialization_and_async_local_set(self):
        # Model Gst 1.20's borrowed boxed value: a chained get_reply().get_value()
        # loses its owner immediately, before serializing the answer SDP.
        class Reply:
            def has_field(self, name):
                return False

            def get_value(self, name):
                owner = weakref.ref(self)

                def as_text():
                    if owner() is None:
                        raise AssertionError("SDP owner was freed before serialization")
                    return "v=0\r\n"

                return SimpleNamespace(sdp=SimpleNamespace(as_text=as_text), owner=owner)

        sender = GStreamerWebRTCSender()
        sender._Gst = SimpleNamespace(
            PromiseResult=SimpleNamespace(REPLIED="replied"),
            Promise=SimpleNamespace(new_with_change_func=lambda callback, data, _: (callback, data)),
        )
        sender._peer = Mock()
        sender._on_answer = Mock()
        promise = SimpleNamespace(wait=lambda: "replied", get_reply=lambda: Reply())
        sender._answer_created(promise)
        operation, answer, callback_data = sender._peer.emit.call_args.args
        self.assertEqual(operation, "set-local-description")
        del promise
        gc.collect()
        self.assertIsNotNone(answer.owner(), "Native set-local-description can complete asynchronously")
        callback, text = callback_data
        callback(SimpleNamespace(wait=lambda: "replied", get_reply=lambda: None), text)
        sender._on_answer.assert_called_once_with("v=0\r\n")

    def test_dedicated_thread_negotiation_waits_for_each_promise(self):
        answer = SimpleNamespace(sdp=SimpleNamespace(as_text=lambda: "v=0\r\n"))
        answer_reply = SimpleNamespace(has_field=lambda _: False,
                                       get_value=lambda name: answer)
        promises = [
            SimpleNamespace(wait=lambda: "replied", get_reply=lambda: None,
                            interrupt=Mock()),
            SimpleNamespace(wait=lambda: "replied", get_reply=lambda: answer_reply,
                            interrupt=Mock()),
            SimpleNamespace(wait=lambda: "replied", get_reply=lambda: None,
                            interrupt=Mock()),
        ]
        sender = GStreamerWebRTCSender()
        sender._Gst = SimpleNamespace(
            PromiseResult=SimpleNamespace(REPLIED="replied"),
            Promise=SimpleNamespace(new=Mock(side_effect=promises)),
        )
        sender._on_state = Mock()
        sender._on_answer = Mock()
        peer = Mock()
        sender._negotiate(peer, object())
        self.assertEqual([call.args[0] for call in peer.emit.call_args_list], [
            "set-remote-description", "create-answer", "set-local-description"
        ])
        sender._on_answer.assert_called_once_with("v=0\r\n")
        self.assertEqual(sender.stats()["state"], "connecting")


if __name__ == "__main__":
    unittest.main()
