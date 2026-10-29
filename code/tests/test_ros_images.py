"""Image transport contracts without hardware or DDS."""
import unittest

import test_ros_vision


class RosImageTransportTests(unittest.TestCase):
    run_contract = test_ros_vision.RosVisionAdapterTests.run_contract

    def test_capture_completion_monotonic_round_trips(self):
        self.run_contract('''
            from node_runtime.ros.images import message_to_frame
            frame = CameraFrame(np.zeros((720, 1280, 3), np.uint8), 7, 10.0, 10_000_000_123)
            message = frame_to_message(frame, 'side', 100.0, 10.0)
            assert message.capture_complete_monotonic_ns == 10_000_000_123
            restored = message_to_frame(message, 100.0, 10.0)
            assert restored.read_completed_monotonic_ns == 10_000_000_123
        ''')

    def test_native_color_and_strided_masks_use_bulk_uint8_transport(self):
        self.run_contract('''
            from array import array
            from node_runtime.ros.images import decode_image
            color = np.arange(720*1280*3, dtype=np.uint8).reshape(720,1280,3)
            mask = np.arange(32*64, dtype=np.uint8).reshape(32,64)[:,::2]
            for pixels, encoding in ((color,'bgr8'), (mask,'mono8')):
                header = Header()
                header.frame_id = 'source-camera'
                message = encode_image(pixels, header)
                assert isinstance(message.data, array) and message.data.typecode == 'B'
                assert message.header is header and message.encoding == encoding
                assert message.step == pixels.shape[1] * (3 if pixels.ndim == 3 else 1)
                decoded = decode_image(message)
                assert np.array_equal(decoded, pixels)
                pixels[:] = 0
                assert np.any(decoded) and any(message.data)
        ''')


if __name__ == '__main__':
    unittest.main()
