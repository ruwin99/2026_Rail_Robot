"""Network upload is independent of camera, inference and STM callbacks."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rclpy.node import Node
from std_msgs.msg import String

from firebase_uploader import upload_dashboard_manifest_from_environment
from ..inspection import validate_db_test_dashboard_manifest
from .common import qos_events, qos_state


class UploadNode(Node):
    """Optional consumer of finalized reports; never receives credentials by topic."""
    def __init__(self, uploader=upload_dashboard_manifest_from_environment, *,
                 node_name="report_upload"):
        super().__init__(node_name)
        self._uploader = uploader
        self._upload_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="report-upload")
        self._seen = set()
        self._pending = []
        self._status = self.create_publisher(String, "inspection/upload_status", qos_events)
        self.create_subscription(String, "inspection/finalized", self._enqueue, qos_state)
        self.create_timer(0.2, self._poll)

    def _enqueue(self, message):
        path = Path(message.data).expanduser()
        key = str(path.resolve())
        if key in self._seen:
            return
        self._seen.add(key)
        self._pending.append((key, self._upload_pool.submit(self._upload, path)))

    def _upload(self, path):
        validate_db_test_dashboard_manifest(path)
        result = self._uploader(path)
        if result is None:
            raise RuntimeError("Firebase upload configuration is unavailable.")
        return result

    def _poll(self):
        remaining = []
        for path, future in self._pending:
            if not future.done():
                remaining.append((path, future))
                continue
            message = String()
            try:
                future.result()
                message.data = "uploaded: " + path
                self.get_logger().info("Dashboard report uploaded: " + path)
            except Exception as exc:
                message.data = "upload failed: " + str(exc)
                # 임무는 이미 끝났으므로 fault로 올리지 않는다. 터미널에서도 실패를 알 수 있게 남긴다.
                self.get_logger().warning("Dashboard report upload failed: " + str(exc))
            self._status.publish(message)
        self._pending = remaining

    def destroy_node(self):
        self._upload_pool.shutdown(wait=False, cancel_futures=True)
        return super().destroy_node()
