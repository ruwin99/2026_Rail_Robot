from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
import tempfile
import unittest
from pathlib import Path


ONNX_AVAILABLE = importlib.util.find_spec("onnx") is not None
ORT_AVAILABLE = importlib.util.find_spec("onnxruntime") is not None

if ONNX_AVAILABLE:
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    TOOLS = Path(__file__).resolve().parents[1] / "tools"
    sys.path.insert(0, str(TOOLS))
    import build_rust_optimized_onnx as builder  # noqa: E402


@unittest.skipUnless(ONNX_AVAILABLE, "onnx is not installed")
class RustOptimizedWrapperTests(unittest.TestCase):
    @staticmethod
    def _source_model():
        initializers = [
            numpy_helper.from_array(np.array([0], dtype=np.int64), name="starts"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="ends"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes"),
            numpy_helper.from_array(np.array([1], dtype=np.int64), name="steps"),
        ]
        nodes = [
            helper.make_node(
                "Slice",
                ["images", "starts", "ends", "axes", "steps"],
                ["first_channel"],
            ),
            helper.make_node(
                "Concat", ["images", "first_channel"], ["logits"], axis=1
            ),
        ]
        graph = helper.make_graph(
            nodes,
            "fixture",
            [
                helper.make_tensor_value_info(
                    "images", TensorProto.FLOAT, [1, 3, 240, 1280]
                )
            ],
            [
                helper.make_tensor_value_info(
                    "logits", TensorProto.FLOAT, [1, 4, 240, 1280]
                )
            ],
            initializer=initializers,
        )
        model = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 18)]
        )
        model.ir_version = 8
        onnx.checker.check_model(model)
        return model

    def test_appends_exact_outputs_without_changing_source_initializers(self) -> None:
        model = self._source_model()
        before = builder._initializer_digests(model)

        builder.add_rust_postprocessor(model, "a" * 64)
        onnx.checker.check_model(model, full_check=True)

        self.assertEqual(
            [output.name for output in model.graph.output],
            [
                builder.RUST_MAP_NAME,
                builder.LOGITS_NOT_NAN_NAME,
                builder.LOGITS_FINITE_ABS_NAME,
            ],
        )
        after = builder._initializer_digests(model)
        self.assertEqual({name: after[name] for name in before}, before)
        metadata = {item.key: item.value for item in model.metadata_props}
        self.assertEqual(metadata["rail_robot.source_sha256"], "a" * 64)
        self.assertEqual(metadata["rail_robot.rust_class_order"], "Good,Fair,Poor,Severe")

    @unittest.skipUnless(ORT_AVAILABLE, "onnxruntime is not installed")
    def test_onnxruntime_matches_all_307200_argmax_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.onnx"
            optimized = Path(directory) / "optimized.onnx"
            source_model = self._source_model()
            onnx.save_model(source_model, str(source))
            builder.add_rust_postprocessor(source_model, "fixture-source")
            onnx.save_model(source_model, str(optimized))
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                builder.verify_with_onnxruntime(source, optimized)

        self.assertIn("307200/307200", output.getvalue())

    @unittest.skipUnless(ORT_AVAILABLE, "onnxruntime is not installed")
    def test_finite_flag_rejects_nan_and_both_infinities(self) -> None:
        import onnxruntime as ort

        with tempfile.TemporaryDirectory() as directory:
            optimized = Path(directory) / "optimized.onnx"
            model = self._source_model()
            builder.add_rust_postprocessor(model, "fixture-source")
            onnx.save_model(model, str(optimized))
            session = ort.InferenceSession(
                str(optimized), providers=["CPUExecutionProvider"]
            )
            flag_names = (
                builder.LOGITS_NOT_NAN_NAME,
                builder.LOGITS_FINITE_ABS_NAME,
            )
            cases = (
                (np.zeros((1, 3, 240, 1280), dtype=np.float32), (1, 1)),
                (np.full((1, 3, 240, 1280), np.nan, dtype=np.float32), (0, 0)),
                (np.full((1, 3, 240, 1280), np.inf, dtype=np.float32), (1, 0)),
                (np.full((1, 3, 240, 1280), -np.inf, dtype=np.float32), (1, 0)),
            )
            for model_input, expected_flags in cases:
                with self.subTest(
                    expected=expected_flags, value=model_input.flat[0]
                ):
                    actual_flags = session.run(
                        list(flag_names), {"images": model_input}
                    )
                    for actual, expected in zip(
                        actual_flags, expected_flags, strict=True
                    ):
                        np.testing.assert_array_equal(
                            actual, np.array([expected], dtype=np.uint8)
                        )

    def test_rejects_wrong_source_sha_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "wrong-source.onnx"
            optimized = Path(directory) / "must-not-exist.onnx"
            onnx.save_model(self._source_model(), str(source))

            with self.assertRaisesRegex(ValueError, "unapproved.*SHA-256"):
                builder.build_optimized_model(source, optimized)

            self.assertFalse(optimized.exists())


if __name__ == "__main__":
    unittest.main()
