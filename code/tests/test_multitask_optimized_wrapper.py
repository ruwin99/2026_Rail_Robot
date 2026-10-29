from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np


try:
    import onnx
    from onnx import TensorProto, helper, numpy_helper, shape_inference
except ImportError:  # pragma: no cover - dependency is optional on non-export hosts
    onnx = None


BUILDER_PATH = (
    Path(__file__).resolve().parents[1] / "tools" / "build_multitask_optimized_onnx.py"
)


@unittest.skipIf(onnx is None, "onnx is not installed")
class MultitaskOptimizedWrapperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        spec = importlib.util.spec_from_file_location("multitask_wrapper_builder", BUILDER_PATH)
        assert spec is not None and spec.loader is not None
        cls.builder = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.builder)

    @staticmethod
    def _source_model() -> "onnx.ModelProto":
        images = helper.make_tensor_value_info(
            "images", TensorProto.FLOAT, [1, 3, 240, 1280]
        )
        logits = helper.make_tensor_value_info(
            "multitask_logits", TensorProto.FLOAT, [1, 5, 240, 1280]
        )
        weights = numpy_helper.from_array(
            np.zeros((5, 3, 1, 1), dtype=np.float32), name="weights"
        )
        bias = numpy_helper.from_array(np.zeros(5, dtype=np.float32), name="bias")
        graph = helper.make_graph(
            [
                helper.make_node(
                    "Conv",
                    ["images", "weights", "bias"],
                    ["multitask_logits"],
                    name="source/conv",
                )
            ],
            "test_source",
            [images],
            [logits],
            [weights, bias],
        )
        return helper.make_model(
            graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=8
        )

    def test_custom_p050_min20_contract(self) -> None:
        model = self._source_model()
        self.builder._add_postprocessor(
            model,
            "e" * 64,
            probability_threshold=0.5,
            min_component_pixels=20,
        )
        model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
        onnx.checker.check_model(model, full_check=True)

        outputs = {item.name: item for item in model.graph.output}
        self.assertEqual(
            set(outputs),
            {
                "rust_class_map",
                "rust_class_counts",
                "rust_poor_severe",
                "crack_candidate_map",
                "crack_candidate_pixels",
                "crack_probability_threshold",
                "multitask_outputs_finite",
            },
        )
        initializers = {
            item.name: numpy_helper.to_array(item) for item in model.graph.initializer
        }
        self.assertEqual(float(initializers["deploy/logit_threshold"]), 0.0)
        metadata = {item.key: item.value for item in model.metadata_props}
        self.assertEqual(metadata["rail_robot.crack_probability_threshold"], "0.5")
        self.assertEqual(
            metadata["rail_robot.crack_min_component_pixels"],
            "20 (external CPU OpenCV)",
        )
        self.assertEqual(metadata["rail_robot.source_sha256"], "e" * 64)

    def test_legacy_defaults_are_preserved(self) -> None:
        model = self._source_model()
        self.builder._add_postprocessor(model, "f" * 64)
        initializers = {
            item.name: numpy_helper.to_array(item) for item in model.graph.initializer
        }
        expected_logit = np.float32(math.log(0.99 / 0.01))
        self.assertEqual(initializers["deploy/logit_threshold"].dtype, np.float32)
        self.assertEqual(
            float(initializers["deploy/logit_threshold"]), float(expected_logit)
        )
        metadata = {item.key: item.value for item in model.metadata_props}
        self.assertEqual(metadata["rail_robot.crack_probability_threshold"], "0.99")
        self.assertEqual(
            metadata["rail_robot.crack_min_component_pixels"],
            "1024 (external CPU OpenCV)",
        )

    def test_invalid_threshold_and_component_are_rejected(self) -> None:
        for threshold in (0.0, 1.0, -0.1, 1.1):
            with self.subTest(threshold=threshold):
                with self.assertRaises(ValueError):
                    self.builder._add_postprocessor(
                        self._source_model(), "a" * 64, threshold, 20
                    )
        with self.assertRaises(ValueError):
            self.builder._add_postprocessor(self._source_model(), "a" * 64, 0.5, 0)


if __name__ == "__main__":
    unittest.main()
