from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper, shape_inference


INPUT_NAME = "images"
RAW_OUTPUT_NAME = "multitask_logits"
INPUT_SHAPE = [1, 3, 240, 1280]
RAW_OUTPUT_SHAPE = [1, 5, 240, 1280]
DEFAULT_PROBABILITY_THRESHOLD = 0.99
CRACK_Y_START = 112
DEFAULT_MIN_COMPONENT_PIXELS = 1024
FINITE_ABS_LIMIT = np.float32(65504.0)


def _shape(value_info: onnx.ValueInfoProto) -> list[int | str | None]:
    result: list[int | str | None] = []
    for dim in value_info.type.tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            result.append(int(dim.dim_value))
        elif dim.HasField("dim_param"):
            result.append(dim.dim_param)
        else:
            result.append(None)
    return result


def _initializer(name: str, value: np.ndarray) -> onnx.TensorProto:
    return numpy_helper.from_array(value, name=name)


def _add_postprocessor(
    model: onnx.ModelProto,
    source_sha256: str,
    probability_threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
    min_component_pixels: int = DEFAULT_MIN_COMPONENT_PIXELS,
) -> None:
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("Crack probability threshold must be between zero and one.")
    if min_component_pixels <= 0:
        raise ValueError("Crack minimum component pixels must be positive.")
    logit_threshold = np.float32(
        math.log(probability_threshold / (1.0 - probability_threshold))
    )
    ai_onnx_opset = next(
        (
            item.version
            for item in model.opset_import
            if item.domain in ("", "ai.onnx")
        ),
        None,
    )
    if ai_onnx_opset is None:
        raise ValueError("Source ONNX must declare the ai.onnx opset.")

    def reduce_min_node(input_name: str, output_name: str, name: str) -> onnx.NodeProto:
        if ai_onnx_opset >= 18:
            return helper.make_node(
                "ReduceMin",
                [input_name, "deploy/all_axes"],
                [output_name],
                name=name,
                keepdims=0,
            )
        return helper.make_node(
            "ReduceMin",
            [input_name],
            [output_name],
            name=name,
            axes=[0, 1, 2, 3],
            keepdims=0,
        )

    graph = model.graph
    if len(graph.input) != 1 or graph.input[0].name != INPUT_NAME:
        raise ValueError("Source ONNX input must be exactly 'images'.")
    if _shape(graph.input[0]) != INPUT_SHAPE:
        raise ValueError(f"Source input shape must be {INPUT_SHAPE}.")
    if len(graph.output) != 1 or graph.output[0].name != RAW_OUTPUT_NAME:
        raise ValueError("Source ONNX output must be exactly 'multitask_logits'.")
    if _shape(graph.output[0]) != RAW_OUTPUT_SHAPE:
        raise ValueError(f"Source output shape must be {RAW_OUTPUT_SHAPE}.")
    if graph.output[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        raise ValueError("Source multitask_logits must be FLOAT.")

    initializers = [
        _initializer("deploy/rust_starts", np.array([0], dtype=np.int64)),
        _initializer("deploy/rust_ends", np.array([4], dtype=np.int64)),
        _initializer("deploy/channel_axis", np.array([1], dtype=np.int64)),
        _initializer("deploy/unit_step", np.array([1], dtype=np.int64)),
        _initializer("deploy/spatial_axes", np.array([0, 1, 2], dtype=np.int64)),
        _initializer("deploy/vector_axis", np.array([0], dtype=np.int64)),
        _initializer("deploy/count_slice_starts", np.array([2], dtype=np.int64)),
        _initializer("deploy/count_slice_ends", np.array([4], dtype=np.int64)),
        _initializer("deploy/crack_starts", np.array([0, 4, 112, 0], dtype=np.int64)),
        _initializer("deploy/crack_ends", np.array([1, 5, 240, 1280], dtype=np.int64)),
        _initializer("deploy/crack_axes", np.array([0, 1, 2, 3], dtype=np.int64)),
        _initializer("deploy/crack_steps", np.array([1, 1, 1, 1], dtype=np.int64)),
        _initializer("deploy/crack_squeeze_axis", np.array([1], dtype=np.int64)),
        _initializer(
            "deploy/logit_threshold", np.array(logit_threshold, dtype=np.float32)
        ),
        _initializer("deploy/int32_zero", np.array(0, dtype=np.int32)),
        _initializer("deploy/finite_abs_limit", np.array(FINITE_ABS_LIMIT, dtype=np.float32)),
    ]
    if ai_onnx_opset >= 18:
        initializers.append(
            _initializer("deploy/all_axes", np.array([0, 1, 2, 3], dtype=np.int64))
        )
    for class_id in range(4):
        initializers.append(
            _initializer(
                f"deploy/class_{class_id}", np.array(class_id, dtype=np.int32)
            )
        )
    graph.initializer.extend(initializers)

    nodes: list[onnx.NodeProto] = [
        helper.make_node(
            "Slice",
            [
                RAW_OUTPUT_NAME,
                "deploy/rust_starts",
                "deploy/rust_ends",
                "deploy/channel_axis",
                "deploy/unit_step",
            ],
            ["deploy/rust_logits"],
            name="deploy/rust_slice",
        ),
        helper.make_node(
            "ArgMax",
            ["deploy/rust_logits"],
            ["deploy/rust_class_i64"],
            name="deploy/rust_argmax",
            axis=1,
            keepdims=0,
            select_last_index=0,
        ),
        helper.make_node(
            "Cast",
            ["deploy/rust_class_i64"],
            ["deploy/rust_class_i32"],
            name="deploy/rust_class_i32",
            to=TensorProto.INT32,
        ),
        helper.make_node(
            "Cast",
            ["deploy/rust_class_i32"],
            ["deploy/rust_class_float"],
            name="deploy/rust_class_float",
            to=TensorProto.FLOAT,
        ),
        helper.make_node(
            "Cast",
            ["deploy/rust_class_float"],
            ["rust_class_map"],
            name="deploy/rust_class_u8",
            to=TensorProto.UINT8,
        ),
    ]

    count_vectors: list[str] = []
    for class_id in range(4):
        equal_name = f"deploy/rust_class_{class_id}_equal"
        int_name = f"deploy/rust_class_{class_id}_i32"
        sum_name = f"deploy/rust_class_{class_id}_sum"
        vector_name = f"deploy/rust_class_{class_id}_count"
        nodes.extend(
            [
                helper.make_node(
                    "Equal",
                    ["deploy/rust_class_i32", f"deploy/class_{class_id}"],
                    [equal_name],
                    name=f"deploy/rust_class_{class_id}_compare",
                ),
                helper.make_node(
                    "Cast",
                    [equal_name],
                    [int_name],
                    name=f"deploy/rust_class_{class_id}_cast",
                    to=TensorProto.INT32,
                ),
                helper.make_node(
                    "ReduceSum",
                    [int_name, "deploy/spatial_axes"],
                    [sum_name],
                    name=f"deploy/rust_class_{class_id}_reduce",
                    keepdims=0,
                ),
                helper.make_node(
                    "Unsqueeze",
                    [sum_name, "deploy/vector_axis"],
                    [vector_name],
                    name=f"deploy/rust_class_{class_id}_unsqueeze",
                ),
            ]
        )
        count_vectors.append(vector_name)

    nodes.extend(
        [
            helper.make_node(
                "Concat",
                count_vectors,
                ["rust_class_counts"],
                name="deploy/rust_counts",
                axis=0,
            ),
            helper.make_node(
                "Slice",
                [
                    "rust_class_counts",
                    "deploy/count_slice_starts",
                    "deploy/count_slice_ends",
                    "deploy/vector_axis",
                    "deploy/unit_step",
                ],
                ["deploy/rust_poor_severe_counts"],
                name="deploy/rust_poor_severe_slice",
            ),
            helper.make_node(
                "ReduceSum",
                ["deploy/rust_poor_severe_counts", "deploy/vector_axis"],
                ["deploy/rust_poor_severe_sum"],
                name="deploy/rust_poor_severe_reduce",
                keepdims=0,
            ),
            helper.make_node(
                "Greater",
                ["deploy/rust_poor_severe_sum", "deploy/int32_zero"],
                ["deploy/rust_poor_severe_bool"],
                name="deploy/rust_poor_severe_compare",
            ),
            helper.make_node(
                "Unsqueeze",
                ["deploy/rust_poor_severe_bool", "deploy/vector_axis"],
                ["deploy/rust_poor_severe_vector"],
                name="deploy/rust_poor_severe_unsqueeze",
            ),
            helper.make_node(
                "Cast",
                ["deploy/rust_poor_severe_vector"],
                ["deploy/rust_poor_severe_float"],
                name="deploy/rust_poor_severe_float",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Cast",
                ["deploy/rust_poor_severe_float"],
                ["rust_poor_severe"],
                name="deploy/rust_poor_severe_u8",
                to=TensorProto.UINT8,
            ),
            helper.make_node(
                "Slice",
                [
                    RAW_OUTPUT_NAME,
                    "deploy/crack_starts",
                    "deploy/crack_ends",
                    "deploy/crack_axes",
                    "deploy/crack_steps",
                ],
                ["deploy/crack_roi_4d"],
                name="deploy/crack_roi_slice",
            ),
            helper.make_node(
                "Squeeze",
                ["deploy/crack_roi_4d", "deploy/crack_squeeze_axis"],
                ["deploy/crack_roi"],
                name="deploy/crack_squeeze",
            ),
            helper.make_node(
                "GreaterOrEqual",
                ["deploy/crack_roi", "deploy/logit_threshold"],
                ["deploy/crack_candidate_bool"],
                name="deploy/crack_logit_threshold",
            ),
            helper.make_node(
                "Cast",
                ["deploy/crack_candidate_bool"],
                ["deploy/crack_candidate_float"],
                name="deploy/crack_candidate_float",
                to=TensorProto.FLOAT,
            ),
            helper.make_node(
                "Cast",
                ["deploy/crack_candidate_float"],
                ["crack_candidate_map"],
                name="deploy/crack_candidate_u8",
                to=TensorProto.UINT8,
            ),
            helper.make_node(
                "Cast",
                ["deploy/crack_candidate_bool"],
                ["deploy/crack_candidate_i32"],
                name="deploy/crack_candidate_i32",
                to=TensorProto.INT32,
            ),
            helper.make_node(
                "ReduceSum",
                ["deploy/crack_candidate_i32", "deploy/spatial_axes"],
                ["deploy/crack_candidate_sum"],
                name="deploy/crack_candidate_reduce",
                keepdims=0,
            ),
            helper.make_node(
                "Unsqueeze",
                ["deploy/crack_candidate_sum", "deploy/vector_axis"],
                ["crack_candidate_pixels"],
                name="deploy/crack_candidate_unsqueeze",
            ),
            helper.make_node(
                "Constant",
                [],
                ["crack_probability_threshold"],
                name="deploy/crack_probability_threshold",
                value=numpy_helper.from_array(
                    np.array([probability_threshold], dtype=np.float32)
                ),
            ),
            helper.make_node(
                "Abs",
                [INPUT_NAME],
                ["deploy/absolute_input"],
                name="deploy/absolute_input",
            ),
            helper.make_node(
                "LessOrEqual",
                ["deploy/absolute_input", "deploy/finite_abs_limit"],
                ["deploy/input_finite_abs_bool"],
                name="deploy/input_finite_abs_compare",
            ),
            helper.make_node(
                "IsNaN",
                [INPUT_NAME],
                ["deploy/input_is_nan_bool"],
                name="deploy/input_is_nan",
            ),
            helper.make_node(
                "Not",
                ["deploy/input_is_nan_bool"],
                ["deploy/input_not_nan_bool"],
                name="deploy/input_not_nan",
            ),
            helper.make_node(
                "And",
                ["deploy/input_finite_abs_bool", "deploy/input_not_nan_bool"],
                ["deploy/input_finite_bool"],
                name="deploy/input_finite_and_not_nan",
            ),
            helper.make_node(
                "Cast",
                ["deploy/input_finite_bool"],
                ["deploy/input_finite_float"],
                name="deploy/input_finite_float",
                to=TensorProto.FLOAT,
            ),
            reduce_min_node(
                "deploy/input_finite_float",
                "deploy/input_finite_scalar",
                "deploy/input_finite_reduce",
            ),
            helper.make_node(
                "Abs",
                [RAW_OUTPUT_NAME],
                ["deploy/absolute_logits"],
                name="deploy/absolute_logits",
            ),
            helper.make_node(
                "LessOrEqual",
                ["deploy/absolute_logits", "deploy/finite_abs_limit"],
                ["deploy/finite_abs_bool"],
                name="deploy/finite_abs_compare",
            ),
            helper.make_node(
                "IsNaN",
                [RAW_OUTPUT_NAME],
                ["deploy/is_nan_bool"],
                name="deploy/is_nan",
            ),
            helper.make_node(
                "Not",
                ["deploy/is_nan_bool"],
                ["deploy/not_nan_bool"],
                name="deploy/not_nan",
            ),
            helper.make_node(
                "And",
                ["deploy/finite_abs_bool", "deploy/not_nan_bool"],
                ["deploy/finite_bool"],
                name="deploy/finite_and_not_nan",
            ),
            helper.make_node(
                "Cast",
                ["deploy/finite_bool"],
                ["deploy/finite_float"],
                name="deploy/finite_float",
                to=TensorProto.FLOAT,
            ),
            reduce_min_node(
                "deploy/finite_float",
                "deploy/logits_finite_scalar",
                "deploy/logits_finite_reduce",
            ),
            helper.make_node(
                "Mul",
                ["deploy/input_finite_scalar", "deploy/logits_finite_scalar"],
                ["deploy/finite_scalar"],
                name="deploy/input_and_logits_finite",
            ),
            helper.make_node(
                "Unsqueeze",
                ["deploy/finite_scalar", "deploy/vector_axis"],
                ["deploy/finite_vector"],
                name="deploy/finite_unsqueeze",
            ),
            helper.make_node(
                "Cast",
                ["deploy/finite_vector"],
                ["multitask_outputs_finite"],
                name="deploy/finite_u8",
                to=TensorProto.UINT8,
            ),
        ]
    )
    graph.node.extend(nodes)

    del graph.output[:]
    graph.output.extend(
        [
            helper.make_tensor_value_info(
                "rust_class_map", TensorProto.UINT8, [1, 240, 1280]
            ),
            helper.make_tensor_value_info(
                "rust_class_counts", TensorProto.INT32, [4]
            ),
            helper.make_tensor_value_info(
                "rust_poor_severe", TensorProto.UINT8, [1]
            ),
            helper.make_tensor_value_info(
                "crack_candidate_map", TensorProto.UINT8, [1, 128, 1280]
            ),
            helper.make_tensor_value_info(
                "crack_candidate_pixels", TensorProto.INT32, [1]
            ),
            helper.make_tensor_value_info(
                "crack_probability_threshold", TensorProto.FLOAT, [1]
            ),
            helper.make_tensor_value_info(
                "multitask_outputs_finite", TensorProto.UINT8, [1]
            ),
        ]
    )

    properties = {item.key: item.value for item in model.metadata_props}
    properties.update(
        {
            "rail_robot.variant": "optimized-maps-scalars-v1",
            "rail_robot.source_sha256": source_sha256,
            "rail_robot.input_preprocessing": (
                "external RGB float32 0..1, ImageNet mean/std"
            ),
            "rail_robot.rust_class_order": "Good,Fair,Poor,Severe",
            "rail_robot.rust_postprocess": "raw argmax channels 0:4",
            "rail_robot.crack_postprocess": (
                f"raw channel 4 rows {CRACK_Y_START}:240, logit >= "
                f"{float(logit_threshold):.15g}"
            ),
            "rail_robot.crack_probability_threshold": str(probability_threshold),
            "rail_robot.crack_connectivity": "8 (external CPU OpenCV)",
            "rail_robot.crack_min_component_pixels": (
                f"{min_component_pixels} (external CPU OpenCV)"
            ),
            "rail_robot.finite_guard": (
                "all normalized inputs and raw logits are not NaN and abs <= "
                f"{float(FINITE_ABS_LIMIT):.1f}"
            ),
        }
    )
    helper.set_model_props(model, properties)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append the rail_robot GPU postprocessing contract to a raw 5-channel ONNX."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--crack-probability-threshold",
        type=float,
        default=DEFAULT_PROBABILITY_THRESHOLD,
        help="embedded crack threshold in probability space (default: 0.99)",
    )
    parser.add_argument(
        "--crack-min-component-pixels",
        type=int,
        default=DEFAULT_MIN_COMPONENT_PIXELS,
        help="external 8-connected component minimum recorded in metadata (default: 1024)",
    )
    parser.add_argument(
        "--approved-source-sha256",
        help="optional required SHA-256 for the raw source ONNX",
    )
    args = parser.parse_args()

    if not 0.0 < args.crack_probability_threshold < 1.0:
        parser.error("--crack-probability-threshold must be between zero and one")
    if args.crack_min_component_pixels <= 0:
        parser.error("--crack-min-component-pixels must be positive")

    source_bytes = args.source.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if args.approved_source_sha256 is not None:
        approved_sha256 = args.approved_source_sha256.lower()
        if len(approved_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in approved_sha256
        ):
            parser.error("--approved-source-sha256 must be 64 hexadecimal characters")
        if source_sha256 != approved_sha256:
            parser.error(
                "source ONNX SHA-256 mismatch: "
                f"expected {approved_sha256}, got {source_sha256}"
            )
    model = onnx.load_from_string(source_bytes)
    onnx.checker.check_model(model)
    _add_postprocessor(
        model,
        source_sha256,
        probability_threshold=args.crack_probability_threshold,
        min_component_pixels=args.crack_min_component_pixels,
    )
    model = shape_inference.infer_shapes(model, strict_mode=True, data_prop=True)
    onnx.checker.check_model(model, full_check=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(args.output))
    output_sha256 = hashlib.sha256(args.output.read_bytes()).hexdigest()
    print(f"source_sha256={source_sha256}")
    print(f"output={args.output.resolve()}")
    print(f"output_sha256={output_sha256}")
    logit_threshold = math.log(
        args.crack_probability_threshold / (1.0 - args.crack_probability_threshold)
    )
    print(f"crack_probability_threshold={args.crack_probability_threshold:.15g}")
    print(f"crack_logit_threshold={logit_threshold:.15g}")
    print(
        f"crack_min_component_pixels={args.crack_min_component_pixels} "
        "(external CPU OpenCV)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
