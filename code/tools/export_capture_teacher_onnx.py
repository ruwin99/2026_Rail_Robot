#!/usr/bin/env python3
"""Convert the pinned Virginia Tech teacher pickle and export fixed-shape ONNX.

The raw ``.pt`` file is a legacy whole-model Python pickle.  Unpickling can
execute code, so this tool calls ``torch.load(..., weights_only=False)`` only
after the bytes read from the same open file have matched the pinned SHA-256.
Do not generalize this script into a loader for arbitrary checkpoints.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import hmac
import io
import importlib
import inspect
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_CHECKPOINT_RELATIVE = Path(
    "models/virginia_tech_cssd/model/"
    "Corrosion Condition State Classification - Trained Model/"
    "var_original_wbatch_2_plus/var_original_wbatch_2_plus_weights_40.pt"
)
OFFICIAL_REPOSITORY_RELATIVE = Path(
    "models/virginia_tech_cssd/official_code"
)
DEFAULT_BUNDLE_RELATIVE = Path(
    "models/virginia_tech_cssd/export/weighted_ce_r101_state_dict.pt"
)
DEFAULT_ONNX_RELATIVE = Path(
    "models/virginia_tech_cssd/export/weighted_ce_r101_fp32_opset18.onnx"
)

EXPECTED_RAW_SHA256 = (
    "9110b28a7d027679076f831d2987d34be3be47883a9d724456810452a32e1dd5"
)
EXPECTED_OFFICIAL_COMMIT = "f3d098b09784ac7e78b160906952e7bc79940fb1"
OFFICIAL_REPOSITORY_URL = (
    "https://github.com/beric7/corrosion_cs_classification.git"
)
SOURCE_MODEL_URL = (
    "https://data.lib.vt.edu/articles/code/"
    "Trained_Model_for_the_Semantic_Segmentation_of_Corrosion_Condition_States/"
    "16628668"
)

ARCHITECTURE = "deeplabv3plus_resnet101"
FACTORY_QUALNAME = "network.modeling.deeplabv3plus_resnet101"
NUM_CLASSES = 4
OUTPUT_STRIDE = 8
CLASS_NAMES = ("Good", "Fair", "Poor", "Severe")
INPUT_NAME = "images"
OUTPUT_NAME = "logits"
INPUT_SHAPE = (1, 3, 512, 512)
OUTPUT_SHAPE = (1, 4, 512, 512)
OPSET_VERSION = 18
EXPECTED_PARAMETER_COUNT = 58_749_604
EQUIVALENCE_SEED = 44_101
ORT_RTOL = 1e-4
ORT_ATOL = 1e-4


class ExportError(RuntimeError):
    """Expected preflight, integrity, conversion, or export failure."""


@dataclass(frozen=True)
class OfficialSource:
    repository: Path
    training_directory: Path
    commit: str
    origin: str | None


@dataclass(frozen=True)
class Dependencies:
    torch: Any
    numpy: Any
    onnx: Any
    onnxscript: Any
    onnxruntime: Any | None
    onnxruntime_import_error: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_project(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _run_git(repository: Path, *arguments: str, check: bool = True) -> str:
    command = [
        "git",
        "-c",
        f"safe.directory={repository}",
        "-C",
        str(repository),
        *arguments,
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise ExportError(
            "Git is required to verify the pinned Virginia Tech source commit."
        ) from exc
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ExportError(
            f"Git source verification failed ({' '.join(arguments)}): {detail}"
        )
    return completed.stdout.strip()


def verify_official_source(repository: Path) -> OfficialSource:
    repository = repository.resolve()
    if not repository.is_dir():
        raise ExportError(
            f"Virginia Tech official-code repository was not found: {repository}"
        )

    training_directory = repository / "Training - Testing"
    required_files = (
        training_directory / "network" / "__init__.py",
        training_directory / "network" / "modeling.py",
        training_directory / "network" / "_deeplab.py",
        training_directory / "network" / "utils.py",
        training_directory / "network" / "backbone" / "resnet.py",
        training_directory / "network" / "backbone" / "mobilenetv2.py",
    )
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing:
        raise ExportError(
            "Pinned official model source is incomplete; missing: "
            + ", ".join(missing)
        )

    commit = _run_git(repository, "rev-parse", "HEAD").lower()
    if commit != EXPECTED_OFFICIAL_COMMIT:
        raise ExportError(
            "Refusing to import unpinned official model code: "
            f"expected commit {EXPECTED_OFFICIAL_COMMIT}, got {commit}."
        )
    worktree_changes = _run_git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
    )
    if worktree_changes:
        raise ExportError(
            "Refusing to import a non-clean official model checkout. "
            f"Changes, including untracked files:\n{worktree_changes}"
        )
    tracked_training_files = set(
        _run_git(
            repository,
            "ls-files",
            "--",
            training_directory.relative_to(repository).as_posix(),
        ).splitlines()
    )
    disk_training_files = {
        path.relative_to(repository).as_posix()
        for path in training_directory.rglob("*")
        if path.is_file()
    }
    ignored_or_untracked_files = sorted(
        disk_training_files - tracked_training_files
    )
    source_symlinks = sorted(
        path.relative_to(repository).as_posix()
        for path in training_directory.rglob("*")
        if path.is_symlink()
    )
    if ignored_or_untracked_files or source_symlinks:
        details = ignored_or_untracked_files + source_symlinks
        raise ExportError(
            "Refusing official source files hidden by Git ignore/exclude rules or "
            "filesystem symlinks:\n" + "\n".join(details[:50])
        )
    origin = _run_git(
        repository,
        "config",
        "--get",
        "remote.origin.url",
        check=False,
    )
    return OfficialSource(
        repository=repository,
        training_directory=training_directory,
        commit=commit,
        origin=origin or None,
    )


def _required_import(name: str, install_hint: str) -> Any:
    try:
        return importlib.import_module(name)
    except (ImportError, OSError) as exc:
        raise ExportError(
            f"Required dependency {name!r} is unavailable. {install_hint} "
            f"Original error: {exc}"
        ) from exc


def load_dependencies(ort_mode: str) -> Dependencies:
    torch = _required_import(
        "torch",
        "Install a current CPU-capable PyTorch build (PyTorch 2.7 or newer).",
    )
    torch_version = _module_version(torch)
    torch_match = re.match(r"^(\d+)\.(\d+)", torch_version)
    if torch_match is None or tuple(map(int, torch_match.groups())) < (2, 7):
        raise ExportError(
            "PyTorch 2.7 or newer is required for this dynamo ONNX export. "
            f"Detected {torch_version!r}; PyTorch 2.6 has known BatchNorm export "
            "failures with some ONNX Script releases."
        )
    numpy = _required_import("numpy", "Install numpy.")
    onnx = _required_import("onnx", "Install onnx.")
    onnxscript = _required_import(
        "onnxscript",
        "Install onnxscript for the torch.export-based ONNX exporter.",
    )

    load_parameters = inspect.signature(torch.load).parameters
    export_parameters = inspect.signature(torch.onnx.export).parameters
    if "weights_only" not in load_parameters:
        raise ExportError(
            "This PyTorch build lacks torch.load(weights_only=...). "
            "Install PyTorch 2.7 or newer."
        )
    if "dynamo" not in export_parameters or "external_data" not in export_parameters:
        raise ExportError(
            "This PyTorch build lacks the required torch.export-based ONNX API. "
            "Install PyTorch 2.7 or newer."
        )

    onnxruntime = None
    ort_error = None
    if ort_mode != "skip":
        try:
            onnxruntime = importlib.import_module("onnxruntime")
        except (ImportError, OSError) as exc:
            ort_error = str(exc)
            if ort_mode == "require":
                raise ExportError(
                    "ONNX Runtime verification was required but onnxruntime is "
                    f"unavailable: {exc}"
                ) from exc
    return Dependencies(
        torch=torch,
        numpy=numpy,
        onnx=onnx,
        onnxscript=onnxscript,
        onnxruntime=onnxruntime,
        onnxruntime_import_error=ort_error,
    )


@contextlib.contextmanager
def _blocked_legacy_torchvision_import() -> Iterator[None]:
    """Supply only the removed helper imported by the pinned legacy source."""

    def block_pretrained_download(*_args: Any, **_kwargs: Any) -> Any:
        raise ExportError(
            "Pretrained backbone download was requested unexpectedly. "
            "The capture teacher must be reconstructed with "
            "pretrained_backbone=False."
        )

    torchvision = types.ModuleType("torchvision")
    torchvision.__path__ = []  # type: ignore[attr-defined]
    models = types.ModuleType("torchvision.models")
    models.__path__ = []  # type: ignore[attr-defined]
    utils = types.ModuleType("torchvision.models.utils")
    utils.load_state_dict_from_url = block_pretrained_download
    torchvision.models = models
    models.utils = utils

    names_and_modules = {
        "torchvision": torchvision,
        "torchvision.models": models,
        "torchvision.models.utils": utils,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in names_and_modules}
    sys.modules.update(names_and_modules)
    try:
        yield
    finally:
        for name, old_module in previous.items():
            if old_module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old_module


def import_official_factory(torch: Any, source: OfficialSource) -> tuple[Any, type[Any]]:
    conflicts = sorted(
        name for name in sys.modules if name == "network" or name.startswith("network.")
    )
    if conflicts:
        raise ExportError(
            "A module named 'network' was imported before the pinned official code. "
            "Run this exporter in a fresh Python process."
        )

    sys.path.insert(0, str(source.training_directory))
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        with _blocked_legacy_torchvision_import():
            modeling = importlib.import_module("network.modeling")
            deeplab = importlib.import_module("network._deeplab")
    except Exception as exc:
        raise ExportError(
            "Could not import the pinned Virginia Tech DeepLabV3+ source. "
            f"Original error: {exc}"
        ) from exc
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode
        try:
            sys.path.remove(str(source.training_directory))
        except ValueError:
            pass

    for module in (modeling, deeplab):
        module_path = Path(module.__file__).resolve()
        try:
            module_path.relative_to(source.training_directory)
        except ValueError as exc:
            raise ExportError(
                f"Imported {module.__name__!r} outside the pinned source tree: "
                f"{module_path}"
            ) from exc

    factory = getattr(modeling, ARCHITECTURE, None)
    legacy_type = getattr(deeplab, "DeepLabV3", None)
    if not callable(factory) or not isinstance(legacy_type, type):
        raise ExportError(
            f"Pinned factory contract {FACTORY_QUALNAME} / DeepLabV3 was not found."
        )
    factory_parameters = inspect.signature(factory).parameters
    expected_parameters = {
        "num_classes",
        "output_stride",
        "pretrained_backbone",
    }
    if not expected_parameters.issubset(factory_parameters):
        raise ExportError(
            f"Pinned factory signature changed unexpectedly: {factory_parameters}"
        )
    return factory, legacy_type


def load_verified_legacy_pickle(torch: Any, checkpoint: Path) -> tuple[Any, str, int]:
    """Hash and unpickle the exact same in-memory bytes, with no unsafe fallback."""

    if not checkpoint.is_file():
        raise ExportError(f"Raw teacher checkpoint was not found: {checkpoint}")
    try:
        with checkpoint.open("rb") as stream:
            checkpoint_bytes = stream.read()
        size_bytes = len(checkpoint_bytes)
        actual_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        if not hmac.compare_digest(actual_sha256, EXPECTED_RAW_SHA256):
            raise ExportError(
                "Refusing to unpickle the legacy teacher because its SHA-256 "
                "does not match the pinned digest: "
                f"expected={EXPECTED_RAW_SHA256}, actual={actual_sha256}."
            )
        verified_stream = io.BytesIO(checkpoint_bytes)
        try:
            # This is intentionally the only weights_only=False call in this tool.
            # These verified bytes are a whole-model pickle from the pinned source.
            legacy_model = torch.load(
                verified_stream,
                map_location="cpu",
                weights_only=False,
            )
        finally:
            verified_stream.close()
            del checkpoint_bytes
    except ExportError:
        raise
    except OSError as exc:
        raise ExportError(f"Could not read raw teacher checkpoint: {exc}") from exc
    except Exception as exc:
        raise ExportError(
            "The pinned legacy teacher could not be deserialized on CPU. "
            f"Original error: {exc}"
        ) from exc
    return legacy_model, actual_sha256, size_bytes


def build_teacher(factory: Any) -> Any:
    try:
        return factory(
            num_classes=NUM_CLASSES,
            output_stride=OUTPUT_STRIDE,
            pretrained_backbone=False,
        )
    except Exception as exc:
        raise ExportError(
            "Could not construct the official DeepLabV3+ ResNet101 model with "
            "pretrained_backbone=False."
        ) from exc


def _validate_model(torch: Any, model: Any, legacy_type: type[Any], label: str) -> int:
    if not isinstance(model, legacy_type):
        raise ExportError(
            f"{label} has unexpected type {type(model)!r}; expected {legacy_type!r}."
        )
    model.eval()
    model.cpu()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise ExportError(
            f"{label} parameter count is {parameter_count:,}; expected "
            f"{EXPECTED_PARAMETER_COUNT:,}."
        )
    for tensor_name, tensor in list(model.named_parameters()) + list(
        model.named_buffers()
    ):
        if tensor.device.type != "cpu":
            raise ExportError(f"{label} tensor {tensor_name!r} is not on CPU.")
        if tensor.is_floating_point() and tensor.dtype != torch.float32:
            raise ExportError(
                f"{label} tensor {tensor_name!r} is {tensor.dtype}, not float32."
            )
    return parameter_count


def make_equivalence_input(torch: Any) -> Any:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(EQUIVALENCE_SEED)
    return torch.randint(
        0,
        256,
        INPUT_SHAPE,
        generator=generator,
        dtype=torch.int32,
        device="cpu",
    ).to(dtype=torch.float32)


def run_torch_equivalence(
    torch: Any,
    legacy_model: Any,
    rebuilt_model: Any,
    example: Any,
) -> dict[str, Any]:
    with torch.inference_mode():
        legacy_logits = legacy_model(example)
        rebuilt_logits = rebuilt_model(example)
    for label, logits in (
        ("legacy", legacy_logits),
        ("rebuilt", rebuilt_logits),
    ):
        if tuple(logits.shape) != OUTPUT_SHAPE:
            raise ExportError(
                f"{label} logits shape is {tuple(logits.shape)}, expected {OUTPUT_SHAPE}."
            )
        if logits.dtype != torch.float32:
            raise ExportError(f"{label} logits dtype is {logits.dtype}, expected float32.")
        if not bool(torch.isfinite(logits).all().item()):
            raise ExportError(f"{label} logits contain NaN or infinity.")

    difference = (legacy_logits - rebuilt_logits).abs()
    max_abs_error = float(difference.max().item())
    mean_abs_error = float(difference.mean().item())
    logits_exact = bool(torch.equal(legacy_logits, rebuilt_logits))
    legacy_argmax = legacy_logits.argmax(dim=1)
    rebuilt_argmax = rebuilt_logits.argmax(dim=1)
    argmax_mismatches = int(
        torch.count_nonzero(legacy_argmax != rebuilt_argmax).item()
    )
    if not logits_exact or argmax_mismatches:
        raise ExportError(
            "Legacy/rebuilt equivalence failed: "
            f"logits_exact={logits_exact}, max_abs_error={max_abs_error}, "
            f"argmax_mismatches={argmax_mismatches}."
        )
    return {
        "input_seed": EQUIVALENCE_SEED,
        "logits_exact": logits_exact,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "argmax_mismatch_pixels": argmax_mismatches,
    }


def _bundle_metadata(
    source_filename: str,
    source_sha256: str,
    source_size_bytes: int,
    official_source: OfficialSource,
    parameter_count: int,
    torch_version: str,
    created_utc: str,
) -> dict[str, Any]:
    return {
        "format_version": 1,
        "state_dict": None,
        "architecture": ARCHITECTURE,
        "factory": FACTORY_QUALNAME,
        "num_classes": NUM_CLASSES,
        "output_stride": OUTPUT_STRIDE,
        "class_names": list(CLASS_NAMES),
        "parameter_count": parameter_count,
        "teacher_preprocessing": {
            "layout": "NCHW",
            "color_order": "BGR",
            "dtype": "float32",
            "value_range": [0.0, 255.0],
            "scale_divisor": None,
            "mean": None,
            "std": None,
            "input_shape": list(INPUT_SHAPE),
        },
        "source_file": source_filename,
        "source_size_bytes": source_size_bytes,
        "source_sha256": source_sha256,
        "source_sha256_provenance": (
            "Pinned digest calculated from the local official-download snapshot; "
            "not a publisher-signed checksum."
        ),
        "source_url": SOURCE_MODEL_URL,
        "official_code_url": OFFICIAL_REPOSITORY_URL,
        "official_code_origin": official_source.origin,
        "official_code_commit": official_source.commit,
        "torch_version": torch_version,
        "created_utc": created_utc,
    }


def _validate_bundle(bundle: Any) -> Any:
    if not isinstance(bundle, dict):
        raise ExportError("Converted teacher bundle is not a dictionary.")
    expected = {
        "format_version": 1,
        "architecture": ARCHITECTURE,
        "factory": FACTORY_QUALNAME,
        "num_classes": NUM_CLASSES,
        "output_stride": OUTPUT_STRIDE,
        "class_names": list(CLASS_NAMES),
        "parameter_count": EXPECTED_PARAMETER_COUNT,
        "source_sha256": EXPECTED_RAW_SHA256,
        "official_code_commit": EXPECTED_OFFICIAL_COMMIT,
    }
    for key, expected_value in expected.items():
        if bundle.get(key) != expected_value:
            raise ExportError(
                f"Converted bundle field {key!r} is {bundle.get(key)!r}; "
                f"expected {expected_value!r}."
            )
    state_dict = bundle.get("state_dict")
    if not isinstance(state_dict, dict) or not state_dict:
        raise ExportError("Converted bundle contains no reusable state_dict.")
    return state_dict


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _static_onnx_shape(value_info: Any) -> tuple[int, ...]:
    dimensions: list[int] = []
    for dimension in value_info.type.tensor_type.shape.dim:
        if not dimension.HasField("dim_value"):
            raise ExportError(
                f"ONNX tensor {value_info.name!r} has a dynamic dimension."
            )
        dimensions.append(int(dimension.dim_value))
    return tuple(dimensions)


def validate_onnx(onnx: Any, onnx_path: Path) -> dict[str, Any]:
    try:
        model = onnx.load(str(onnx_path), load_external_data=False)
    except Exception as exc:
        raise ExportError(f"Could not load the exported ONNX: {exc}") from exc

    external_initializers = [
        initializer.name
        for initializer in model.graph.initializer
        if initializer.data_location == onnx.TensorProto.EXTERNAL
        or bool(initializer.external_data)
    ]
    if external_initializers:
        raise ExportError(
            "ONNX export unexpectedly used external tensor data; a single-file "
            "artifact is required. Initializers: "
            + ", ".join(external_initializers[:10])
        )
    try:
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ExportError(f"ONNX checker rejected the exported model: {exc}") from exc

    inputs = list(model.graph.input)
    outputs = list(model.graph.output)
    if [value.name for value in inputs] != [INPUT_NAME]:
        raise ExportError(
            f"ONNX inputs must be exactly {[INPUT_NAME]}, got "
            f"{[value.name for value in inputs]}."
        )
    if [value.name for value in outputs] != [OUTPUT_NAME]:
        raise ExportError(
            f"ONNX outputs must be exactly {[OUTPUT_NAME]}, got "
            f"{[value.name for value in outputs]}."
        )
    input_shape = _static_onnx_shape(inputs[0])
    output_shape = _static_onnx_shape(outputs[0])
    if input_shape != INPUT_SHAPE or output_shape != OUTPUT_SHAPE:
        raise ExportError(
            "ONNX fixed-shape contract mismatch: "
            f"input={input_shape}, output={output_shape}."
        )
    float_type = onnx.TensorProto.FLOAT
    if inputs[0].type.tensor_type.elem_type != float_type:
        raise ExportError("ONNX input is not FP32.")
    if outputs[0].type.tensor_type.elem_type != float_type:
        raise ExportError("ONNX output is not FP32.")

    default_opsets = [
        int(item.version) for item in model.opset_import if item.domain in ("", "ai.onnx")
    ]
    if default_opsets != [OPSET_VERSION]:
        raise ExportError(
            f"ONNX default opset must be {[OPSET_VERSION]}, got {default_opsets}."
        )
    del model
    gc.collect()
    return {
        "status": "passed",
        "checker": "onnx.checker.check_model",
        "input_name": INPUT_NAME,
        "input_shape": list(input_shape),
        "input_dtype": "float32",
        "output_name": OUTPUT_NAME,
        "output_shape": list(output_shape),
        "output_dtype": "float32",
        "opset": OPSET_VERSION,
        "external_initializers": 0,
    }


def verify_with_onnxruntime(
    dependencies: Dependencies,
    onnx_path: Path,
    example: Any,
    pytorch_logits: Any,
    ort_mode: str,
) -> dict[str, Any]:
    ort = dependencies.onnxruntime
    if ort_mode == "skip":
        return {"status": "skipped_by_cli"}
    if ort is None:
        return {
            "status": "skipped_not_installed",
            "import_error": dependencies.onnxruntime_import_error,
        }

    np = dependencies.numpy
    try:
        session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"],
        )
        outputs = session.run(
            [OUTPUT_NAME],
            {INPUT_NAME: example.detach().cpu().numpy()},
        )
    except Exception as exc:
        raise ExportError(f"ONNX Runtime CPU verification failed: {exc}") from exc
    if len(outputs) != 1:
        raise ExportError(f"ONNX Runtime returned {len(outputs)} outputs, expected one.")
    ort_logits = np.asarray(outputs[0])
    reference = pytorch_logits.detach().cpu().numpy()
    if tuple(ort_logits.shape) != OUTPUT_SHAPE:
        raise ExportError(
            f"ONNX Runtime logits shape is {ort_logits.shape}, expected {OUTPUT_SHAPE}."
        )
    if ort_logits.dtype != np.float32:
        raise ExportError(
            f"ONNX Runtime logits dtype is {ort_logits.dtype}, expected float32."
        )
    if not bool(np.isfinite(ort_logits).all()):
        raise ExportError("ONNX Runtime logits contain NaN or infinity.")

    difference = np.abs(reference - ort_logits)
    max_abs_error = float(difference.max())
    mean_abs_error = float(difference.mean())
    logits_close = bool(
        np.allclose(reference, ort_logits, rtol=ORT_RTOL, atol=ORT_ATOL)
    )
    argmax_mismatches = int(
        np.count_nonzero(
            np.argmax(reference, axis=1) != np.argmax(ort_logits, axis=1)
        )
    )
    if not logits_close or argmax_mismatches:
        raise ExportError(
            "PyTorch/ONNX Runtime equivalence failed: "
            f"allclose={logits_close}, max_abs_error={max_abs_error}, "
            f"argmax_mismatches={argmax_mismatches}."
        )
    return {
        "status": "passed",
        "provider": "CPUExecutionProvider",
        "rtol": ORT_RTOL,
        "atol": ORT_ATOL,
        "logits_allclose": logits_close,
        "max_abs_error": max_abs_error,
        "mean_abs_error": mean_abs_error,
        "argmax_mismatch_pixels": argmax_mismatches,
    }


def _module_version(module: Any) -> str:
    return str(getattr(module, "__version__", "unknown"))


def _validate_destinations(
    checkpoint: Path,
    destinations: tuple[Path, Path, Path],
    overwrite: bool,
) -> None:
    if len(set(destinations)) != len(destinations):
        raise ExportError("Bundle, ONNX, and metadata outputs must be different files.")
    if checkpoint in destinations:
        raise ExportError("An output path must not overwrite the raw legacy checkpoint.")
    for destination in destinations:
        if destination.is_dir():
            raise ExportError(f"Output path is a directory: {destination}")
        if destination.exists() and not overwrite:
            raise ExportError(
                f"Output already exists: {destination}. Pass --overwrite explicitly "
                "to replace generated artifacts."
            )


def _commit_output_set(
    replacements: tuple[tuple[Path, Path], ...],
    overwrite: bool,
) -> None:
    """Commit generated artifacts as one rollback-protected set."""

    backups: dict[Path, Path] = {}
    committed: list[Path] = []
    try:
        for _temporary, destination in replacements:
            if not destination.exists():
                continue
            if not overwrite:
                raise ExportError(
                    f"Output appeared during export and overwrite is disabled: "
                    f"{destination}"
                )
            backup = _temporary_path(destination)
            try:
                os.replace(destination, backup)
            except Exception:
                backup.unlink(missing_ok=True)
                raise
            backups[destination] = backup

        for temporary, destination in replacements:
            os.replace(temporary, destination)
            committed.append(destination)
    except Exception as exc:
        rollback_errors: list[str] = []
        for destination in reversed(committed):
            try:
                destination.unlink(missing_ok=True)
            except OSError as rollback_exc:
                rollback_errors.append(f"remove {destination}: {rollback_exc}")
        for destination, backup in backups.items():
            try:
                os.replace(backup, destination)
            except OSError as rollback_exc:
                rollback_errors.append(
                    f"restore {destination} from {backup}: {rollback_exc}"
                )
        detail = (
            " Rollback errors: " + "; ".join(rollback_errors)
            if rollback_errors
            else " Previous outputs were restored."
        )
        raise ExportError(f"Could not commit the generated artifact set.{detail}") from exc
    else:
        for backup in backups.values():
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass


@contextlib.contextmanager
def _exclusive_output_locks(destinations: tuple[Path, ...]) -> Iterator[None]:
    locks: list[tuple[int, Path]] = []
    try:
        for destination in sorted(destinations, key=lambda path: str(path).lower()):
            lock_path = destination.with_name(f".{destination.name}.export.lock")
            try:
                descriptor = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o600,
                )
            except FileExistsError as exc:
                raise ExportError(
                    "Another exporter may be writing the same artifacts. "
                    f"Lock exists: {lock_path}. Remove a stale lock only after "
                    "confirming no exporter is running."
                ) from exc
            os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
            locks.append((descriptor, lock_path))
        yield
    finally:
        for descriptor, lock_path in reversed(locks):
            try:
                os.close(descriptor)
            finally:
                try:
                    lock_path.unlink(missing_ok=True)
                except OSError:
                    pass


def export_teacher(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint = _resolve_from_project(args.raw_checkpoint)
    official_repository = _resolve_from_project(args.official_code)
    bundle_output = _resolve_from_project(args.bundle_output)
    onnx_output = _resolve_from_project(args.onnx_output)
    metadata_output = (
        _resolve_from_project(args.metadata_output)
        if args.metadata_output is not None
        else Path(f"{onnx_output}.metadata.json")
    )
    _validate_destinations(
        checkpoint,
        (bundle_output, onnx_output, metadata_output),
        args.overwrite,
    )

    dependencies = load_dependencies(args.ort)
    torch = dependencies.torch
    official_source = verify_official_source(official_repository)
    factory, legacy_type = import_official_factory(torch, official_source)
    official_source = verify_official_source(official_repository)

    print(
        "SECURITY: the raw .pt is a Python pickle; unsafe deserialization remains "
        "disabled until its pinned SHA-256 matches."
    )
    legacy_model, source_sha256, source_size_bytes = load_verified_legacy_pickle(
        torch,
        checkpoint,
    )
    print(f"Verified raw checkpoint SHA-256: {source_sha256}")

    bundle_temporary: Path | None = None
    onnx_temporary: Path | None = None
    metadata_temporary: Path | None = None
    try:
        parameter_count = _validate_model(
            torch,
            legacy_model,
            legacy_type,
            "Legacy teacher",
        )
        rebuilt_model = build_teacher(factory)
        _validate_model(torch, rebuilt_model, legacy_type, "Rebuilt teacher")
        try:
            rebuilt_model.load_state_dict(legacy_model.state_dict(), strict=True)
        except Exception as exc:
            raise ExportError(
                f"Strict legacy state_dict load into the official factory failed: {exc}"
            ) from exc
        _validate_model(torch, rebuilt_model, legacy_type, "Loaded rebuilt teacher")

        example = make_equivalence_input(torch)
        legacy_equivalence = run_torch_equivalence(
            torch,
            legacy_model,
            rebuilt_model,
            example,
        )
        print("Legacy/rebuilt logits and argmax equivalence: passed")

        created_utc = datetime.now(timezone.utc).isoformat()
        bundle = _bundle_metadata(
            source_filename=checkpoint.name,
            source_sha256=source_sha256,
            source_size_bytes=source_size_bytes,
            official_source=official_source,
            parameter_count=parameter_count,
            torch_version=_module_version(torch),
            created_utc=created_utc,
        )
        bundle["state_dict"] = legacy_model.state_dict()
        bundle_temporary = _temporary_path(bundle_output)
        try:
            torch.save(bundle, bundle_temporary)
        except Exception as exc:
            raise ExportError(f"Could not save state_dict bundle: {exc}") from exc

        del bundle, legacy_model, rebuilt_model
        gc.collect()

        try:
            safe_bundle = torch.load(
                bundle_temporary,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise ExportError(
                f"Converted bundle could not be reloaded with weights_only=True: {exc}"
            ) from exc
        state_dict = _validate_bundle(safe_bundle)
        export_model = build_teacher(factory)
        try:
            export_model.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            raise ExportError(
                f"Strict state_dict reload from converted bundle failed: {exc}"
            ) from exc
        _validate_model(torch, export_model, legacy_type, "Bundle-reloaded teacher")
        del safe_bundle, state_dict
        gc.collect()

        with torch.inference_mode():
            pytorch_logits = export_model(example)
        if tuple(pytorch_logits.shape) != OUTPUT_SHAPE:
            raise ExportError(
                f"Export reference output shape is {tuple(pytorch_logits.shape)}, "
                f"expected {OUTPUT_SHAPE}."
            )

        onnx_temporary = _temporary_path(onnx_output)
        try:
            torch.onnx.export(
                export_model,
                (example,),
                str(onnx_temporary),
                input_names=[INPUT_NAME],
                output_names=[OUTPUT_NAME],
                opset_version=OPSET_VERSION,
                dynamo=True,
                external_data=False,
            )
        except Exception as exc:
            raise ExportError(f"FP32 ONNX export failed: {exc}") from exc

        onnx_validation = validate_onnx(dependencies.onnx, onnx_temporary)
        print("ONNX checker and fixed I/O contract: passed")
        ort_validation = verify_with_onnxruntime(
            dependencies,
            onnx_temporary,
            example,
            pytorch_logits,
            args.ort,
        )
        print(f"ONNX Runtime verification: {ort_validation['status']}")

        bundle_sha256 = sha256_file(bundle_temporary)
        onnx_sha256 = sha256_file(onnx_temporary)
        metadata = {
            "format_version": 1,
            "artifact_purpose": "capture_teacher",
            "created_utc": created_utc,
            "model": {
                "architecture": ARCHITECTURE,
                "factory": FACTORY_QUALNAME,
                "num_classes": NUM_CLASSES,
                "output_stride": OUTPUT_STRIDE,
                "class_names": list(CLASS_NAMES),
                "parameter_count": parameter_count,
            },
            "source": {
                "file": checkpoint.name,
                "size_bytes": source_size_bytes,
                "sha256": source_sha256,
                "sha256_provenance": (
                    "Pinned digest calculated from the local official-download "
                    "snapshot; not a publisher-signed checksum."
                ),
                "url": SOURCE_MODEL_URL,
                "serialization": (
                    "legacy whole-model Python pickle; loaded once on CPU with "
                    "weights_only=False only after SHA-256 verification"
                ),
            },
            "official_code": {
                "url": OFFICIAL_REPOSITORY_URL,
                "origin": official_source.origin,
                "commit": official_source.commit,
                "tracked_worktree_clean": True,
                "pretrained_backbone": False,
                "pretrained_downloads_blocked": True,
            },
            "preprocessing": {
                "layout": "NCHW",
                "color_order": "BGR",
                "dtype": "float32",
                "value_range": [0.0, 255.0],
                "scale_divisor": None,
                "mean": None,
                "std": None,
            },
            "onnx_contract": onnx_validation,
            "validation": {
                "legacy_vs_rebuilt": legacy_equivalence,
                "onnxruntime": ort_validation,
            },
            "artifacts": {
                "state_dict_bundle": {
                    "path": str(bundle_output),
                    "size_bytes": bundle_temporary.stat().st_size,
                    "sha256": bundle_sha256,
                    "reload": "torch.load(map_location='cpu', weights_only=True)",
                },
                "onnx": {
                    "path": str(onnx_output),
                    "size_bytes": onnx_temporary.stat().st_size,
                    "sha256": onnx_sha256,
                },
            },
            "software": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "torch": _module_version(torch),
                "numpy": _module_version(dependencies.numpy),
                "onnx": _module_version(dependencies.onnx),
                "onnxscript": _module_version(dependencies.onnxscript),
                "onnxruntime": (
                    _module_version(dependencies.onnxruntime)
                    if dependencies.onnxruntime is not None
                    else None
                ),
            },
        }
        metadata_temporary = _temporary_path(metadata_output)
        metadata_temporary.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        metadata_sha256 = sha256_file(metadata_temporary)

        destinations = (bundle_output, onnx_output, metadata_output)
        with _exclusive_output_locks(destinations):
            _validate_destinations(checkpoint, destinations, args.overwrite)
            _commit_output_set(
                (
                    (bundle_temporary, bundle_output),
                    (onnx_temporary, onnx_output),
                    (metadata_temporary, metadata_output),
                ),
                args.overwrite,
            )
        bundle_temporary = None
        onnx_temporary = None
        metadata_temporary = None
        return {
            "bundle": bundle_output,
            "bundle_sha256": bundle_sha256,
            "onnx": onnx_output,
            "onnx_sha256": onnx_sha256,
            "metadata": metadata_output,
            "metadata_sha256": metadata_sha256,
            "ort_status": ort_validation["status"],
        }
    finally:
        for temporary in (bundle_temporary, onnx_temporary, metadata_temporary):
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Verify and convert the pinned Virginia Tech weighted-CE ResNet101 "
            "legacy pickle, then export fixed [1,3,512,512] FP32 ONNX. Relative "
            "paths are resolved from the rail_robot project root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--raw-checkpoint",
        type=Path,
        default=RAW_CHECKPOINT_RELATIVE,
        help=(
            "legacy whole-model .pt pickle; only the SHA-256 pinned in this script "
            "is accepted"
        ),
    )
    parser.add_argument(
        "--official-code",
        type=Path,
        default=OFFICIAL_REPOSITORY_RELATIVE,
        help="clean official Virginia Tech git clone at the pinned commit",
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=DEFAULT_BUNDLE_RELATIVE,
        help="safe reusable state_dict bundle output",
    )
    parser.add_argument(
        "--onnx-output",
        type=Path,
        default=DEFAULT_ONNX_RELATIVE,
        help="single-file FP32 ONNX output",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="metadata JSON output (default: <onnx-output>.metadata.json)",
    )
    parser.add_argument(
        "--ort",
        choices=("auto", "require", "skip"),
        default="auto",
        help=(
            "ONNX Runtime CPU verification policy: auto runs when installed, "
            "require fails if unavailable, skip does not import it"
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly replace existing generated outputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = export_teacher(args)
    except ExportError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("error: export interrupted", file=sys.stderr)
        return 130

    print(f"State-dict bundle: {result['bundle']}")
    print(f"  SHA-256: {result['bundle_sha256']}")
    print(f"FP32 ONNX: {result['onnx']}")
    print(f"  SHA-256: {result['onnx_sha256']}")
    print(f"Metadata JSON: {result['metadata']}")
    print(f"  SHA-256: {result['metadata_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
