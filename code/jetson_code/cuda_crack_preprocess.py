from __future__ import annotations

import ctypes
import math
import struct
from typing import Any

import numpy as np

# [실시간 균열 GPU 전처리] 하단 ROI 180×1280 BGR uint8을 GPU로 올려 세로 INTER_AREA 180→128,
# BGR→RGB, x/127.5-1 정규화를 한 커널에서 끝내고 TensorRT 입력(1,3,128,1280) FP32에 바로 쓴다.
# CPU 경로(hrsegnet_crack_detector.resize_realtime_roi + preprocess_into)와 비트 단위로 같아야 한다.
# 그래서 OpenCV computeResizeAreaTab의 가중치(double로 계산 후 float 저장)와 ResizeArea의 누적 순서,
# saturate_cast<uchar>의 반올림(가까운 짝수)을 그대로 따른다. 가로 배율은 1이라 열은 독립적이다.

SOURCE_HEIGHT = 180
TARGET_HEIGHT = 128
WIDTH = 1280
CHANNELS = 3
RAW_SHAPE = (SOURCE_HEIGHT, WIDTH, CHANNELS)
INPUT_SHAPE = (1, CHANNELS, TARGET_HEIGHT, WIDTH)
RAW_NBYTES = SOURCE_HEIGHT * WIDTH * CHANNELS
THREADS_PER_BLOCK = 256
KERNEL_NAME = b"crack_area_resize_normalize"


def area_resize_table(
    source: int = SOURCE_HEIGHT, target: int = TARGET_HEIGHT
) -> tuple[list[int], list[int], list[int], list[float]]:
    """OpenCV computeResizeAreaTab: (출력 행별 시작, 개수, 원본 행, float32 가중치)."""
    scale = source / target
    starts: list[int] = []
    counts: list[int] = []
    rows: list[int] = []
    alphas: list[float] = []
    for dst in range(target):
        fsx1 = dst * scale
        fsx2 = fsx1 + scale
        cell = min(scale, source - fsx1)
        sx1 = math.ceil(fsx1)
        sx2 = min(math.floor(fsx2), source - 1)
        sx1 = min(sx1, sx2)
        starts.append(len(rows))
        if sx1 - fsx1 > 1e-3:
            rows.append(sx1 - 1)
            alphas.append(float(np.float32((sx1 - fsx1) / cell)))
        for sx in range(sx1, sx2):
            rows.append(sx)
            alphas.append(float(np.float32(1.0 / cell)))
        if fsx2 - sx2 > 1e-3:
            rows.append(sx2)
            alphas.append(float(np.float32(min(min(fsx2 - sx2, 1.0), cell) / cell)))
        counts.append(len(rows) - starts[-1])
    return starts, counts, rows, alphas


def _float_literal(value: float) -> str:
    # 10진 표기 변환 오차를 없애려고 float32 비트 패턴을 그대로 넣는다.
    return f"__int_as_float(0x{struct.unpack('<I', struct.pack('<f', value))[0]:08x})"


def _kernel_source(fused: bool = False) -> str:
    starts, counts, rows, alphas = area_resize_table()
    accumulate = (
        "sum{c} = __fmaf_rn(beta, (float)pixel[{c}], sum{c});"
        if fused
        else "sum{c} = __fadd_rn(sum{c}, __fmul_rn(beta, (float)pixel[{c}]));"
    )
    body = "\n        ".join(accumulate.format(c=c) for c in range(CHANNELS))
    plane = TARGET_HEIGHT * WIDTH
    return f"""
__constant__ int ROW_START[{TARGET_HEIGHT}] = {{{", ".join(map(str, starts))}}};
__constant__ int ROW_COUNT[{TARGET_HEIGHT}] = {{{", ".join(map(str, counts))}}};
__constant__ int ROW_SOURCE[{len(rows)}] = {{{", ".join(map(str, rows))}}};

__device__ __forceinline__ float row_alpha(int index) {{
    switch (index) {{
{chr(10).join(f"    case {i}: return {_float_literal(a)};" for i, a in enumerate(alphas))}
    default: return 0.0f;
    }}
}}

__device__ __forceinline__ float normalized(float accumulated) {{
    int rounded = __float2int_rn(accumulated);
    rounded = rounded < 0 ? 0 : (rounded > 255 ? 255 : rounded);
    return __fsub_rn(__fdiv_rn((float)rounded, 127.5f), 1.0f);
}}

extern "C" __global__ void crack_area_resize_normalize(
    const unsigned char* bgr,
    float* input) {{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= {plane}) {{
        return;
    }}
    const int y = index / {WIDTH};
    const int x = index - y * {WIDTH};
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f;
    const int start = ROW_START[y];
    for (int k = 0; k < ROW_COUNT[y]; ++k) {{
        const float beta = row_alpha(start + k);
        const unsigned char* pixel = bgr + (ROW_SOURCE[start + k] * {WIDTH} + x) * 3;
        {body}
    }}
    input[index] = normalized(sum2);
    input[{plane} + index] = normalized(sum1);
    input[{2 * plane} + index] = normalized(sum0);
}}
"""


class CudaCrackPreprocessor:
    """Upload a 180-row BGR ROI and write the 128-row normalized NCHW tensor on the TRT stream."""

    def __init__(self, cudart: Any, *, fused: bool = False) -> None:
        self._cudart = cudart
        self._cuda: Any = None
        self._nvrtc: Any = None
        self._module: Any = None
        self._function: Any = None
        self._host_raw_pointer: Any = None
        self._host_raw: np.ndarray | None = None
        self._device_raw: Any = None
        self.fused = bool(fused)
        try:
            from cuda import cuda, nvrtc
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Python with cuda.cuda and cuda.nvrtc is required for "
                "GPU crack preprocessing."
            ) from exc
        self._cuda = cuda
        self._nvrtc = nvrtc
        try:
            self._driver_call(cuda.cuInit(0), "cuInit")
            if int(self._driver_call(cuda.cuCtxGetCurrent(), "cuCtxGetCurrent")) == 0:
                raise RuntimeError("CUDA crack preprocessing requires the active TensorRT context.")
            device = self._driver_call(cuda.cuCtxGetDevice(), "cuCtxGetDevice")
            attribute = cuda.CUdevice_attribute
            major = self._driver_call(
                cuda.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device),
                "cuDeviceGetAttribute(compute major)",
            )
            minor = self._driver_call(
                cuda.cuDeviceGetAttribute(attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device),
                "cuDeviceGetAttribute(compute minor)",
            )
            cubin = self._compile_cubin(int(major), int(minor))
            self._module = self._driver_call(cuda.cuModuleLoadData(np.char.array(cubin)), "cuModuleLoadData")
            self._function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, KERNEL_NAME),
                "cuModuleGetFunction(crack_area_resize_normalize)",
            )
            self._host_raw_pointer = self._runtime_call(
                cudart.cudaHostAlloc(RAW_NBYTES, 0), "cudaHostAlloc(crack uint8 ROI)"
            )
            byte_pointer = ctypes.cast(ctypes.c_void_p(int(self._host_raw_pointer)), ctypes.POINTER(ctypes.c_ubyte))
            self._host_raw = np.ctypeslib.as_array(byte_pointer, shape=(RAW_NBYTES,)).reshape(RAW_SHAPE)
            self._device_raw = self._runtime_call(cudart.cudaMalloc(RAW_NBYTES), "cudaMalloc(crack uint8 ROI)")
        except Exception:
            self.close()
            raise

    def _compile_cubin(self, major: int, minor: int) -> bytes:
        nvrtc = self._nvrtc
        program = self._nvrtc_call(
            nvrtc.nvrtcCreateProgram(_kernel_source(self.fused).encode("utf-8"), b"crack_preprocess.cu", 0, [], []),
            "nvrtcCreateProgram",
        )
        try:
            # --fmad=false: 명시하지 않은 곱셈·덧셈도 FMA로 합치지 않게 해 CPU와 같은 반올림을 유지한다.
            options = [f"--gpu-architecture=sm_{major}{minor}".encode("ascii"), b"--std=c++11", b"--fmad=false"]
            result = nvrtc.nvrtcCompileProgram(program, len(options), options)
            try:
                self._nvrtc_call(result, "nvrtcCompileProgram")
            except RuntimeError as exc:
                size = self._nvrtc_call(nvrtc.nvrtcGetProgramLogSize(program), "nvrtcGetProgramLogSize")
                log = b" " * int(size)
                nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError(f"{exc}: {log.decode('utf-8', 'replace').strip()}") from exc
            size = self._nvrtc_call(nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize")
            cubin = b" " * int(size)
            self._nvrtc_call(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
            return cubin
        finally:
            try:
                nvrtc.nvrtcDestroyProgram(program)
            except Exception:
                pass

    def _driver_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if result[0] != self._cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed with CUDA driver error {result[0]}.")
        return None if len(result) == 1 else result[1]

    def _nvrtc_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if result[0] != self._nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{operation} failed with NVRTC error {result[0]}.")
        return None if len(result) == 1 else result[1]

    def _runtime_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if result[0] != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed with CUDA runtime error {result[0]}.")
        return None if len(result) == 1 else result[1]

    def stage(self, frame: np.ndarray) -> None:
        if self._host_raw is None:
            raise RuntimeError("CUDA crack preprocessor is closed.")
        if not isinstance(frame, np.ndarray) or frame.shape != RAW_SHAPE or frame.dtype != np.uint8:
            raise ValueError(f"CUDA crack input must be uint8 with shape {RAW_SHAPE}.")
        np.copyto(self._host_raw, frame, casting="no")

    def enqueue(self, tensor_pointer: Any, stream: Any) -> None:
        # 같은 stream에 원본 H2D → 커널을 넣는다. 호출자가 이어서 TensorRT를 실행한다.
        if self._function is None or self._device_raw is None:
            raise RuntimeError("CUDA crack preprocessor is closed.")
        cudart = self._cudart
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._device_raw, self._host_raw.ctypes.data, RAW_NBYTES,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream,
            ),
            "cudaMemcpyAsync(crack uint8 ROI)",
        )
        arguments = (
            self._cuda.CUdeviceptr(int(self._device_raw)),
            self._cuda.CUdeviceptr(int(tensor_pointer)),
        )
        grid = (TARGET_HEIGHT * WIDTH + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self._driver_call(
            self._cuda.cuLaunchKernel(
                self._function, grid, 1, 1, THREADS_PER_BLOCK, 1, 1, 0,
                self._cuda.CUstream(int(stream)), (arguments, (None, None)), 0,
            ),
            "cuLaunchKernel(crack_area_resize_normalize)",
        )

    def close(self) -> None:
        cudart = self._cudart
        if self._device_raw is not None:
            try:
                cudart.cudaFree(self._device_raw)
            except Exception:
                pass
            self._device_raw = None
        if self._host_raw_pointer is not None:
            try:
                cudart.cudaFreeHost(self._host_raw_pointer)
            except Exception:
                pass
            self._host_raw_pointer = None
        if self._module is not None and self._cuda is not None:
            try:
                self._cuda.cuModuleUnload(self._module)
            except Exception:
                pass
        self._module = None
        self._function = None
        self._host_raw = None
