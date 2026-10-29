from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

# [균열 GPU 후처리] HrSegNet의 배경/균열 두 logit 차이를 임계값과 비교해 후보 마스크를 만든다.
# 학습 가중치는 변경하지 않는다. 픽셀 후보 이후의 작은 연결 성분 제거/면적 계산은
# hrsegnet_crack_detector.py의 result_from_candidate_map에서 처리한다.

THREADS_PER_BLOCK = 256
THRESHOLD_KERNEL_NAME = b"crack_logit_margin_threshold"
MAX_KERNEL_NAME = b"crack_logit_margin_max"
RESTORE_KERNEL_NAME = b"crack_logit_margin_restore"


def linear_restore_rows(source: int, target: int) -> list[tuple[int, int, float, float]]:
    """OpenCV INTER_LINEAR 세로 좌표: (윗행, 아랫행, 윗행 가중치, 아랫행 가중치)를 float32로 만든다."""
    # fy=(dy+0.5)*src/dst-0.5, 경계 밖 행은 가장자리 행으로 고정한다(cv2.resize와 같음).
    scale = source / target
    rows = []
    for dst in range(target):
        fy = np.float32((dst + 0.5) * scale - 0.5)
        sy = int(np.floor(fy))
        fy = np.float32(fy - np.float32(sy))
        upper = min(max(sy, 0), source - 1)
        lower = min(max(sy + 1, 0), source - 1)
        rows.append((upper, lower, float(np.float32(1.0) - fy), float(fy)))
    return rows


def _float_bits(value: float) -> str:
    return f"0x{np.array(value, dtype=np.float32).view(np.uint32).item():08x}u"


def _restore_kernel_source(
    height: int, width: int, restore_height: int, threshold_literal: str
) -> str:
    # [180행 복원] 0/1 후보를 NEAREST로 늘리면 가는 균열 경계가 계단 모양이 되어 균열 IoU가 약 3%p 떨어진다.
    # 대신 logit 차이를 세로 선형 보간한 뒤 같은 임계값과 비교한다. 가로 배율은 1이다.
    rows = linear_restore_rows(height, restore_height)
    plane = height * width
    upper = ", ".join(str(row[0]) for row in rows)
    lower = ", ".join(str(row[1]) for row in rows)
    upper_weight = ", ".join(_float_bits(row[2]) for row in rows)
    lower_weight = ", ".join(_float_bits(row[3]) for row in rows)
    return f"""
__constant__ int RESTORE_UPPER[{restore_height}] = {{{upper}}};
__constant__ int RESTORE_LOWER[{restore_height}] = {{{lower}}};
__constant__ unsigned int RESTORE_UPPER_WEIGHT[{restore_height}] = {{{upper_weight}}};
__constant__ unsigned int RESTORE_LOWER_WEIGHT[{restore_height}] = {{{lower_weight}}};

extern "C" __global__ void crack_logit_margin_restore(
    const float* logits,
    unsigned char* restored_map) {{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= {restore_height * width}) {{
        return;
    }}
    const int y = index / {width};
    const int x = index - y * {width};
    const int upper = RESTORE_UPPER[y] * {width} + x;
    const int lower = RESTORE_LOWER[y] * {width} + x;
    const float upper_margin = __fsub_rn(logits[{plane} + upper], logits[upper]);
    const float lower_margin = __fsub_rn(logits[{plane} + lower], logits[lower]);
    const float margin = __fadd_rn(
        __fmul_rn(upper_margin, __uint_as_float(RESTORE_UPPER_WEIGHT[y])),
        __fmul_rn(lower_margin, __uint_as_float(RESTORE_LOWER_WEIGHT[y])));
    // 비정상 logit은 임계값 커널의 invalid_flag가 이미 오류로 처리한다.
    restored_map[index] = (finite_float(margin) && margin >= {threshold_literal})
        ? (unsigned char)1 : (unsigned char)0;
}}
"""


@dataclass(frozen=True)
class CudaLogitMarginOutput:
    candidate_map: np.ndarray
    max_margin: float
    threshold_ms: float | None = None
    candidate_d2h_ms: float | None = None


def _threshold_literal(margin_threshold: float) -> str:
    threshold_text = f"{float(np.float32(margin_threshold)):.9g}"
    if "." not in threshold_text and "e" not in threshold_text.lower():
        threshold_text += ".0"
    return f"{threshold_text}f"


def _kernel_source(pixel_count: int, block_count: int, margin_threshold: float) -> str:
    # [확률 대신 점수 차이 비교] margin=crack-background이면 2클래스 softmax의 균열 확률은 sigmoid(margin)이다.
    # 그래서 p>=t를 margin>=log(t/(1-t))로 바꿀 수 있다. t=0.5면 기준0, t=0.55면 약0.2007이다.
    # 여기서는 픽셀마다 지수함수를 계산하지 않고 비교한다. 임계값을 바꿔도 학습 파라미터가 바뀌지는 않는다.
    # 첫 커널은 후보(0/1)·블록별 최대차·오류를, 두 번째는 전체 최대차를 구한다. CUDA 소스는 아래 문자열이다.
    threshold_literal = _threshold_literal(margin_threshold)
    return f"""
__device__ __forceinline__ bool finite_float(float value) {{
    return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}}

extern "C" __global__ void crack_logit_margin_threshold(
    const float* logits,
    unsigned char* candidate_map,
    float* block_maxima,
    unsigned int* invalid_flag) {{
    __shared__ float maxima[{THREADS_PER_BLOCK}];
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    float margin = -3.402823466e+38F;

    if (pixel < {pixel_count}) {{
        const float background = logits[pixel];
        const float crack = logits[{pixel_count} + pixel];
        margin = crack - background;
        const bool finite = finite_float(background) && finite_float(crack)
            && finite_float(margin);
        if (!finite) {{
            atomicExch(invalid_flag, 1u);
            margin = -3.402823466e+38F;
        }}
        candidate_map[pixel] =
            (finite && margin >= {threshold_literal}) ? (unsigned char)1 : (unsigned char)0;
    }}

    maxima[threadIdx.x] = margin;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {{
        if (threadIdx.x < offset) {{
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) {{
        block_maxima[blockIdx.x] = maxima[0];
    }}
}}

extern "C" __global__ void crack_logit_margin_max(
    const float* block_maxima,
    float* max_margin) {{
    __shared__ float maxima[{THREADS_PER_BLOCK}];
    float value = -3.402823466e+38F;
    for (int index = threadIdx.x; index < {block_count}; index += blockDim.x) {{
        value = fmaxf(value, block_maxima[index]);
    }}
    maxima[threadIdx.x] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {{
        if (threadIdx.x < offset) {{
            maxima[threadIdx.x] = fmaxf(maxima[threadIdx.x], maxima[threadIdx.x + offset]);
        }}
        __syncthreads();
    }}
    if (threadIdx.x == 0) {{
        max_margin[0] = maxima[0];
    }}
}}
"""


class CudaLogitMarginPostprocessor:
    """Threshold two-class NCHW logits on the TensorRT CUDA stream."""

    def __init__(
        self,
        cudart: Any,
        output_shape: tuple[int, int, int, int],
        margin_threshold: float,
        *,
        measure_timing: bool = False,
        restore_height: int | None = None,
    ) -> None:
        if len(output_shape) != 4 or output_shape[0] != 1 or output_shape[1] != 2:
            raise ValueError(
                "CUDA logit-margin threshold requires static [1,2,H,W] logits."
            )
        _, _, height, width = output_shape
        if height <= 0 or width <= 0 or not np.isfinite(margin_threshold):
            raise ValueError("Invalid CUDA logit-margin output shape or threshold.")
        if restore_height is not None and (
            isinstance(restore_height, bool) or not isinstance(restore_height, int) or restore_height <= height
        ):
            raise ValueError("restore_height must be an integer larger than the logit height.")
        self.restore_height = restore_height

        self._cudart = cudart
        self.height = int(height)
        self.width = int(width)
        self.pixel_count = self.height * self.width
        self.block_count = (
            self.pixel_count + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK
        self.d2h_nbytes = self.pixel_count + 2 * np.dtype(np.uint32).itemsize
        self._measure_timing = bool(measure_timing)
        self._cuda: Any = None
        self._nvrtc: Any = None
        self._module: Any = None
        self._threshold_function: Any = None
        self._max_function: Any = None
        self._restore_function: Any = None
        self._device_restored_map: Any = None
        self._host_restored_pointer: Any = None
        self._host_restored: np.ndarray | None = None
        self._device_candidate_map: Any = None
        self._device_block_maxima: Any = None
        self._device_max_margin: Any = None
        self._device_invalid_flag: Any = None
        self._host_candidate_pointer: Any = None
        self._host_max_pointer: Any = None
        self._host_invalid_pointer: Any = None
        self._host_candidate: np.ndarray | None = None
        self._host_max: np.ndarray | None = None
        self._host_invalid: np.ndarray | None = None
        self._timing_events: tuple[Any, ...] = ()

        try:
            from cuda import cuda, nvrtc
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Python with cuda.cuda and cuda.nvrtc is required for "
                "GPU logit-margin thresholding."
            ) from exc
        self._cuda = cuda
        self._nvrtc = nvrtc

        try:
            self._driver_call(cuda.cuInit(0), "cuInit")
            current_context = self._driver_call(cuda.cuCtxGetCurrent(), "cuCtxGetCurrent")
            if int(current_context) == 0:
                raise RuntimeError(
                    "CUDA logit-margin threshold requires the active TensorRT context."
                )
            device = self._driver_call(cuda.cuCtxGetDevice(), "cuCtxGetDevice")
            major = self._driver_call(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                    device,
                ),
                "cuDeviceGetAttribute(compute major)",
            )
            minor = self._driver_call(
                cuda.cuDeviceGetAttribute(
                    cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                    device,
                ),
                "cuDeviceGetAttribute(compute minor)",
            )
            cubin = self._compile_cubin(int(major), int(minor), float(margin_threshold))
            self._module = self._driver_call(
                cuda.cuModuleLoadData(np.char.array(cubin)), "cuModuleLoadData"
            )
            self._threshold_function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, THRESHOLD_KERNEL_NAME),
                "cuModuleGetFunction(crack_logit_margin_threshold)",
            )
            self._max_function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, MAX_KERNEL_NAME),
                "cuModuleGetFunction(crack_logit_margin_max)",
            )
            if self.restore_height is not None:
                restored_nbytes = self.restore_height * self.width
                self._restore_function = self._driver_call(
                    cuda.cuModuleGetFunction(self._module, RESTORE_KERNEL_NAME),
                    "cuModuleGetFunction(crack_logit_margin_restore)",
                )
                self._host_restored_pointer = self._runtime_call(
                    cudart.cudaHostAlloc(restored_nbytes, 0),
                    "cudaHostAlloc(crack restored map)",
                )
                self._host_restored = self._host_array(
                    self._host_restored_pointer, (self.restore_height, self.width), np.uint8
                )
                self._device_restored_map = self._runtime_call(
                    cudart.cudaMalloc(restored_nbytes), "cudaMalloc(crack restored map)"
                )

            candidate_nbytes = self.pixel_count * np.dtype(np.uint8).itemsize
            block_nbytes = self.block_count * np.dtype(np.float32).itemsize
            scalar_nbytes = np.dtype(np.uint32).itemsize
            self._host_candidate_pointer = self._runtime_call(
                cudart.cudaHostAlloc(candidate_nbytes, 0),
                "cudaHostAlloc(crack candidate map)",
            )
            self._host_max_pointer = self._runtime_call(
                cudart.cudaHostAlloc(scalar_nbytes, 0),
                "cudaHostAlloc(crack max margin)",
            )
            self._host_invalid_pointer = self._runtime_call(
                cudart.cudaHostAlloc(scalar_nbytes, 0),
                "cudaHostAlloc(crack invalid flag)",
            )
            self._host_candidate = self._host_array(
                self._host_candidate_pointer, (self.height, self.width), np.uint8
            )
            self._host_max = self._host_array(
                self._host_max_pointer, (1,), np.float32
            )
            self._host_invalid = self._host_array(
                self._host_invalid_pointer, (1,), np.uint32
            )
            self._device_candidate_map = self._runtime_call(
                cudart.cudaMalloc(candidate_nbytes), "cudaMalloc(crack candidate map)"
            )
            self._device_block_maxima = self._runtime_call(
                cudart.cudaMalloc(block_nbytes), "cudaMalloc(crack block maxima)"
            )
            self._device_max_margin = self._runtime_call(
                cudart.cudaMalloc(scalar_nbytes), "cudaMalloc(crack max margin)"
            )
            self._device_invalid_flag = self._runtime_call(
                cudart.cudaMalloc(scalar_nbytes), "cudaMalloc(crack invalid flag)"
            )
            if self._measure_timing:
                self._timing_events = tuple(
                    self._runtime_call(cudart.cudaEventCreate(), "cudaEventCreate")
                    for _ in range(4)
                )
        except Exception:
            self.close()
            raise

    def _compile_cubin(self, major: int, minor: int, margin_threshold: float) -> bytes:
        nvrtc = self._nvrtc
        program = self._nvrtc_call(
            nvrtc.nvrtcCreateProgram(
                (
                    _kernel_source(self.pixel_count, self.block_count, margin_threshold)
                    + (
                        ""
                        if self.restore_height is None
                        else _restore_kernel_source(
                            self.height,
                            self.width,
                            self.restore_height,
                            _threshold_literal(margin_threshold),
                        )
                    )
                ).encode("utf-8"),
                b"crack_logit_margin.cu",
                0,
                [],
                [],
            ),
            "nvrtcCreateProgram",
        )
        try:
            options = [
                f"--gpu-architecture=sm_{major}{minor}".encode("ascii"),
                b"--std=c++11",
                # 복원 보간의 곱·덧셈을 FMA로 합치지 않아 CPU 기준(cv2)과 같은 반올림에 가깝게 한다.
                b"--fmad=false",
            ]
            compile_result = nvrtc.nvrtcCompileProgram(program, len(options), options)
            log = self._program_log(program)
            try:
                self._nvrtc_call(compile_result, "nvrtcCompileProgram")
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}{': ' + log if log else ''}") from exc
            cubin_size = self._nvrtc_call(
                nvrtc.nvrtcGetCUBINSize(program), "nvrtcGetCUBINSize"
            )
            cubin = b" " * int(cubin_size)
            self._nvrtc_call(nvrtc.nvrtcGetCUBIN(program, cubin), "nvrtcGetCUBIN")
            return cubin
        finally:
            try:
                self._nvrtc_call(
                    nvrtc.nvrtcDestroyProgram(program), "nvrtcDestroyProgram"
                )
            except Exception:
                pass

    def _program_log(self, program: Any) -> str:
        try:
            log_size = self._nvrtc_call(
                self._nvrtc.nvrtcGetProgramLogSize(program),
                "nvrtcGetProgramLogSize",
            )
            if int(log_size) <= 1:
                return ""
            log = b" " * int(log_size)
            self._nvrtc_call(
                self._nvrtc.nvrtcGetProgramLog(program, log),
                "nvrtcGetProgramLog",
            )
            return log.decode("utf-8", errors="replace").strip("\x00 \n")
        except Exception:
            return ""

    @staticmethod
    def _host_array(pointer: Any, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        dtype = np.dtype(dtype)
        byte_count = int(np.prod(shape)) * dtype.itemsize
        byte_pointer = ctypes.cast(
            ctypes.c_void_p(int(pointer)), ctypes.POINTER(ctypes.c_ubyte)
        )
        byte_array = np.ctypeslib.as_array(byte_pointer, shape=(byte_count,))
        return byte_array.view(dtype).reshape(shape)

    def _driver_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA driver status.")
        error = result[0]
        if error != self._cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"{operation} failed with CUDA driver error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    def _nvrtc_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no NVRTC status.")
        error = result[0]
        if error != self._nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError(f"{operation} failed with NVRTC error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    def _runtime_call(self, result: tuple[Any, ...], operation: str) -> Any:
        if not result:
            raise RuntimeError(f"{operation} returned no CUDA runtime status.")
        error = result[0]
        if error != self._cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"{operation} failed with CUDA runtime error {error}.")
        if len(result) == 1:
            return None
        if len(result) == 2:
            return result[1]
        return result[1:]

    def _elapsed_ms(self, start: Any, end: Any) -> float:
        return float(
            self._runtime_call(
                self._cudart.cudaEventElapsedTime(start, end),
                "cudaEventElapsedTime",
            )
        )

    def process(
        self, logits_pointer: Any, stream: Any, *, restore: bool = False
    ) -> CudaLogitMarginOutput:
        # [출력 버퍼 후처리] 입력 계약은 [1,2,H,W] FP32 logit이며 채널0=배경, 채널1=균열이다.
        # 같은 TensorRT stream에서 임계화 → 최대값 축약 → 후보/최대값/오류의 CPU 복사 순으로 실행한다.
        # max_margin은 '가장 강한 한 픽셀' 정보다. 균열 면적이나 이미지 전체의 신뢰도로 해석하지 않는다.
        if self._threshold_function is None or self._host_candidate is None:
            raise RuntimeError("CUDA logit-margin postprocessor is closed.")
        cudart = self._cudart
        scalar_nbytes = np.dtype(np.uint32).itemsize
        self._runtime_call(
            cudart.cudaMemsetAsync(
                self._device_invalid_flag, 0, scalar_nbytes, stream
            ),
            "cudaMemsetAsync(crack invalid flag)",
        )
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[0], stream),
                "cudaEventRecord(threshold start)",
            )
        threshold_arguments = (
            self._cuda.CUdeviceptr(int(logits_pointer)),
            self._cuda.CUdeviceptr(int(self._device_candidate_map)),
            self._cuda.CUdeviceptr(int(self._device_block_maxima)),
            self._cuda.CUdeviceptr(int(self._device_invalid_flag)),
        )
        self._driver_call(
            self._cuda.cuLaunchKernel(
                self._threshold_function,
                self.block_count,
                1,
                1,
                THREADS_PER_BLOCK,
                1,
                1,
                0,
                self._cuda.CUstream(int(stream)),
                (threshold_arguments, (None, None, None, None)),
                0,
            ),
            "cuLaunchKernel(crack_logit_margin_threshold)",
        )
        max_arguments = (
            # 여러 thread block의 최대값을 한 값으로 줄이는 단계다. 후보 면적을 합산하는 연산이 아니다.
            self._cuda.CUdeviceptr(int(self._device_block_maxima)),
            self._cuda.CUdeviceptr(int(self._device_max_margin)),
        )
        self._driver_call(
            self._cuda.cuLaunchKernel(
                self._max_function,
                1,
                1,
                1,
                THREADS_PER_BLOCK,
                1,
                1,
                0,
                self._cuda.CUstream(int(stream)),
                (max_arguments, (None, None)),
                0,
            ),
            "cuLaunchKernel(crack_logit_margin_max)",
        )
        if restore:
            # 180행 후보는 logit 차이를 선형 보간해 다시 임계화한다. 128행 후보는 쓰지 않는다.
            if self._restore_function is None:
                raise RuntimeError("CUDA logit-margin restore was not configured.")
            restore_pixels = self.restore_height * self.width
            self._driver_call(
                self._cuda.cuLaunchKernel(
                    self._restore_function,
                    (restore_pixels + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK,
                    1,
                    1,
                    THREADS_PER_BLOCK,
                    1,
                    1,
                    0,
                    self._cuda.CUstream(int(stream)),
                    (
                        (
                            self._cuda.CUdeviceptr(int(logits_pointer)),
                            self._cuda.CUdeviceptr(int(self._device_restored_map)),
                        ),
                        (None, None),
                    ),
                    0,
                ),
                "cuLaunchKernel(crack_logit_margin_restore)",
            )
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[1], stream),
                "cudaEventRecord(threshold end)",
            )
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[2], stream),
                "cudaEventRecord(candidate D2H start)",
            )
        for host, device, operation in (
            # 전체 두 채널 logit 대신 uint8 후보와 작은 진단값만 돌려받아 D2H 전송량을 줄인다.
            (
                (self._host_restored, self._device_restored_map, "restored candidate map")
                if restore
                else (self._host_candidate, self._device_candidate_map, "candidate map")
            ),
            (self._host_max, self._device_max_margin, "max margin"),
            (self._host_invalid, self._device_invalid_flag, "invalid flag"),
        ):
            self._runtime_call(
                cudart.cudaMemcpyAsync(
                    host.ctypes.data,
                    device,
                    host.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    stream,
                ),
                f"cudaMemcpyAsync({operation})",
            )
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[3], stream),
                "cudaEventRecord(candidate D2H end)",
            )
        self._runtime_call(
            cudart.cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(GPU logit-margin threshold)",
        )
        if int(self._host_invalid[0]) != 0:
            raise ValueError(
                "HrSegNet logits or their logit margin contain NaN or infinity."
            )
        threshold_ms = None
        candidate_d2h_ms = None
        if self._timing_events:
            threshold_ms = self._elapsed_ms(
                self._timing_events[0], self._timing_events[1]
            )
            candidate_d2h_ms = self._elapsed_ms(
                self._timing_events[2], self._timing_events[3]
            )
        return CudaLogitMarginOutput(
            candidate_map=(self._host_restored if restore else self._host_candidate).copy(),
            max_margin=float(self._host_max[0]),
            threshold_ms=threshold_ms,
            candidate_d2h_ms=candidate_d2h_ms,
        )

    def close(self) -> None:
        cudart = self._cudart
        for event in self._timing_events:
            try:
                self._runtime_call(cudart.cudaEventDestroy(event), "cudaEventDestroy")
            except Exception:
                pass
        self._timing_events = ()
        for attribute in (
            "_device_candidate_map",
            "_device_restored_map",
            "_device_block_maxima",
            "_device_max_margin",
            "_device_invalid_flag",
        ):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    self._runtime_call(cudart.cudaFree(allocation), f"cudaFree({attribute})")
                except Exception:
                    pass
                setattr(self, attribute, None)
        for attribute in (
            "_host_candidate_pointer",
            "_host_restored_pointer",
            "_host_max_pointer",
            "_host_invalid_pointer",
        ):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    self._runtime_call(
                        cudart.cudaFreeHost(allocation), f"cudaFreeHost({attribute})"
                    )
                except Exception:
                    pass
                setattr(self, attribute, None)
        if self._module is not None and self._cuda is not None:
            try:
                self._driver_call(
                    self._cuda.cuModuleUnload(self._module), "cuModuleUnload"
                )
            except Exception:
                pass
        self._module = None
        self._threshold_function = None
        self._max_function = None
        self._restore_function = None
        self._host_restored = None
        self._host_candidate = None
        self._host_max = None
        self._host_invalid = None
