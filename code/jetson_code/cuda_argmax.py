from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

# [녹 모델 출력 해석] 엔진이 만든 픽셀별 클래스 점수(logit)를 GPU에서 클래스 번호로 바꾼다.
# 이 코드는 학습 모델이 아닌 결정적 후처리다. 부식 면적/등급 집계는 rust_detector.py에서 이어진다.

THREADS_PER_BLOCK = 256
KERNEL_NAME = b"rust_argmax_finite"


@dataclass(frozen=True)
class CudaArgmaxStageTiming:
    gpu_argmax_ms: float
    class_map_d2h_ms: float


def _kernel_source(class_count: int, pixel_count: int) -> str:
    # [argmax] 같은 픽셀의 모든 클래스 점수 중 최대인 번호를 고른다. 동점은 앞 번호를 유지한다.
    # 예: [2.0, -1.0, 0.5]면 class=0이다. 점수는 확률이 아니며 이 연산에 신뢰도 임계값은 없다.
    # softmax를 생략해도 최대 클래스 순위는 같다. 어떤 채널이든 NaN/Inf면 invalid_flag를 세운다.
    return f"""
__device__ __forceinline__ bool finite_float(float value) {{
    return (__float_as_uint(value) & 0x7f800000u) != 0x7f800000u;
}}

extern "C" __global__ void rust_argmax_finite(
    const float* logits,
    unsigned char* class_map,
    unsigned int* invalid_flag) {{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= {pixel_count}) {{
        return;
    }}

    float best_value = logits[pixel];
    unsigned char best_class = 0;
    bool finite = finite_float(best_value);

#pragma unroll
    for (int class_id = 1; class_id < {class_count}; ++class_id) {{
        const float candidate = logits[class_id * {pixel_count} + pixel];
        finite = finite && finite_float(candidate);
        if (candidate > best_value) {{
            best_value = candidate;
            best_class = (unsigned char)class_id;
        }}
    }}

    class_map[pixel] = best_class;
    if (!finite) {{
        atomicExch(invalid_flag, 1u);
    }}
}}
"""


class CudaArgmaxPostprocessor:
    """Run finite validation and first-tie argmax on TensorRT device logits."""

    def __init__(
        self,
        cudart: Any,
        output_shape: tuple[int, int, int, int],
        *,
        measure_timing: bool = False,
    ) -> None:
        if len(output_shape) != 4 or output_shape[0] != 1:
            raise ValueError(
                "CUDA argmax requires a static NCHW output with batch size one."
            )
        _, class_count, height, width = output_shape
        if class_count <= 1 or height <= 0 or width <= 0:
            raise ValueError(f"Invalid CUDA argmax output shape: {output_shape}")

        self._cudart = cudart
        self.class_count = int(class_count)
        self.height = int(height)
        self.width = int(width)
        self.pixel_count = self.height * self.width
        self.d2h_nbytes = self.pixel_count + np.dtype(np.uint32).itemsize
        self.measure_timing = bool(measure_timing)
        self.last_stage_timing: CudaArgmaxStageTiming | None = None
        self._cuda: Any = None
        self._nvrtc: Any = None
        self._module: Any = None
        self._function: Any = None
        self._device_class_map: Any = None
        self._device_invalid_flag: Any = None
        self._host_class_map_pointer: Any = None
        self._host_invalid_flag_pointer: Any = None
        self._host_class_map: np.ndarray | None = None
        self._host_invalid_flag: np.ndarray | None = None
        self._timing_events: tuple[Any, ...] = ()

        try:
            from cuda import cuda, nvrtc
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Python with cuda.cuda and cuda.nvrtc is required for "
                "GPU rust argmax."
            ) from exc

        self._cuda = cuda
        self._nvrtc = nvrtc
        try:
            self._driver_call(cuda.cuInit(0), "cuInit")
            current_context = self._driver_call(
                cuda.cuCtxGetCurrent(),
                "cuCtxGetCurrent",
            )
            if int(current_context) == 0:
                raise RuntimeError(
                    "CUDA argmax requires the active TensorRT CUDA context."
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
            cubin = self._compile_cubin(int(major), int(minor))
            self._module = self._driver_call(
                cuda.cuModuleLoadData(np.char.array(cubin)),
                "cuModuleLoadData",
            )
            self._function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, KERNEL_NAME),
                "cuModuleGetFunction(rust_argmax_finite)",
            )

            map_nbytes = self.pixel_count * np.dtype(np.uint8).itemsize
            flag_nbytes = np.dtype(np.uint32).itemsize
            self._host_class_map_pointer = self._runtime_call(
                cudart.cudaHostAlloc(map_nbytes, 0),
                "cudaHostAlloc(class map)",
            )
            self._host_invalid_flag_pointer = self._runtime_call(
                cudart.cudaHostAlloc(flag_nbytes, 0),
                "cudaHostAlloc(invalid flag)",
            )
            self._host_class_map = self._host_array(
                self._host_class_map_pointer,
                (self.height, self.width),
                np.uint8,
            )
            self._host_invalid_flag = self._host_array(
                self._host_invalid_flag_pointer,
                (1,),
                np.uint32,
            )
            self._device_class_map = self._runtime_call(
                cudart.cudaMalloc(map_nbytes),
                "cudaMalloc(class map)",
            )
            self._device_invalid_flag = self._runtime_call(
                cudart.cudaMalloc(flag_nbytes),
                "cudaMalloc(invalid flag)",
            )
            if self.measure_timing:
                self._timing_events = tuple(
                    self._runtime_call(
                        cudart.cudaEventCreate(),
                        f"cudaEventCreate(rust argmax {index})",
                    )
                    for index in range(3)
                )
        except Exception:
            self.close()
            raise

    def _compile_cubin(self, major: int, minor: int) -> bytes:
        nvrtc = self._nvrtc
        program = self._nvrtc_call(
            nvrtc.nvrtcCreateProgram(
                _kernel_source(self.class_count, self.pixel_count).encode("utf-8"),
                b"rust_argmax_finite.cu",
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
            ]
            compile_result = nvrtc.nvrtcCompileProgram(
                program,
                len(options),
                options,
            )
            log = self._program_log(program)
            try:
                self._nvrtc_call(compile_result, "nvrtcCompileProgram")
            except RuntimeError as exc:
                raise RuntimeError(f"{exc}{': ' + log if log else ''}") from exc
            cubin_size = self._nvrtc_call(
                nvrtc.nvrtcGetCUBINSize(program),
                "nvrtcGetCUBINSize",
            )
            cubin = b" " * int(cubin_size)
            self._nvrtc_call(
                nvrtc.nvrtcGetCUBIN(program, cubin),
                "nvrtcGetCUBIN",
            )
            return cubin
        finally:
            try:
                self._nvrtc_call(
                    nvrtc.nvrtcDestroyProgram(program),
                    "nvrtcDestroyProgram",
                )
            except Exception:
                pass

    def _program_log(self, program: Any) -> str:
        nvrtc = self._nvrtc
        try:
            log_size = self._nvrtc_call(
                nvrtc.nvrtcGetProgramLogSize(program),
                "nvrtcGetProgramLogSize",
            )
            if int(log_size) <= 1:
                return ""
            log = b" " * int(log_size)
            self._nvrtc_call(
                nvrtc.nvrtcGetProgramLog(program, log),
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
            ctypes.c_void_p(int(pointer)),
            ctypes.POINTER(ctypes.c_ubyte),
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

    def process(self, logits_pointer: Any, stream: Any) -> np.ndarray:
        # [엔진 출력 연결] logits_pointer는 이미 계산한 출력 버퍼다. 여기에 다시 모델을 실행하지 않는다.
        # TensorRT와 같은 stream으로 argmax를 등록해 추론 출력이 준비된 뒤 처리되도록 한다.
        if self._function is None or self._host_class_map is None:
            raise RuntimeError("CUDA argmax postprocessor is closed.")
        self.last_stage_timing = None
        cudart = self._cudart
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[0], stream),
                "cudaEventRecord(rust argmax start)",
            )
        self._runtime_call(
            cudart.cudaMemsetAsync(
                self._device_invalid_flag,
                0,
                np.dtype(np.uint32).itemsize,
                stream,
            ),
            "cudaMemsetAsync(invalid flag)",
        )
        arguments = (
            # 전체 FP32 점수 배열은 GPU에 두고, CPU로 가져올 uint8 클래스 맵과 오류 플래그만 작성한다.
            self._cuda.CUdeviceptr(int(logits_pointer)),
            self._cuda.CUdeviceptr(int(self._device_class_map)),
            self._cuda.CUdeviceptr(int(self._device_invalid_flag)),
        )
        grid_x = (self.pixel_count + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self._driver_call(
            self._cuda.cuLaunchKernel(
                self._function,
                grid_x,
                1,
                1,
                THREADS_PER_BLOCK,
                1,
                1,
                0,
                self._cuda.CUstream(int(stream)),
                (arguments, (None, None, None)),
                0,
            ),
            "cuLaunchKernel(rust_argmax_finite)",
        )
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[1], stream),
                "cudaEventRecord(rust argmax end)",
            )
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._host_class_map.ctypes.data,
                self._device_class_map,
                self._host_class_map.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(class map)",
        )
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._host_invalid_flag.ctypes.data,
                self._device_invalid_flag,
                self._host_invalid_flag.nbytes,
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                stream,
            ),
            "cudaMemcpyAsync(invalid flag)",
        )
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[2], stream),
                "cudaEventRecord(rust class map D2H end)",
            )
        self._runtime_call(
            cudart.cudaStreamSynchronize(stream),
            "cudaStreamSynchronize(GPU argmax)",
        )
        # [완료 확인] 비동기 복사 완료 뒤에야 CPU 배열과 오류 플래그를 읽을 수 있다.
        # NaN/Inf를 검출하면 결과를 '녹 없음'으로 사용하지 않고 아래에서 예외를 낸다.
        if self._timing_events:
            self.last_stage_timing = CudaArgmaxStageTiming(
                gpu_argmax_ms=float(
                    self._runtime_call(
                        cudart.cudaEventElapsedTime(
                            self._timing_events[0], self._timing_events[1]
                        ),
                        "cudaEventElapsedTime(rust argmax)",
                    )
                ),
                class_map_d2h_ms=float(
                    self._runtime_call(
                        cudart.cudaEventElapsedTime(
                            self._timing_events[1], self._timing_events[2]
                        ),
                        "cudaEventElapsedTime(rust class map D2H)",
                    )
                ),
            )
        if int(self._host_invalid_flag[0]) != 0:
            raise ValueError("Segmentation logits contain NaN or infinity.")
        return self._host_class_map.copy()

    def close(self) -> None:
        cudart = self._cudart
        for event in self._timing_events:
            try:
                self._runtime_call(
                    cudart.cudaEventDestroy(event),
                    "cudaEventDestroy(rust argmax)",
                )
            except Exception:
                pass
        self._timing_events = ()
        for attribute in ("_device_class_map", "_device_invalid_flag"):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    self._runtime_call(
                        cudart.cudaFree(allocation),
                        f"cudaFree({attribute})",
                    )
                except Exception:
                    pass
                setattr(self, attribute, None)
        for attribute in (
            "_host_class_map_pointer",
            "_host_invalid_flag_pointer",
        ):
            allocation = getattr(self, attribute, None)
            if allocation is not None:
                try:
                    self._runtime_call(
                        cudart.cudaFreeHost(allocation),
                        f"cudaFreeHost({attribute})",
                    )
                except Exception:
                    pass
                setattr(self, attribute, None)
        if self._module is not None and self._cuda is not None:
            try:
                self._driver_call(
                    self._cuda.cuModuleUnload(self._module),
                    "cuModuleUnload",
                )
            except Exception:
                pass
        self._module = None
        self._function = None
        self._host_class_map = None
        self._host_invalid_flag = None
