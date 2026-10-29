from __future__ import annotations

import ctypes
from dataclasses import dataclass
from typing import Any

import numpy as np

# [녹 모델 GPU 전처리] 학습된 가중치를 계산/변경하는 파일이 아니라 사진을 모델 입력 형식으로 바꾸는 코드다.
# rust_detector.py가 선택한 실시간 경로에서 uint8 BGR → RGB 정규화 → FP32 NCHW 입력 버퍼를 만든다.
# 카메라의 색 순서/ROI 크기/정규화가 학습 전제와 다르면 엔진이 정상 실행돼도 판정이 달라질 수 있다.

INPUT_SHAPE = (1, 3, 240, 1280)
RAW_SHAPE = (240, 1280, 3)
THREADS_PER_BLOCK = 256
KERNEL_NAME = b"rust_bgr_u8_to_imagenet_rgb_nchw"
RAW_NBYTES = int(np.prod(RAW_SHAPE)) * np.dtype(np.uint8).itemsize


@dataclass(frozen=True)
class CudaRustPreprocessTiming:
    input_u8_h2d_ms: float
    gpu_preprocess_ms: float


def _kernel_source(pixel_count: int) -> str:
    """Return a fixed-literal kernel; every CUDA float literal is valid NVRTC C++."""
    # [픽셀 변환 수식] R=(BGR의 R/255 - 0.485)/0.229이며 G/B도 각 평균·표준편차를 사용한다.
    # 이 상수는 전처리 설정이지 이번 추론에서 학습하는 파라미터가 아니다. 값은 음수/1초과도 정상이다.
    # HWC에서 섞여 있는 색을 R평면→G평면→B평면으로 기록한다. N=1이라 입력은 [1,3,240,1280]이다.

    return f"""
extern "C" __global__ void rust_bgr_u8_to_imagenet_rgb_nchw(
    const unsigned char* bgr,
    float* input) {{
    const int pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= {pixel_count}) {{
        return;
    }}

    const int source = pixel * 3;
    const float red = ((float)bgr[source + 2]) / 255.0f;
    const float green = ((float)bgr[source + 1]) / 255.0f;
    const float blue = ((float)bgr[source]) / 255.0f;
    input[pixel] = (red - 0.485f) / 0.229f;
    input[{pixel_count} + pixel] = (green - 0.456f) / 0.224f;
    input[{2 * pixel_count} + pixel] = (blue - 0.406f) / 0.225f;
}}
"""


class CudaRustPreprocessor:
    """Upload native BGR uint8 and write normalized FP32 NCHW on a TRT stream."""
    # 이 커널의 shape는 고정이다. 전체 촬영 teacher 입력을 이 경로에 넣거나 임의 resize하지 않는다.
    # tensor_pointer는 detector가 확보한 입력 메모리 주소이며 학습 가중치 주소가 아니다.

    def __init__(
        self,
        cudart: Any,
        input_shape: tuple[int, int, int, int],
        *,
        measure_timing: bool = False,
    ) -> None:
        if tuple(input_shape) != INPUT_SHAPE:
            raise ValueError(
                f"CUDA Rust preprocessing requires static input {INPUT_SHAPE}."
            )

        self._cudart = cudart
        self._measure_timing = bool(measure_timing)
        self.pixel_count = RAW_SHAPE[0] * RAW_SHAPE[1]
        self._cuda: Any = None
        self._nvrtc: Any = None
        self._module: Any = None
        self._function: Any = None
        self._host_raw_pointer: Any = None
        self._host_raw: np.ndarray | None = None
        self._device_raw: Any = None
        self._timing_events: tuple[Any, ...] = ()

        try:
            from cuda import cuda, nvrtc
        except ImportError as exc:
            raise RuntimeError(
                "CUDA Python with cuda.cuda and cuda.nvrtc is required for "
                "GPU Rust preprocessing."
            ) from exc

        self._cuda = cuda
        self._nvrtc = nvrtc
        try:
            self._driver_call(cuda.cuInit(0), "cuInit")
            current_context = self._driver_call(
                cuda.cuCtxGetCurrent(), "cuCtxGetCurrent"
            )
            if int(current_context) == 0:
                raise RuntimeError(
                    "CUDA Rust preprocessing requires the active TensorRT context."
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
                cuda.cuModuleLoadData(np.char.array(cubin)), "cuModuleLoadData"
            )
            self._function = self._driver_call(
                cuda.cuModuleGetFunction(self._module, KERNEL_NAME),
                "cuModuleGetFunction(rust_bgr_u8_to_imagenet_rgb_nchw)",
            )
            self._host_raw_pointer = self._runtime_call(
                cudart.cudaHostAlloc(RAW_NBYTES, 0),
                "cudaHostAlloc(Rust uint8 input)",
            )
            self._host_raw = self._host_array(
                self._host_raw_pointer, RAW_SHAPE, np.uint8
            )
            self._device_raw = self._runtime_call(
                cudart.cudaMalloc(RAW_NBYTES), "cudaMalloc(Rust uint8 input)"
            )
            if self._measure_timing:
                self._timing_events = tuple(
                    self._runtime_call(
                        cudart.cudaEventCreate(), f"cudaEventCreate(Rust {name})"
                    )
                    for name in ("u8 H2D start", "u8 H2D end", "preprocess end")
                )
        except Exception:
            self.close()
            raise

    def _compile_cubin(self, major: int, minor: int) -> bytes:
        nvrtc = self._nvrtc
        program = self._nvrtc_call(
            nvrtc.nvrtcCreateProgram(
                _kernel_source(self.pixel_count).encode("utf-8"),
                b"rust_cuda_preprocess.cu",
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
                program, len(options), options
            )
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

    def stage(self, frame: np.ndarray) -> None:
        # [CPU 준비] ROI 크기와 uint8을 검사하고 페이지 고정 호스트 버퍼로 복사한다.
        # 여기서는 GPU 계산을 시작하지 않는다. 잘못된 크기는 상위 ROI 선택/카메라 설정부터 확인한다.
        if self._host_raw is None:
            raise RuntimeError("CUDA Rust preprocessor is closed.")
        if not isinstance(frame, np.ndarray) or frame.shape != RAW_SHAPE:
            raise ValueError(f"CUDA Rust input must have shape {RAW_SHAPE}.")
        if frame.dtype != np.uint8:
            raise ValueError("CUDA Rust input must contain uint8 BGR pixels.")
        np.copyto(self._host_raw, frame, casting="no")

    def enqueue(self, tensor_pointer: Any, stream: Any) -> None:
        # [GPU 순서] 원본 바이트를 H2D 전송한 뒤 정규화 커널을 같은 stream에 등록한다.
        # 호출자는 그 stream에 TensorRT 추론을 이어 붙인다. enqueue 반환은 GPU 작업 완료를 뜻하지 않는다.
        # 다음 stage로 버퍼를 재사용하기 전에는 호출자의 추론/후처리 동기화 수명을 따라야 한다.
        if self._function is None or self._device_raw is None:
            raise RuntimeError("CUDA Rust preprocessor is closed.")
        cudart = self._cudart
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[0], stream),
                "cudaEventRecord(Rust uint8 H2D start)",
            )
        self._runtime_call(
            cudart.cudaMemcpyAsync(
                self._device_raw,
                self._host_raw.ctypes.data,
                RAW_NBYTES,
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                stream,
            ),
            "cudaMemcpyAsync(Rust uint8 input)",
        )
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[1], stream),
                "cudaEventRecord(Rust uint8 H2D end)",
            )
        arguments = (
            # 원본 GPU 주소 → TensorRT 입력 GPU 주소다. 가중치 연산은 이 커널 다음 엔진 실행에서 수행된다.
            self._cuda.CUdeviceptr(int(self._device_raw)),
            self._cuda.CUdeviceptr(int(tensor_pointer)),
        )
        grid_x = (
            self.pixel_count + THREADS_PER_BLOCK - 1
        ) // THREADS_PER_BLOCK
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
                (arguments, (None, None)),
                0,
            ),
            "cuLaunchKernel(rust_bgr_u8_to_imagenet_rgb_nchw)",
        )
        if self._timing_events:
            self._runtime_call(
                cudart.cudaEventRecord(self._timing_events[2], stream),
                "cudaEventRecord(Rust GPU preprocess end)",
            )

    def timing(self) -> CudaRustPreprocessTiming:
        if not self._timing_events:
            raise RuntimeError("CUDA Rust preprocessing timing is disabled.")
        return CudaRustPreprocessTiming(
            input_u8_h2d_ms=float(
                self._runtime_call(
                    self._cudart.cudaEventElapsedTime(
                        self._timing_events[0], self._timing_events[1]
                    ),
                    "cudaEventElapsedTime(Rust uint8 H2D)",
                )
            ),
            gpu_preprocess_ms=float(
                self._runtime_call(
                    self._cudart.cudaEventElapsedTime(
                        self._timing_events[1], self._timing_events[2]
                    ),
                    "cudaEventElapsedTime(Rust GPU preprocess)",
                )
            ),
        )

    def close(self) -> None:
        cudart = self._cudart
        for event in self._timing_events:
            try:
                self._runtime_call(cudart.cudaEventDestroy(event), "cudaEventDestroy")
            except Exception:
                pass
        self._timing_events = ()
        if self._device_raw is not None:
            try:
                self._runtime_call(
                    cudart.cudaFree(self._device_raw), "cudaFree(Rust uint8 input)"
                )
            except Exception:
                pass
            self._device_raw = None
        if self._host_raw_pointer is not None:
            try:
                self._runtime_call(
                    cudart.cudaFreeHost(self._host_raw_pointer),
                    "cudaFreeHost(Rust uint8 input)",
                )
            except Exception:
                pass
            self._host_raw_pointer = None
        if self._module is not None and self._cuda is not None:
            try:
                self._driver_call(
                    self._cuda.cuModuleUnload(self._module), "cuModuleUnload"
                )
            except Exception:
                pass
        self._module = None
        self._function = None
        self._host_raw = None
