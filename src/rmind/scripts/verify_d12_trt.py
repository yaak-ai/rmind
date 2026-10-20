"""Build and replay an exported D12 model on its target GPU, without PyTorch."""

import argparse
import ctypes
import ctypes.util
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import tensorrt as trt

trt_api = cast("Any", trt)
WARMUP_ITERATIONS = 3


class Cuda:
    def __init__(self) -> None:
        library = ctypes.util.find_library("cudart")
        if library is None:
            msg = "CUDA runtime library not found"
            raise RuntimeError(msg)
        self.library = ctypes.CDLL(library)
        self.library.cudaMalloc.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
        ]
        self.library.cudaFree.argtypes = [ctypes.c_void_p]
        self.library.cudaMemcpy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
        ]
        self.library.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        self.library.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
        self.library.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p
        self.allocations: list[ctypes.c_void_p] = []
        self.stream = ctypes.c_void_p()
        self.check(self.library.cudaStreamCreate(ctypes.byref(self.stream)))

    def check(self, status: int) -> None:
        if status:
            msg = self.library.cudaGetErrorString(status).decode()
            raise RuntimeError(msg)

    def allocate(self, size: int) -> ctypes.c_void_p:
        pointer = ctypes.c_void_p()
        self.check(self.library.cudaMalloc(ctypes.byref(pointer), size))
        self.allocations.append(pointer)
        return pointer

    def close(self) -> None:
        for pointer in self.allocations:
            self.check(self.library.cudaFree(pointer))
        self.check(self.library.cudaStreamDestroy(self.stream))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--reuse-engine", action="store_true")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    arguments = parser.parse_args()
    if not np.isfinite(arguments.tolerance) or arguments.tolerance <= 0:
        msg = "invalid parity tolerance"
        raise ValueError(msg)
    manifest = json.loads(arguments.manifest.read_text())
    if manifest["schema"] != "d12_flow_matching_v1":
        msg = "unsupported manifest"
        raise ValueError(msg)
    root = arguments.manifest.parent
    logger = trt_api.Logger(trt_api.Logger.INFO)
    build_seconds = None
    if not arguments.reuse_engine:
        builder = trt_api.Builder(logger)
        flags = 1 << int(trt_api.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
        network = builder.create_network(flags)
        onnx_parser = trt_api.OnnxParser(network, logger)
        if not onnx_parser.parse_from_file(str(root / manifest["onnx"])):
            msg = "\n".join(
                str(onnx_parser.get_error(index))
                for index in range(onnx_parser.num_errors)
            )
            raise RuntimeError(msg)
        config = builder.create_builder_config()
        config.clear_flag(trt_api.BuilderFlag.TF32)
        config.set_memory_pool_limit(trt_api.MemoryPoolType.WORKSPACE, 2 << 30)
        started = time.monotonic()
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            msg = "TensorRT build failed"
            raise RuntimeError(msg)
        build_seconds = time.monotonic() - started
        arguments.engine.write_bytes(bytes(serialized))
        del serialized, config, onnx_parser, network, builder
    runtime = trt_api.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(arguments.engine.read_bytes())
    if engine is None:
        msg = "TensorRT deserialization failed"
        raise RuntimeError(msg)
    context = engine.create_execution_context()
    cuda = Cuda()
    try:
        fixtures = manifest["fixture"]["inputs"] + manifest["fixture"]["outputs"]
        if engine.num_io_tensors != len(fixtures):
            msg = "engine tensor count mismatch"
            raise ValueError(msg)
        names = [
            engine.get_tensor_name(index) for index in range(engine.num_io_tensors)
        ]
        if set(names) != {fixture["name"] for fixture in fixtures}:
            msg = "engine tensor names mismatch"
            raise ValueError(msg)
        buffers = {}
        for section, mode in [
            ("inputs", trt_api.TensorIOMode.INPUT),
            ("outputs", trt_api.TensorIOMode.OUTPUT),
        ]:
            for fixture in manifest["fixture"][section]:
                name = fixture["name"]
                if (
                    tuple(engine.get_tensor_shape(name)) != tuple(fixture["shape"])
                    or engine.get_tensor_dtype(name) != trt_api.float32
                    or engine.get_tensor_mode(name) != mode
                ):
                    msg = "engine shape/dtype/mode mismatch: " + name
                    raise ValueError(msg)
                values = np.fromfile(root / fixture["file"], dtype=np.float32)
                if (
                    values.size != np.prod(fixture["shape"])
                    or not np.isfinite(values).all()
                ):
                    msg = "invalid fixture: " + name
                    raise ValueError(msg)
                pointer = cuda.allocate(values.nbytes)
                if not context.set_tensor_address(name, pointer.value):
                    msg = "binding failed: " + name
                    raise RuntimeError(msg)
                if section == "inputs":
                    cuda.check(
                        cuda.library.cudaMemcpy(
                            pointer, values.ctypes.data, values.nbytes, 1
                        )
                    )
                buffers[name] = pointer
        durations = []
        for iteration in range(13):
            started = time.monotonic()
            if not context.execute_async_v3(cuda.stream.value):
                msg = "TensorRT execution failed"
                raise RuntimeError(msg)
            cuda.check(cuda.library.cudaStreamSynchronize(cuda.stream))
            if iteration >= WARMUP_ITERATIONS:
                durations.append((time.monotonic() - started) * 1000)
        errors = {}
        for fixture in manifest["fixture"]["outputs"]:
            reference = np.fromfile(root / fixture["file"], dtype=np.float32)
            actual = np.empty_like(reference)
            cuda.check(
                cuda.library.cudaMemcpy(
                    actual.ctypes.data, buffers[fixture["name"]], actual.nbytes, 2
                )
            )
            if not np.isfinite(actual).all():
                msg = "non-finite output: " + fixture["name"]
                raise ValueError(msg)
            errors[fixture["name"]] = float(np.max(np.abs(actual - reference)))
        passed = max(errors.values()) <= arguments.tolerance
        report = {
            "passed": passed,
            "tensorrt": trt.__version__,
            "host": platform.node(),
            "platform": platform.platform(),
            "engine": str(arguments.engine),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "build_seconds": build_seconds,
            "max_abs": errors,
            "tolerance": arguments.tolerance,
            "engine_only_ms": {
                "median": float(np.median(durations)),
                "max": max(durations),
            },
            "note": "Recorded fixture replay only; not live actuation or deployment approval",
        }
        arguments.report.write_text(json.dumps(report, indent=2) + "\n")
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
        if not passed:
            msg = "TensorRT parity failed"
            raise RuntimeError(msg)
    finally:
        cuda.close()


if __name__ == "__main__":
    main()
