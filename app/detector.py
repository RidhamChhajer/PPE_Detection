"""TensorRT-only RTMDet inference. No PyTorch dependency in this module."""
from pathlib import Path
from dataclasses import dataclass
import hashlib
import threading

import cv2
import numpy as np
import tomli

INPUT_SIZE = 640
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)


@dataclass(frozen=True)
class ModelSpec:
    path: Path
    label: str
    status: str


def model_specs(directory):
    """Discover installed engines and verify their status/provenance sidecars."""
    specifications = []
    for path in sorted(Path(directory).glob('*.engine')):
        metadata_path = path.with_suffix('.toml')
        if not metadata_path.is_file():
            raise RuntimeError(f'Missing model provenance for {path.name}; rebuild with export_tensorrt.py')
        metadata = tomli.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('label') != path.stem:
            raise RuntimeError(f'Engine/class metadata mismatch: {path.name}')
        status = metadata.get('status')
        if status not in ('sample', 'requires_field_validation'):
            raise RuntimeError(f'Unrecognized model validation status: {path.name}')
        if metadata.get('engine_sha256') != hashlib.sha256(path.read_bytes()).hexdigest():
            raise RuntimeError(f'Engine checksum mismatch: {path.name}')
        specifications.append(ModelSpec(path.resolve(), path.stem, status))
    if not specifications:
        raise RuntimeError('No verified TensorRT engines installed. Run the export and verification step first.')
    return specifications


def preprocess(frame):
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
        raise ValueError('Expected a uint8 BGR image')
    height, width = frame.shape[:2]
    if min(height, width) == 0:
        raise ValueError('Empty image')
    ratio = INPUT_SIZE / max(height, width)
    resized_w, resized_h = int(width * ratio + 0.5), int(height * ratio + 0.5)
    padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
    padded[:resized_h, :resized_w] = cv2.resize(frame, (resized_w, resized_h))
    tensor = ((padded.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), (resized_w / width, resized_h / height)


def postprocess(raw, scales, shape, threshold=0.30, nms_iou=0.65):
    if not 0 < threshold <= 1 or not 0 < nms_iou <= 1:
        raise ValueError('Thresholds must be within (0, 1]')
    detections = np.asarray(raw, dtype=np.float32).reshape(-1, 5)
    if not np.isfinite(detections).all():
        raise RuntimeError('Detector returned non-finite outputs')
    detections = detections[detections[:, 4] >= threshold].copy()
    if not len(detections):
        return detections
    detections[:, [0, 2]] /= scales[0]
    detections[:, [1, 3]] /= scales[1]
    order = np.argsort(-detections[:, 4], kind='stable')
    kept = []
    while len(order) and len(kept) < 300:
        index = order[0]
        kept.append(index)
        rest = order[1:]
        boxes = detections[rest, :4]
        box = detections[index, :4]
        intersection = np.maximum(0, np.minimum(box[2:], boxes[:, 2:]) - np.maximum(box[:2], boxes[:, :2])).prod(axis=1)
        area = np.maximum(0, box[2:] - box[:2]).prod()
        other_area = np.maximum(0, boxes[:, 2:] - boxes[:, :2]).prod(axis=1)
        iou = intersection / np.maximum(area + other_area - intersection, 1e-8)
        order = rest[iou <= nms_iou]
    result = detections[kept]
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, shape[1])
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, shape[0])
    return result[(result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])]


class TensorRTEngine:
    """One fixed-shape engine with explicitly owned CUDA resources."""
    def __init__(self, path):
        import tensorrt as trt
        from cuda.bindings import driver
        self.driver = driver
        self._lock = threading.Lock()
        self._pointers = []
        self._cuda_context = None
        self._device = None
        self._stream = None
        self.context = self.engine = self.runtime = None
        self.label = Path(path).stem
        try:
            self._check(driver.cuInit(0))
            self._device = self._check(driver.cuDeviceGet(0))
            self._cuda_context = self._check(driver.cuDevicePrimaryCtxRetain(self._device))
            self._check(driver.cuCtxSetCurrent(self._cuda_context))
            self._stream = self._check(driver.cuStreamCreate(0))
            self.logger = trt.Logger(trt.Logger.WARNING)
            self.runtime = trt.Runtime(self.logger)
            self.engine = self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
            if self.engine is None:
                raise RuntimeError('Cannot load TensorRT engine; rebuild it on this GPU')
            self.context = self.engine.create_execution_context()
            if self.context is None:
                raise RuntimeError('Cannot allocate TensorRT execution context')
            if self.engine.num_io_tensors != 2:
                raise ValueError('Expected one input and one output')
            self.names = [self.engine.get_tensor_name(i) for i in range(2)]
            inputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
            outputs = [n for n in self.names if self.engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError('Invalid engine IO contract')
            self.input_name, self.output_name = inputs[0], outputs[0]
            if tuple(self.engine.get_tensor_shape(self.input_name)) != (1, 3, 640, 640):
                raise ValueError('Expected fixed input shape (1, 3, 640, 640)')
            if tuple(self.engine.get_tensor_shape(self.output_name)) != (1, 8400, 5):
                raise ValueError('Expected single-class RTMDet output (1, 8400, 5)')
            if any(self.engine.get_tensor_dtype(n) != trt.float32 for n in self.names):
                raise ValueError('Engine IO must be float32; internal kernels may use FP16')
            self.host_output = np.empty((1, 8400, 5), dtype=np.float32)
            self.input_ptr = self._allocate(1 * 3 * 640 * 640 * 4)
            self.output_ptr = self._allocate(self.host_output.nbytes)
            for name, pointer in ((self.input_name, self.input_ptr), (self.output_name, self.output_ptr)):
                if not self.context.set_tensor_address(name, int(pointer)):
                    raise RuntimeError(f'Cannot bind {name}')
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(result):
        error, *values = result
        if int(error) != 0:
            raise RuntimeError(f'CUDA driver error: {error}')
        return values[0] if len(values) == 1 else values

    def _allocate(self, size):
        pointer = self._check(self.driver.cuMemAlloc(size))
        self._pointers.append(pointer)
        return pointer

    def infer_raw(self, tensor):
        if tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32 or not tensor.flags.c_contiguous:
            raise ValueError('Input must be contiguous float32 NCHW (1, 3, 640, 640)')
        with self._lock:
            if self.context is None:
                raise RuntimeError('Engine is closed')
            self._check(self.driver.cuCtxSetCurrent(self._cuda_context))
            self._check(self.driver.cuMemcpyHtoD(self.input_ptr, tensor.ctypes.data, tensor.nbytes))
            if not self.context.execute_async_v3(stream_handle=int(self._stream)):
                raise RuntimeError('TensorRT execution failed')
            self._check(self.driver.cuStreamSynchronize(self._stream))
            self._check(self.driver.cuMemcpyDtoH(self.host_output.ctypes.data, self.output_ptr, self.host_output.nbytes))
            return self.host_output.copy()

    def detect(self, frame):
        tensor, scales = preprocess(frame)
        return postprocess(self.infer_raw(tensor), scales, frame.shape)

    def close(self):
        with self._lock:
            if self._cuda_context is None:
                return
            self.driver.cuCtxSetCurrent(self._cuda_context)
            self.context = self.engine = self.runtime = None
            for pointer in self._pointers:
                self.driver.cuMemFree(pointer)
            self._pointers.clear()
            if self._stream is not None:
                self.driver.cuStreamDestroy(self._stream)
                self._stream = None
            self.driver.cuDevicePrimaryCtxRelease(self._device)
            self._cuda_context = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
