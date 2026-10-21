"""TensorRT-only RTMDet inference. No PyTorch dependency in this module."""
import hashlib
import json
import threading
from functools import lru_cache

import cv2
import numpy as np

from .registry import CONTRACT

INPUT_SIZE = 640
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)


def engine_identity(label, labels):
    return json.dumps(dict(contract=CONTRACT, model=label, labels=list(labels)), separators=(',', ':'))


def check_engine_metadata(spec, shape, name):
    if tuple(shape) != (1, 8400, 4 + len(spec.labels)):
        raise ValueError('Engine output class count disagrees with metadata labels')
    identity = spec.entry['tensorrt'].get('identity', spec.model_id)
    if name != engine_identity(identity, spec.labels):
        raise ValueError('Engine embedded class identity disagrees with metadata labels/order')


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
    raw = np.asarray(raw, dtype=np.float32)
    if raw.ndim not in (2, 3) or raw.shape[-1] < 5 or (raw.ndim == 3 and raw.shape[0] != 1):
        raise ValueError('Expected RTMDet output [1,N,4+C] or [N,4+C]')
    raw = raw.reshape(-1, raw.shape[-1])
    if not np.isfinite(raw).all():
        raise RuntimeError('Detector returned non-finite outputs')
    rows, labels = np.where(raw[:, 4:] >= threshold)
    detections = np.column_stack((raw[rows, :4], raw[rows, 4 + labels], labels)).astype(np.float32)
    # Match MMDetection min_bbox_size=0 before NMS/the 300-result limit.
    detections = detections[(detections[:,2] > detections[:,0]) & (detections[:,3] > detections[:,1])]
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
        order = rest[(iou <= nms_iou) | (detections[rest, 5] != detections[index, 5])]
    result = detections[kept]
    result[:, [0, 2]] = np.clip(result[:, [0, 2]], 0, shape[1])
    result[:, [1, 3]] = np.clip(result[:, [1, 3]], 0, shape[0])
    return result[(result[:, 2] > result[:, 0]) & (result[:, 3] > result[:, 1])]


@lru_cache(maxsize=1)
def tensorrt_logger():
    # TensorRT keeps one process-wide logger; retain it across Auto probes and jobs.
    import tensorrt as trt
    return trt.Logger(trt.Logger.WARNING)


class TensorRTEngine:
    """One fixed-shape engine with explicitly owned CUDA resources."""
    def __init__(self, spec):
        import tensorrt as trt
        from cuda.bindings import driver
        self.driver = driver
        self._lock = threading.Lock()
        self._pointers = []
        self._cuda_context = None
        self._device = None
        self._stream = None
        self.context = self.engine = self.runtime = None
        self.labels = spec.labels
        try:
            self._check(driver.cuInit(0))
            self._device = self._check(driver.cuDeviceGet(0))
            self._cuda_context = self._check(driver.cuDevicePrimaryCtxRetain(self._device))
            self._check(driver.cuCtxSetCurrent(self._cuda_context))
            self._stream = self._check(driver.cuStreamCreate(0))
            self.logger = tensorrt_logger()
            self.runtime = trt.Runtime(self.logger)
            engine_bytes = spec.path.read_bytes()
            if hashlib.sha256(engine_bytes).hexdigest() != spec.digest:
                raise RuntimeError('Engine checksum changed since discovery')
            self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
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
            output_shape = tuple(self.engine.get_tensor_shape(self.output_name))
            check_engine_metadata(spec, output_shape, self.engine.name)
            if any(self.engine.get_tensor_dtype(n) != trt.float32 for n in self.names):
                raise ValueError('Engine IO must be float32; internal kernels may use FP16')
            self.host_output = np.empty(output_shape, dtype=np.float32)
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
