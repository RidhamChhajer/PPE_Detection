"""Portable ONNX inference and optional local TensorRT acceleration."""
from dataclasses import dataclass
import json
import numpy as np

from .detector import preprocess, postprocess, ModelSpec, CONTRACT
from .registry import sha256


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_label: str


def structured(rows, labels):
    return [Detection(*map(float,row[:5]),int(row[5]),labels[int(row[5])]) for row in rows]


class Detector:
    def __init__(self, spec, provider='cpu'):
        self.spec = spec
        self.session = self.trt = None
        if sha256(spec.path) != spec.digest:
            raise ValueError('Model checksum changed since discovery')
        if spec.backend == 'onnx':
            import onnxruntime as ort
            providers = ['CPUExecutionProvider']
            if provider == 'cuda':
                if 'CUDAExecutionProvider' not in ort.get_available_providers():
                    raise RuntimeError('CUDA provider unavailable; install onnxruntime-gpu instead of onnxruntime')
                providers.insert(0,'CUDAExecutionProvider')
            elif provider != 'cpu':
                raise ValueError('Provider must be cpu or cuda')
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(spec.path), sess_options=options, providers=providers)
            if provider == 'cuda' and self.session.get_providers()[0] != 'CUDAExecutionProvider':
                raise RuntimeError('CUDA provider could not initialize; select cpu explicitly')
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            if len(inputs) != 1 or len(outputs) != 1:
                raise ValueError('Expected one input and one output')
            if (inputs[0].name, inputs[0].type, inputs[0].shape) != ('images','tensor(float)',[1,3,640,640]):
                raise ValueError('Incompatible ONNX input contract')
            if (outputs[0].name, outputs[0].type, outputs[0].shape) != ('detections','tensor(float)',[1,8400,4+len(spec.labels)]):
                raise ValueError('Incompatible ONNX output/class contract')
            metadata = self.session.get_modelmeta().custom_metadata_map
            if metadata.get('contract') != CONTRACT or metadata.get('model_id') != spec.model_id or json.loads(metadata.get('labels','[]')) != list(spec.labels):
                raise ValueError('ONNX embedded class/version metadata disagrees with registry')
        else:
            from .detector import TensorRTEngine
            identity = spec.entry['tensorrt'].get('identity', spec.model_id)
            legacy = ModelSpec(spec.path, identity, '', spec.labels, spec.digest, CONTRACT)
            self.trt = TensorRTEngine(spec.path, legacy)

    def infer_raw(self, tensor):
        if tensor.shape != (1,3,640,640) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError('Expected finite float32 NCHW input')
        raw = self.session.run(['detections'], {'images':tensor})[0] if self.session else self.trt.infer_raw(tensor)
        if raw.shape != (1,8400,4+len(self.spec.labels)) or raw.dtype != np.float32 or not np.isfinite(raw).all():
            raise RuntimeError('Invalid/non-finite detector output')
        return raw

    def detect(self, frame):
        tensor, scales = preprocess(frame)
        rows = postprocess(self.infer_raw(tensor), scales, frame.shape,
                           self.spec.entry['confidence_threshold'], self.spec.entry['nms_iou_threshold'])
        return structured(rows, self.spec.labels)

    def close(self):
        self.session = None
        if self.trt:
            self.trt.close()
            self.trt = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()
