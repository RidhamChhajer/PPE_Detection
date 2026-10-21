"""Verified TensorRT, OpenVINO and ONNX CPU inference with automatic selection."""
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np

from .detector import preprocess, postprocess, CONTRACT
from .registry import sha256, model_specs, DEFAULT_REGISTRY


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


def openvino_device(core, requested):
    if requested == 'CPU':
        return 'CPU', core.get_property('CPU', 'FULL_DEVICE_NAME')
    for name in core.available_devices:
        if name.split('.')[0] == 'GPU':
            label = core.get_property(name, 'FULL_DEVICE_NAME')
            if 'intel' in label.casefold():
                return name, label
    raise RuntimeError('OpenVINO Intel GPU is unavailable. Check the Intel graphics driver or use --backend onnx.')


class Detector:
    def __init__(self, spec):
        self.spec = spec
        self.session = self.trt = self.compiled = self.request = None
        self.device_name = 'CPU' if spec.backend == 'onnx' else 'NVIDIA GPU'
        if sha256(spec.path) != spec.digest:
            raise ValueError('Model checksum changed since discovery')
        if spec.backend in ('onnx', 'openvino'):
            try:
                import onnxruntime as ort
            except ImportError as error:
                raise RuntimeError('CPU inference requires: python -m pip install -r requirements-cpu.txt') from error
            options = ort.SessionOptions()
            options.intra_op_num_threads = 4
            options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(str(spec.path), sess_options=options, providers=['CPUExecutionProvider'])
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
            if spec.backend == 'openvino':
                # Validate the hash-verified ONNX contract before handing it to another runtime.
                self.session = None
                try:
                    import openvino as ov
                except ImportError as error:
                    raise RuntimeError('OpenVINO requires: python -m pip install -r requirements-auto.txt') from error
                core = ov.Core()
                target, label = openvino_device(core, spec.device)
                self.device_name = f'{target} · {label}'
                self.compiled = core.compile_model(str(spec.path), target, {
                    'PERFORMANCE_HINT': 'LATENCY', 'INFERENCE_PRECISION_HINT': ov.Type.f32,
                })
                self.request = self.compiled.create_infer_request()
        else:
            from .detector import TensorRTEngine
            self.trt = TensorRTEngine(spec)

    def infer_raw(self, tensor):
        if tensor.shape != (1,3,640,640) or tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise ValueError('Expected finite float32 NCHW input')
        if self.session:
            raw = self.session.run(['detections'], {'images':tensor})[0]
        elif self.request is not None:
            self.request.infer({0: tensor})
            raw = self.request.get_output_tensor(0).data.copy()
        else:
            raw = self.trt.infer_raw(tensor)
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
        self.request = self.compiled = None
        if self.trt:
            self.trt.close()
            self.trt = None

    def __enter__(self): return self
    def __exit__(self, *args): self.close()


def select_runtime(directory, registry=None, backend='auto', device='GPU', factory=Detector):
    """Auto tests both models on each candidate; explicit choices never silently fall back."""
    if backend not in ('auto', 'onnx', 'tensorrt', 'openvino'):
        raise ValueError('Unknown inference backend')
    failures = []
    candidates = ('tensorrt', 'openvino', 'onnx') if backend == 'auto' else (backend,)
    for candidate in candidates:
        manifest = registry or (Path(directory)/'registry.local.json' if candidate == 'tensorrt' else DEFAULT_REGISTRY)
        try:
            specs = model_specs(directory, manifest, candidate, device=device)
            selected_device = device if candidate == 'openvino' else 'NVIDIA GPU' if candidate == 'tensorrt' else 'CPU'
            if backend == 'auto':
                for spec in specs:
                    with factory(spec) as detector:
                        detector.infer_raw(np.zeros((1, 3, 640, 640), dtype=np.float32))
                        selected_device = detector.device_name
            return specs, manifest, dict(backend=candidate, device=selected_device, automatic=backend == 'auto',
                                        fallback_reasons=failures)
        except Exception as error:
            if backend != 'auto':
                raise
            failures.append(f'{candidate}: {error}')
    raise RuntimeError('No usable inference backend. ' + '; '.join(failures))
