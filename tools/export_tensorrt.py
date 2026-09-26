"""Shared RTMDet export/build math; CLI forwards to versioned export_models."""
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.model_run import configure_reference_precision
from app.detector import MEAN, STD, engine_identity


def check_contract(cfg):
    pre = cfg.model.data_preprocessor
    if pre.get('bgr_to_rgb', False) or not np.allclose(pre.mean, MEAN) or not np.allclose(pre.std, STD):
        raise ValueError('Saved preprocessing differs from runtime BGR normalization')
    for dataset in (cfg.val_dataloader.dataset, cfg.test_dataloader.dataset):
        resize = next(p for p in dataset.pipeline if p.type == 'Resize')
        pad = next(p for p in dataset.pipeline if p.type == 'Pad')
        if tuple(resize.scale) != (640,640) or not resize.keep_ratio or tuple(pad.size) != (640,640) or tuple(pad.pad_val.img) != (114,114,114):
            raise ValueError('Saved resize/padding differs from runtime')
    head = cfg.model.bbox_head
    test = cfg.model.test_cfg
    if head.anchor_generator.offset != 0 or tuple(head.anchor_generator.strides) != (8,16,32):
        raise ValueError('Unsupported RTMDet point generator')
    if test.nms.iou_threshold != .65 or test.max_per_img != 300 or test.get('min_bbox_size',0) != 0 or test.nms_pre < 6400 * head.num_classes:
        raise ValueError('Saved NMS contract differs from runtime')


def load_wrapper(checkpoint, cfg):
    import torch
    from mmdet.apis import init_detector
    torch.set_num_threads(4)
    configure_reference_precision()
    model = init_detector(cfg, str(checkpoint), device='cuda:0')
    for name, value in model.state_dict().items():
        if not torch.isfinite(value).all():
            raise RuntimeError(f'Non-finite checkpoint tensor: {name}')

    class ExportModel(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector
            self.classes = detector.bbox_head.num_classes
            for level, stride in enumerate((8,16,32)):
                coords = torch.arange(640//stride, dtype=torch.float32)*stride
                yy, xx = torch.meshgrid(coords, coords, indexing='ij')
                self.register_buffer(f'points_{level}', torch.stack([xx,yy], -1).reshape(1,-1,2))

        def forward(self, images):
            scores, distances = self.detector.bbox_head(self.detector.extract_feat(images))
            levels = []
            for index in range(3):
                distance = distances[index].permute(0,2,3,1).reshape(1,-1,4)
                points = getattr(self, f'points_{index}')
                boxes = torch.cat([points-distance[:,:,:2], points+distance[:,:,2:]], -1).clamp(0,640)
                probability = scores[index].permute(0,2,3,1).reshape(1,-1,self.classes).sigmoid()
                levels.append(torch.cat([boxes, probability], -1))
            return torch.cat(levels, 1)

    return ExportModel(model).eval().cuda(), model


def export(checkpoint, cfg, path):
    import torch
    import onnx
    wrapper, _ = load_wrapper(checkpoint, cfg)
    with torch.inference_mode():
        torch.onnx.export(wrapper, torch.zeros(1,3,640,640,device='cuda'), str(path),
                          input_names=['images'], output_names=['detections'], opset_version=17)
    onnx.checker.check_model(onnx.load(str(path)))
    print(f'ONNX PASS: {path}', flush=True)


def fp16_layer(name, precision):
    """Stable-score mode keeps shared features and classification in FP32."""
    return precision == 'fp16' or name.startswith('/bbox_head/reg_convs.')


def build(onnx_path, engine_path, name, labels, precision):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(onnx_path.read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    network.name = engine_identity(name, labels)
    if not builder.platform_has_fast_fp16:
        raise RuntimeError('FP16 acceleration unavailable')
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    if precision == 'stable_scores':
        config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for index in range(network.num_layers):
            layer = network.get_layer(index)
            dtype = trt.float16 if fp16_layer(layer.name, precision) else trt.float32
            floating = False
            for output in range(layer.num_outputs):
                tensor = layer.get_output(output)
                if tensor.dtype in (trt.float32, trt.float16) and not tensor.is_shape_tensor:
                    layer.set_output_type(output, dtype)
                    floating = True
            if floating and layer.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE):
                layer.precision = dtype
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512*1024**2)
    config.builder_optimization_level = 3
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT build failed')
    engine_path.write_bytes(bytes(engine))
    print('FP16 candidate built; not installed.', flush=True)


def compare_detections(reference, actual, box_limit, score_limit):
    """One-to-one matching requires the same class, location and confidence."""
    from scipy.optimize import linear_sum_assignment
    if len(reference) != len(actual):
        raise RuntimeError(f'Post-NMS count mismatch: reference={len(reference)}, actual={len(actual)}')
    maximum_box = maximum_score = 0.
    for label in set(reference[:,5]) | set(actual[:,5]):
        expected = reference[reference[:,5] == label]
        found = actual[actual[:,5] == label]
        if len(expected) != len(found):
            raise RuntimeError(f'Post-NMS class {label} count mismatch')
        if not len(expected):
            continue
        boxes = np.abs(expected[:,None,:4]-found[None,:,:4]).max(axis=2)
        scores = np.abs(expected[:,None,4]-found[None,:,4])
        valid = (boxes <= box_limit) & (scores <= score_limit)
        row, col = linear_sum_assignment(np.where(valid, boxes/max(box_limit,1e-9)+scores/max(score_limit,1e-9), 1e9))
        if not valid[row,col].all():
            raise RuntimeError(f'Post-NMS class {label} boxes/confidences exceed tolerance: box={boxes.min(axis=1).max():.6f}')
        maximum_box = max(maximum_box, float(boxes[row,col].max()))
        maximum_score = max(maximum_score, float(scores[row,col].max()))
    return maximum_box, maximum_score

if __name__ == '__main__':
    from tools.export_models import main
    main()
