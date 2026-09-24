"""Export, build, verify, then publish the selected RTMDet-S FP16 engine."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUTPUT = ROOT / 'artifacts/phase4'


def load_wrapper(checkpoint):
    import torch
    from mmdet.apis import init_detector
    torch.set_num_threads(4)
    model = init_detector(str(checkpoint.parent / 'goggles_rtmdet_s.py'), str(checkpoint), device='cuda:0')

    class ExportModel(torch.nn.Module):
        def __init__(self, detector):
            super().__init__()
            self.detector = detector
            for level, stride in enumerate((8, 16, 32)):
                coords = torch.arange(640 // stride, dtype=torch.float32) * stride
                yy, xx = torch.meshgrid(coords, coords, indexing='ij')
                self.register_buffer(f'points_{level}', torch.stack([xx, yy], -1).reshape(1, -1, 2))

        def forward(self, images):
            scores, distances = self.detector.bbox_head(self.detector.extract_feat(images))
            levels = []
            for index in range(3):
                distance = distances[index].permute(0, 2, 3, 1).reshape(1, -1, 4)
                points = getattr(self, f'points_{index}')
                boxes = torch.cat([points - distance[:, :, :2], points + distance[:, :, 2:]], -1).clamp(0, 640)
                probability = scores[index].permute(0, 2, 3, 1).reshape(1, -1, 1).sigmoid()
                levels.append(torch.cat([boxes, probability], -1))
            return torch.cat(levels, 1)

    return ExportModel(model).eval().cuda(), model


def export(checkpoint):
    import onnx
    import torch
    wrapper, _ = load_wrapper(checkpoint)
    with torch.inference_mode():
        torch.onnx.export(wrapper, torch.zeros(1, 3, 640, 640, device='cuda'), str(OUTPUT / 'goggles.onnx'),
                          input_names=['images'], output_names=['detections'], opset_version=17,
                          do_constant_folding=True)
    onnx.checker.check_model(onnx.load(str(OUTPUT / 'goggles.onnx')))
    print('ONNX export and validation: PASS', flush=True)


def build():
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse((OUTPUT / 'goggles.onnx').read_bytes()):
        raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    if not builder.platform_has_fast_fp16:
        raise RuntimeError('FP16 acceleration unavailable on this GPU')
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.clear_flag(trt.BuilderFlag.TF32)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 512 * 1024**2)
    config.builder_optimization_level = 3
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        raise RuntimeError('TensorRT build failed')
    (OUTPUT / 'goggles.engine').write_bytes(bytes(engine))
    print('FP16 engine built; parity verification still required.', flush=True)


def verify(checkpoint, sample=False):
    import cv2
    import numpy as np
    import torch
    import tensorrt as trt
    from mmdet.apis import inference_detector
    from app.detector import TensorRTEngine, preprocess, postprocess
    from mmengine.config import Config
    from tools.validate_coco import check_independent_splits
    wrapper, model = load_wrapper(checkpoint)
    config = Config.fromfile(str(checkpoint.parent / 'goggles_rtmdet_s.py'))
    datasets = [config.train_dataloader.dataset, config.val_dataloader.dataset, config.test_dataloader.dataset]
    documents = {split: json.loads(Path(dataset.ann_file).read_text())
                 for split, dataset in zip(('train', 'valid', 'test'), datasets)}
    if not sample:
        check_independent_splits(documents)
        if config.get('data_regime', 'sample') == 'sample':
            raise RuntimeError('Sample checkpoints require explicit --sample and retain demo labeling')
    document = documents['test']
    image_root = Path(config.test_dataloader.dataset.data_prefix.img)
    maximum_box = maximum_score = final_box_difference = 0.0
    native_difference = 0.0
    with TensorRTEngine(OUTPUT / 'goggles.engine') as engine:
        for image in document['images']:
            frame = cv2.imread(str(image_root / image['file_name']))
            tensor, scales = preprocess(frame)
            with torch.inference_mode():
                reference = wrapper(torch.from_numpy(tensor).cuda()).cpu().numpy()
                native = inference_detector(model, frame).pred_instances.cpu()
            actual = engine.infer_raw(tensor)
            if not np.isfinite(actual).all():
                raise RuntimeError('Non-finite TensorRT output')
            selected = reference[0, :, 4] >= 0.10
            if selected.any():
                maximum_box = max(maximum_box, float(np.abs(reference[0, selected, :4] - actual[0, selected, :4]).max()))
                maximum_score = max(maximum_score, float(np.abs(reference[0, selected, 4] - actual[0, selected, 4]).max()))
            processed = postprocess(reference, scales, frame.shape)
            native = native[native.scores >= 0.30]
            native_boxes = native.bboxes.numpy()
            native_boxes[:, [0, 2]] = native_boxes[:, [0, 2]].clip(0, frame.shape[1])
            native_boxes[:, [1, 3]] = native_boxes[:, [1, 3]].clip(0, frame.shape[0])
            if len(processed) != len(native_boxes):
                raise RuntimeError('Export adapter and MMDetection detection counts differ')
            if len(processed):
                native_difference = max(native_difference, float(np.abs(processed[:, :4] - native_boxes).max()))
                np.testing.assert_allclose(processed[:, 4], native.scores.numpy(), atol=1e-4, rtol=1e-4)
            trt_detections = postprocess(actual, scales, frame.shape)
            if len(trt_detections) != len(processed):
                raise RuntimeError('FP16 changed detection counts at confidence 0.30')
            remaining = list(range(len(trt_detections)))
            for detection in processed:
                differences = np.abs(trt_detections[remaining, :4] - detection[:4]).max(axis=1)
                nearest = int(np.argmin(differences))
                matched = remaining.pop(nearest)
                difference = float(differences[nearest])
                final_box_difference = max(final_box_difference, difference)
                if difference > 3.0 or abs(float(trt_detections[matched, 4] - detection[4])) > 0.03:
                    raise RuntimeError('Final post-NMS detections exceed parity tolerances')
        if maximum_box > 2.0 or maximum_score > 0.03 or native_difference > 0.05:
            raise RuntimeError(f'Parity tolerance exceeded: boxes={maximum_box}, scores={maximum_score}, native={native_difference}')
        for _ in range(10):
            engine.infer_raw(tensor)
        timings = []
        for _ in range(50):
            started = time.perf_counter()
            engine.infer_raw(tensor)
            timings.append((time.perf_counter() - started) * 1000)
    destination = ROOT / 'models/goggles.engine'
    destination.parent.mkdir(exist_ok=True)
    temporary = destination.with_suffix('.engine.tmp')
    shutil.copyfile(OUTPUT / 'goggles.engine', temporary)
    temporary.replace(destination)
    digest = hashlib.sha256(destination.read_bytes()).hexdigest()
    # TOML is model provenance/configuration, never a detection output.
    metadata = destination.with_suffix('.toml')
    metadata.write_text(f'label = "goggles"\nstatus = "{"sample" if sample else "requires_field_validation"}"\n'
                        f'engine_sha256 = "{digest}"\ncheckpoint = "{checkpoint.name}"\n', encoding='utf-8')
    report = [f'TensorRT: {trt.__version__}; internal FP16, float32 IO; GPU: {torch.cuda.get_device_name(0)}',
              f'Checkpoint: {checkpoint.name}', f'Checkpoint SHA256: {hashlib.sha256(checkpoint.read_bytes()).hexdigest()}',
              f'Engine SHA256: {hashlib.sha256(destination.read_bytes()).hexdigest()}',
              f'Test images: {len(document["images"])}; detection counts matched at confidence 0.30.',
              f'Maximum raw box difference: {maximum_box:.6f} input pixels (limit 2.0, reference confidence >=0.10).',
              f'Maximum raw confidence difference: {maximum_score:.6f} (limit 0.03).',
              f'Export adapter versus native MMDetection boxes: {native_difference:.6f} original pixels (limit 0.05).',
              f'Maximum matched post-NMS box difference: {final_box_difference:.6f} original pixels (limit 3.0).',
              f'TensorRT latency, 50 warmed calls including host/device copies: median={np.median(timings):.2f} ms; p95={np.percentile(timings,95):.2f} ms.',
              f'PASS: FP16 parity verified; engine published with {"SAMPLE / DEMO ONLY" if sample else "FIELD VALIDATION REQUIRED"} status.',
              'Rebuild and reverify on the RTX 3050 target; do not copy this GPU-specific engine.']
    (OUTPUT / 'conversion.txt').write_text('\n'.join(report) + '\n', encoding='utf-8')
    print('\n'.join(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['all', 'export', 'build', 'verify'], default='all')
    parser.add_argument('--run-dir', type=Path, default=ROOT / 'work_dirs/goggles')
    parser.add_argument('--sample', action='store_true', help='Keep the resulting engine explicitly marked demo-only')
    args = parser.parse_args()
    os.chdir(ROOT)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    from tools.evaluate import best_checkpoint
    checkpoint = best_checkpoint(args.run_dir.resolve())
    evaluation = checkpoint.parent / 'evaluation/evaluation.txt'
    if not evaluation.is_file() or checkpoint.name not in evaluation.read_text():
        raise RuntimeError('Evaluate the selected checkpoint before conversion')
    if args.stage == 'all':
        # Separate processes release PyTorch GPU allocations before engine building.
        for stage in ('export', 'build', 'verify'):
            command = [sys.executable, __file__, '--stage', stage, '--run-dir', str(args.run_dir.resolve())]
            if args.sample:
                command.append('--sample')
            subprocess.run(command, check=True)
    elif args.stage == 'export':
        export(checkpoint)
    elif args.stage == 'build':
        build()
    else:
        verify(checkpoint, sample=args.sample)


if __name__ == '__main__':
    main()
