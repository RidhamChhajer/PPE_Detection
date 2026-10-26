"""CUDA-only YOLO26 training and held-out evaluation; runs locally or on Kaggle."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "config" / "ultralytics"))
os.environ.setdefault("YOLO_AUTOINSTALL", "false")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["train", "evaluate"])
    parser.add_argument("view", choices=["upper", "lower"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--model", choices=["yolo26m.pt"], default="yolo26m.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "TEMPORARY" / "training")
    args = parser.parse_args()
    import torch
    import yaml
    from ultralytics import YOLO, settings

    pretrained = ROOT / "TEMPORARY" / "pretrained"
    pretrained.mkdir(parents=True, exist_ok=True)
    settings.update({"sync": False, "weights_dir": str(pretrained)})
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable. Run Setup GPU.cmd or enable a GPU in Kaggle.")
    if not args.data.is_file():
        parser.error("Prepared data.yaml does not exist")
    spec = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    # Resolve data relative to its YAML so copying data folders to Kaggle stays portable.
    spec["path"] = str(args.data.parent.resolve())
    args.output.mkdir(parents=True, exist_ok=True)
    resolved_data = args.output / f"{args.view}-resolved-data.yaml"
    resolved_data.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    if args.action == "train":
        model = YOLO(str(args.weights.resolve()) if args.weights else str(pretrained / args.model))
        result = model.train(
            data=str(resolved_data.resolve()),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=0,
            workers=0,
            project=str(args.output.resolve()),
            name=args.view,
            seed=42,
            deterministic=True,
            patience=25,
            plots=True,
            cache=False,
            close_mosaic=10,
        )
        print(
            "Training finished. Review independent validation and test before installing weights."
        )
        print("Run directory:", model.trainer.save_dir)
        print("Metrics:", result.results_dict)
    else:
        if not args.weights or not args.weights.is_file():
            parser.error("Evaluate requires a trusted local --weights file")
        model = YOLO(str(args.weights.resolve()))
        result = model.val(
            data=str(resolved_data.resolve()),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            device=0,
            workers=0,
            plots=True,
            project=str(args.output.resolve()),
            name=args.view + "-heldout",
        )
        rows = []
        for i, cls in enumerate(result.box.ap_class_index):
            rows.append(
                {
                    "class": model.names[int(cls)],
                    "precision": float(result.box.p[i]),
                    "recall": float(result.box.r[i]),
                    "AP50": float(result.box.ap50[i]),
                    "AP50_95": float(result.box.ap[i]),
                }
            )
        target = Path(result.save_dir) / "class-metrics.json"
        target.write_text(
            json.dumps(
                {
                    "classes": rows,
                    "aggregate": result.results_dict,
                    "note": "Metrics use validation matching; calibrate deployment threshold separately. No worker-level accuracy claim.",
                },
                indent=2,
            )
        )
        print(target)


if __name__ == "__main__":
    main()
