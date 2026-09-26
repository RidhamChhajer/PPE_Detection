"""Create source-separated splits from a merged goggles COCO pool."""

import json
from pathlib import Path
import shutil

from validate_training_dataset import validate_document


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/goggles"
GROUP_SPLITS = {
    "hallway-2026-09-15": "train",
    "hallway-2026-09-17": "valid",
    "hallway-2026-08-27": "test",
}


def main():
    annotation_root = DATASET / "annotations"
    image_root = DATASET / "images"
    source_path = annotation_root / "instances.json"
    document = json.loads(source_path.read_text(encoding="utf-8"))
    validate_document(document, image_root, ("goggles",))
    targets = [annotation_root / f"{split}.json" for split in ("train", "valid", "test")]
    targets += [image_root / split for split in ("train", "valid", "test")]
    if any(target.exists() for target in targets):
        raise FileExistsError("Split output already exists; refusing to overwrite it")

    images_by_split = {split: [] for split in ("train", "valid", "test")}
    image_split = {}
    for image in document["images"]:
        group = image.get("source_group")
        if group not in GROUP_SPLITS:
            raise ValueError(f"Unknown source_group {group!r}")
        split = GROUP_SPLITS[group]
        images_by_split[split].append(image)
        image_split[image["id"]] = split

    annotations_by_split = {split: [] for split in images_by_split}
    for annotation in document["annotations"]:
        annotations_by_split[image_split[annotation["image_id"]]].append(annotation)

    report = {"policy": "complete source_group assignments; no group crosses splits", "splits": {}}
    for split in ("train", "valid", "test"):
        destination = image_root / split
        destination.mkdir()
        for image in images_by_split[split]:
            shutil.copy2(image_root / image["file_name"], destination / image["file_name"])
        split_document = {
            "info": dict(document.get("info", {}), split=split,
                         split_policy="source-separated; limited recordings"),
            "licenses": document.get("licenses", []),
            "images": images_by_split[split],
            "annotations": annotations_by_split[split],
            "categories": document["categories"],
        }
        (annotation_root / f"{split}.json").write_text(
            json.dumps(split_document, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        annotated = {annotation["image_id"] for annotation in annotations_by_split[split]}
        report["splits"][split] = {
            "source_groups": sorted({image["source_group"] for image in images_by_split[split]}),
            "images": len(images_by_split[split]),
            "annotations": len(annotations_by_split[split]),
            "negative_images": sum(image["id"] not in annotated for image in images_by_split[split]),
        }
    (DATASET / "split_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
