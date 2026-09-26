"""Validate a split COCO detection dataset and write ground-truth previews."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def _index(items, kind):
    require(isinstance(items, list), f"{kind} must be a list")
    result = {}
    for item in items:
        require(isinstance(item, dict), f"Invalid {kind} record")
        key = item.get("id")
        require(type(key) is int and key >= 0, f"Invalid {kind} ID: {key!r}")
        require(key not in result, f"Duplicate {kind} ID: {key}")
        result[key] = item
    return result


def validate_document(document, image_root: Path, expected_classes: tuple[str, ...]):
    images = _index(document.get("images"), "image")
    annotations = _index(document.get("annotations"), "annotation")
    categories = _index(document.get("categories"), "category")
    require(images, "Split contains no images")
    names = tuple(category.get("name") for category in categories.values())
    require(names == expected_classes,
            f"Expected categories {expected_classes!r} in annotation order, found {names!r}")

    root = image_root.resolve()
    file_names = set()
    decoded = {}
    groups = Counter()
    for image_id, image in images.items():
        name = image.get("file_name")
        require(isinstance(name, str) and name, f"Image {image_id}: missing file_name")
        relative = Path(name)
        path = (root / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(root),
                f"Image {image_id}: unsafe file_name")
        require(str(path).casefold() not in file_names, f"Duplicate file_name: {name}")
        file_names.add(str(path).casefold())
        require(path.is_file(), f"Image {image_id}: missing file {name}")
        frame = cv2.imread(str(path))
        require(frame is not None, f"Image {image_id}: cannot decode {name}")
        height, width = frame.shape[:2]
        require((image.get("width"), image.get("height")) == (width, height),
                f"Image {image_id}: declared dimensions disagree with decoded image")
        decoded[image_id] = frame
        group = image.get("source_group", image.get("extra", {}).get("source_group"))
        groups[group if isinstance(group, str) and group.strip() else "MISSING"] += 1

    per_image = Counter()
    per_class = Counter()
    for annotation_id, annotation in annotations.items():
        image_id = annotation.get("image_id")
        category_id = annotation.get("category_id")
        require(type(image_id) is int and image_id in images,
                f"Annotation {annotation_id}: unknown image_id")
        require(type(category_id) is int and category_id in categories,
                f"Annotation {annotation_id}: unknown category_id")
        require(not annotation.get("ignore", False), f"Annotation {annotation_id}: ignore is unsupported")
        require(annotation.get("iscrowd", 0) in (0, 1),
                f"Annotation {annotation_id}: invalid iscrowd")
        box = annotation.get("bbox")
        require(isinstance(box, list) and len(box) == 4,
                f"Annotation {annotation_id}: invalid bbox")
        require(all(type(value) in (int, float) and math.isfinite(value) for value in box),
                f"Annotation {annotation_id}: non-finite bbox")
        x, y, width, height = box
        image = images[image_id]
        require(width >= 1 and height >= 1 and x >= 0 and y >= 0 and
                x + width <= image["width"] and y + height <= image["height"],
                f"Annotation {annotation_id}: bbox outside image: {box}")
        area = annotation.get("area")
        require(type(area) in (int, float) and math.isfinite(area) and
                math.isclose(area, width * height, rel_tol=1e-5, abs_tol=1e-5),
                f"Annotation {annotation_id}: area disagrees with bbox")
        per_image[image_id] += 1
        per_class[categories[category_id]["name"]] += 1

    return {
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": sum(image_id not in per_image for image_id in images),
        "class_counts": dict(per_class),
        "source_groups": dict(groups),
    }, decoded


def write_preview(document, decoded, expected_classes, path: Path):
    annotations = defaultdict(list)
    for annotation in document["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    categories = {category["id"]: category["name"] for category in document["categories"]}
    negatives = [image for image in document["images"] if not annotations[image["id"]]]
    positives = [image for image in document["images"] if annotations[image["id"]]]
    chosen = positives[::max(1, len(positives) // 9)][:9] + negatives[:3]
    panels = []
    palette = [(0, 220, 0), (0, 80, 255), (255, 80, 0)]
    for image in chosen:
        frame = decoded[image["id"]].copy()
        for annotation in annotations[image["id"]]:
            x, y, width, height = annotation["bbox"]
            class_name = categories[annotation["category_id"]]
            color = palette[expected_classes.index(class_name) % len(palette)]
            cv2.rectangle(frame, (round(x), round(y)),
                          (round(x + width), round(y + height)), color, 2)
            cv2.putText(frame, class_name, (max(0, round(x)), max(15, round(y) - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        scale = min(360 / frame.shape[1], 240 / frame.shape[0])
        frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)))
        panel = np.full((275, 380, 3), 255, dtype=np.uint8)
        panel[30:30 + frame.shape[0], 10:10 + frame.shape[1]] = frame
        status = "NEGATIVE" if not annotations[image["id"]] else image["file_name"][:45]
        cv2.putText(panel, status, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)
        panels.append(panel)
    require(panels, "Cannot make preview without images")
    rows = []
    for index in range(0, len(panels), 3):
        row = panels[index:index + 3]
        row += [np.full_like(panels[0], 255)] * (3 - len(row))
        rows.append(np.hstack(row))
    path.parent.mkdir(parents=True, exist_ok=True)
    require(cv2.imwrite(str(path), np.vstack(rows)), f"Cannot save preview {path}")


def validate_dataset(annotation_root: Path, image_root: Path,
                     expected_classes: tuple[str, ...], preview_root: Path | None = None):
    documents = {}
    statistics = {}
    for split in SPLITS:
        annotation_path = annotation_root / f"{split}.json"
        require(annotation_path.is_file(), f"Missing annotation file: {annotation_path}")
        document = json.loads(annotation_path.read_text(encoding="utf-8"))
        stats, decoded = validate_document(document, image_root / split, expected_classes)
        documents[split] = document
        statistics[split] = stats
        if preview_root is not None:
            write_preview(document, decoded, expected_classes, preview_root / f"{split}.jpg")
    return documents, statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument("--preview-output", type=Path)
    args = parser.parse_args()
    _, statistics = validate_dataset(args.annotations.resolve(), args.images.resolve(),
                                     tuple(args.classes),
                                     args.preview_output.resolve() if args.preview_output else None)
    print(json.dumps(statistics, indent=2))


if __name__ == "__main__":
    main()
