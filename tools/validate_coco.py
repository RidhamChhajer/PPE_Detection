"""Import a legacy goggles COCO export and verify MMDetection loading.

Original images and annotations are read-only. Cleaned annotations and previews
are written to the explicit output directory; original images stay unchanged.
"""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLITS = ("train", "valid", "test")
SOURCE_CLASSES = {"goggles on face", "goggles"}
PROJECT_CLASS = "PPE-Googles-and-Boots"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_independent_splits(documents):
    """Check provenance and negative coverage, not model accuracy or deployment safety.

    source_group must identify a whole recording/session, never an individual
    frame. It is supplied by the dataset owner; filenames are not sufficient.
    """
    owners = {}
    summary = {}
    for split in SPLITS:
        document = documents[split]
        annotated = {ann['image_id'] for ann in document['annotations']}
        groups = set()
        negatives = 0
        for image in document['images']:
            group = image.get('source_group', image.get('extra', {}).get('source_group'))
            require(isinstance(group, str) and bool(group.strip()),
                    f'{split} image {image["id"]}: source_group recording/session metadata is required')
            group = group.strip()
            require(group not in owners or owners[group] == split,
                    f'Source group {group!r} overlaps {owners.get(group)} and {split}')
            owners[group] = split
            groups.add(group)
            negatives += image['id'] not in annotated
        require(bool(groups), f'{split}: no source groups')
        require(negatives > 0, f'{split}: include verified negative images with no goggles annotations')
        summary[split] = (len(groups), negatives)
    require(summary['train'][0] >= 2, 'Training requires more than one independent recording/session')
    return summary


def index_records(records, kind):
    require(isinstance(records, list), f"{kind} must be a list")
    indexed = {}
    for item in records:
        require(isinstance(item, dict), f"Invalid {kind} record")
        key = item.get("id")
        require(type(key) is int and key >= 0, f"Invalid {kind} ID: {key!r}")
        require(key not in indexed, f"Duplicate {kind} ID: {key}")
        indexed[key] = item
    return indexed


def validate_split(document, image_root):
    """Fail on invalid data instead of clipping boxes or dropping annotations."""
    clean = deepcopy(document)
    images = index_records(clean.get("images"), "image")
    annotations = index_records(clean.get("annotations"), "annotation")
    categories = index_records(clean.get("categories"), "category")
    require(bool(images), "Split contains no images")
    real = [key for key, category in categories.items() if category.get("name") in SOURCE_CLASSES]
    require(len(real) == 1, "Expected exactly one goggles category")
    used = Counter(ann.get("category_id") for ann in annotations.values())
    for key, category in categories.items():
        if key != real[0]:
            require(category.get("name") == PROJECT_CLASS and used[key] == 0,
                    f"Cannot remove category {key}: unknown or annotated category")

    filenames = set()
    hashes = {}
    source_names = set()
    root = image_root.resolve()
    for key, image in images.items():
        name = image.get("file_name")
        require(isinstance(name, str) and bool(name), f"Image {key}: missing file_name")
        relative = Path(name)
        path = (root / relative).resolve()
        require(not relative.is_absolute() and path.is_relative_to(root), f"Image {key}: unsafe path")
        require(str(path).casefold() not in filenames, f"Image {key}: duplicate file_name")
        filenames.add(str(path).casefold())
        require(path.is_file(), f"Image {key}: missing file {name}")
        frame = cv2.imread(str(path))
        require(frame is not None, f"Image {key}: cannot decode {name}")
        height, width = frame.shape[:2]
        require(type(image.get("width")) is int and type(image.get("height")) is int,
                f"Image {key}: dimensions must be integers")
        require((image["width"], image["height"]) == (width, height),
                f"Image {key}: declared dimensions disagree with decoded image")
        hashes[name] = hashlib.sha256(frame.tobytes()).hexdigest()
        match = re.match(r"(.+_mp4)-\d+", name)
        if match:
            source_names.add(match.group(1))

    per_image = Counter()
    sizes = []
    for key, ann in annotations.items():
        image_id, category_id = ann.get("image_id"), ann.get("category_id")
        require(type(image_id) is int and image_id in images, f"Annotation {key}: unknown image_id")
        require(type(category_id) is int and category_id in categories,
                f"Annotation {key}: unknown category_id")
        require(category_id == real[0], f"Annotation {key}: not a goggles annotation")
        require(not ann.get("ignore", False), f"Annotation {key}: ignore flag would drop annotation")
        require(ann.get("iscrowd", 0) in (0, 1), f"Annotation {key}: invalid iscrowd")
        require(not ann.get("segmentation"), f"Annotation {key}: expected bounding-box-only data")
        box = ann.get("bbox")
        require(isinstance(box, list) and len(box) == 4, f"Annotation {key}: invalid bbox")
        require(all(type(v) in (int, float) and math.isfinite(v) for v in box),
                f"Annotation {key}: non-finite or non-numeric bbox")
        x, y, w, h = box
        image = images[image_id]
        require(w >= 1 and h >= 1, f"Annotation {key}: box smaller than one pixel")
        require(x >= 0 and y >= 0 and x + w <= image["width"] and y + h <= image["height"],
                f"Annotation {key}: bbox outside image boundaries: {box}")
        area = ann.get("area")
        require(type(area) in (int, float) and math.isfinite(area) and
                math.isclose(area, w * h, rel_tol=1e-6, abs_tol=1e-6),
                f"Annotation {key}: area disagrees with bbox")
        ann["category_id"] = 1
        ann.setdefault("iscrowd", 0)
        per_image[image_id] += 1
        sizes.append((w, h))
    require(bool(annotations), "Split contains no goggles annotations")
    clean["categories"] = [{"id": 1, "name": "goggles", "supercategory": "ppe"}]
    stats = {
        "images": len(images), "annotations": len(annotations),
        "empty_images": sum(key not in per_image for key in images),
        "crowd_annotations": sum(ann["iscrowd"] for ann in annotations.values()),
        "small_boxes": sum(w * h < 32**2 for w, h in sizes),
        "min_width": min(w for w, _ in sizes), "min_height": min(h for _, h in sizes),
        "max_per_image": max(per_image.values()),
    }
    return clean, stats, hashes, source_names


def verify_mmdetection(annotation_path, image_root, document, preview_path):
    """Exercise every image through the real loader and visualize packed GT."""
    from mmdet.datasets import CocoDataset
    from mmdet.utils import register_all_modules
    from mmdet.visualization import DetLocalVisualizer

    register_all_modules(init_default_scope=True)
    dataset = CocoDataset(
        ann_file=str(annotation_path), data_prefix={"img": str(image_root)},
        metainfo={"classes": ("goggles",), "palette": [(0, 255, 0)]},
        test_mode=True, serialize_data=False,
        pipeline=[dict(type="LoadImageFromFile"),
                  dict(type="LoadAnnotations", with_bbox=True), dict(type="PackDetInputs")],
    )
    require(len(dataset) == len(document["images"]), "MMDetection changed image count")
    require(dataset.cat2label == {1: 0}, "Unexpected MMDetection class mapping")
    expected = defaultdict(list)
    for ann in document["annotations"]:
        expected[ann["image_id"]].append(ann)
    visualizer = DetLocalVisualizer(name=f"import_{annotation_path.stem}", line_width=2)
    visualizer.dataset_meta = dataset.metainfo
    preview_indices = set(np.linspace(0, len(dataset) - 1, min(3, len(dataset)), dtype=int))
    previews = []
    count = 0
    for index in range(len(dataset)):
        info = dataset.get_data_info(index)
        packed = dataset[index]
        sample = packed["data_samples"]
        records = expected[info["img_id"]]
        require(len(info["instances"]) == len(records), "MMDetection dropped annotations")
        for crowd, instances in ((0, sample.gt_instances), (1, sample.ignored_instances)):
            subset = [ann for ann in records if ann["iscrowd"] == crowd]
            boxes = np.array([[a["bbox"][0], a["bbox"][1],
                               a["bbox"][0] + a["bbox"][2],
                               a["bbox"][1] + a["bbox"][3]] for a in subset], dtype=np.float32).reshape(-1, 4)
            np.testing.assert_allclose(instances.bboxes.numpy(), boxes, rtol=0, atol=1e-4)
            require(bool((instances.labels == 0).all()), "Unexpected packed class label")
            count += len(instances)
        if index in preview_indices:
            # The 3.3 visualizer expects tensors, while LoadAnnotations emits
            # HorizontalBoxes. Match MMDetection's dataset browsing boundary.
            sample.gt_instances.bboxes = sample.gt_instances.bboxes.tensor
            sample.ignored_instances.bboxes = sample.ignored_instances.bboxes.tensor
            rgb = cv2.cvtColor(cv2.imread(info["img_path"]), cv2.COLOR_BGR2RGB)
            visualizer.add_datasample(str(index), rgb, sample, draw_pred=False, show=False)
            annotated = cv2.cvtColor(visualizer.get_image(), cv2.COLOR_RGB2BGR)
            scale = 640 / annotated.shape[0]
            previews.append(cv2.resize(annotated, (round(annotated.shape[1] * scale), 640)))
    require(count == len(document["annotations"]), "Packed annotation count changed")
    require(cv2.imwrite(str(preview_path), np.hstack(previews)), "Failed to save visualization")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    require(not output.is_relative_to(source) and not source.is_relative_to(output),
            "Source and output must be separate, non-overlapping directories")
    output.mkdir(parents=True, exist_ok=True)
    report = output / "validation.txt"
    report.write_text("RUNNING: acceptance verification has not finished.\n", encoding="utf-8")
    lines = [f"Source: {source}", "Category: goggles, COCO ID 1, MMDetection label 0"]
    validated = {}
    content_splits = defaultdict(set)
    source_splits = defaultdict(set)
    try:
        # Validate all splits before writing any cleaned annotations.
        for split in SPLITS:
            document = json.loads((source / split / "_annotations.coco.json").read_text(encoding="utf-8"))
            clean, stats, hashes, sources = validate_split(document, source / split)
            validated[split] = clean
            line = f"{split}: " + ", ".join(f"{key}={value}" for key, value in stats.items())
            lines.append(line)
            print(line, flush=True)
            for digest in hashes.values():
                content_splits[digest].add(split)
            for name in sources:
                source_splits[name].add(split)
        duplicate_count = sum(len(splits) > 1 for splits in content_splits.values())
        lines.append(f"Identical decoded image content shared across splits: {duplicate_count}")
        for name, splits in sorted(source_splits.items()):
            if len(splits) > 1:
                lines.append(f"WARNING: inferred source video shared across {', '.join(sorted(splits))}: {name}")
        lines.append("Independent source-video, worker or session evaluation is recommended. Filenames cannot certify split independence.")
        for split, clean in validated.items():
            annotation_path = output / f"{split}.json"
            annotation_path.write_text(json.dumps(clean, indent=2, allow_nan=False) + "\n", encoding="utf-8")
            count = verify_mmdetection(annotation_path, source / split, clean, output / f"{split}_preview.jpg")
            lines.append(f"{split}: MMDetection loaded all {len(clean['images'])} images / {count} boxes and saved ground-truth visualization: PASS")
        lines.append("PASS: dataset import and MMDetection visualization complete.")
        print("\n".join(lines[len(SPLITS) + 2:]), flush=True)
        return 0
    except Exception as error:
        lines.append(f"FAIL: {type(error).__name__}: {error}")
        print(lines[-1], file=sys.stderr)
        return 1
    finally:
        report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
