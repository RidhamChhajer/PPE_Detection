"""Audit COCO ZIPs and prepare recording-disjoint YOLO datasets without modifying originals."""

import argparse
import hashlib
import io
import itertools
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

CLASSES = {
    "upper": ["goggles", "no_goggles", "empty_wrist", "not_empty_wrist"],
    "lower": ["SAFETY_SHOE", "NOT_SAFETY_SHOE"],
}


def recording(name):
    match = re.search(r"WIN_\d{8}_\d{2}_\d{2}_\d{2}_Pro", name)
    if match:
        return match.group()
    clean = re.sub(r"^i\d+_", "", Path(name).stem)
    return re.split(r"(?:_mp4|\.mp4|-\d{3,}|_jpg|\.rf\.)", clean, maxsplit=1)[0]


def load_archive(path):
    records = []
    with zipfile.ZipFile(path) as archive:
        for member in archive.namelist():
            if not member.endswith("_annotations.coco.json"):
                continue
            split = Path(member).parent.name
            data = json.loads(archive.read(member))
            mapping = {c["id"]: c["name"] for c in data["categories"]}
            annotations = defaultdict(list)
            for ann in data["annotations"]:
                if ann.get("iscrowd", 0):
                    raise ValueError("Crowd annotations need an explicit conversion policy.")
                annotations[ann["image_id"]].append({**ann, "class": mapping[ann["category_id"]]})
            for image in data["images"]:
                filename = image["file_name"]
                full = str(Path(member).parent / filename).replace("\\", "/")
                if full not in archive.namelist():
                    raise ValueError(f"Missing archive image: {full}")
                records.append(
                    {
                        **image,
                        "member": full,
                        "split": split,
                        "group": recording(filename),
                        "annotations": annotations[image["id"]],
                    }
                )
    if not records:
        raise ValueError("No COCO annotation files found")
    return records


def audit(path, output):
    records = load_archive(path)
    report = {
        "images": len(records),
        "splits": {},
        "invalid_boxes": [],
        "duplicates": [],
        "recordings_across_splits": {},
        "warning": "Geometric checks cannot establish label correctness or factory approval.",
    }
    hashes, groups = {}, defaultdict(set)
    with zipfile.ZipFile(path) as archive:
        for rec in records:
            digest = hashlib.sha256(archive.read(rec["member"])).hexdigest()
            if digest in hashes:
                report["duplicates"].append([hashes[digest], rec["member"]])
            else:
                hashes[digest] = rec["member"]
            groups[rec["group"]].add(rec["split"])
        for split in sorted({r["split"] for r in records}):
            rows = [r for r in records if r["split"] == split]
            counts, sizes = Counter(), defaultdict(list)
            for rec in rows:
                for ann in rec["annotations"]:
                    x, y, w, h = ann["bbox"]
                    counts[ann["class"]] += 1
                    sizes[ann["class"]].append(min(w, h) * 640 / max(rec["width"], rec["height"]))
                    if (
                        x < 0
                        or y < 0
                        or w <= 0
                        or h <= 0
                        or x + w > rec["width"] + 1
                        or y + h > rec["height"] + 1
                    ):
                        report["invalid_boxes"].append(
                            {"image": rec["member"], "id": ann["id"], "bbox": ann["bbox"]}
                        )
            report["splits"][split] = {
                "images": len(rows),
                "classes": dict(counts),
                "empty_images": sum(not r["annotations"] for r in rows),
                "median_short_side_at_640": {
                    k: round(statistics.median(v), 1) for k, v in sizes.items()
                },
                "groups": dict(Counter(r["group"] for r in rows)),
            }
        report["recordings_across_splits"] = {g: sorted(s) for g, s in groups.items() if len(s) > 1}
        output.mkdir(parents=True, exist_ok=True)
        samples = []
        for cls in sum(CLASSES.values(), []):
            candidates = [
                (r, a)
                for r in records
                if r["split"] == "train"
                for a in r["annotations"]
                if a["class"] == cls
            ]
            candidates.sort(key=lambda pair: pair[1]["bbox"][2] * pair[1]["bbox"][3])
            for fraction in (0.1, 0.5, 0.9):
                if not candidates:
                    continue
                rec, ann = candidates[min(len(candidates) - 1, int(len(candidates) * fraction))]
                image = Image.open(io.BytesIO(archive.read(rec["member"])))
                image.load()
                x, y, w, h = ann["bbox"]
                margin = max(w, h) * 0.5
                crop = image.crop(
                    (
                        max(0, x - margin),
                        max(0, y - margin),
                        min(image.width, x + w + margin),
                        min(image.height, y + h + margin),
                    )
                ).convert("RGB")
                tile = Image.new("RGB", (300, 230), "#F2F6FA")
                crop.thumbnail((285, 185))
                tile.paste(crop, ((300 - crop.width) // 2, 5))
                ImageDraw.Draw(tile).text(
                    (8, 200), f"{cls} / box {w:.0f}x{h:.0f}px", fill="#14283F"
                )
                samples.append(tile)
        sheet = Image.new("RGB", (900, 230 * ((len(samples) + 2) // 3)), "white")
        for i, tile in enumerate(samples):
            sheet.paste(tile, ((i % 3) * 300, (i // 3) * 230))
        sheet.save(output / "dataset-crops.jpg")
        rec = next(r for r in records if r["annotations"])
        image = Image.open(io.BytesIO(archive.read(rec["member"]))).convert("RGB")
        draw = ImageDraw.Draw(image)
        for ann in rec["annotations"]:
            x, y, w, h = ann["bbox"]
            draw.rectangle((x, y, x + w, y + h), outline="#BC3547", width=3)
            draw.text((x, max(0, y - 14)), ann["class"], fill="#BC3547")
        image.thumbnail((1280, 720))
        image.save(output / "annotated-scene.jpg")
    (output / "dataset-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def choose_splits(records):
    groups = sorted({r["group"] for r in records})
    if len(groups) < 3:
        raise ValueError("Need at least three independent recordings for train/valid/test.")
    if len(groups) > 10:
        raise ValueError(
            "More than 10 recordings: supply a reviewed split-map JSON instead of exhaustive assignment."
        )
    counts = {
        g: Counter(a["class"] for r in records if r["group"] == g for a in r["annotations"])
        for g in groups
    }
    sizes = Counter(r["group"] for r in records)
    total = sum(counts.values(), Counter())
    ratios = {"train": 0.7, "valid": 0.15, "test": 0.15}
    best, score = None, float("inf")
    for assignment in itertools.product(ratios, repeat=len(groups)):
        if set(assignment) != set(ratios):
            continue
        mapping = dict(zip(groups, assignment))
        candidate = 0
        for split, ratio in ratios.items():
            c = sum((counts[g] for g in groups if mapping[g] == split), Counter())
            if any(not c[k] for k in total):
                candidate = float("inf")
                break
            candidate += (
                abs(sum(sizes[g] for g in groups if mapping[g] == split) / len(records) - ratio) * 2
            )
            candidate += sum(abs(c[k] / total[k] - ratio) for k in total) / len(total)
        if candidate < score:
            best, score = mapping, candidate
    if best is None:
        raise ValueError(
            "Cannot create recording-disjoint splits with every class present. Capture more varied recordings or supply a reviewed split map."
        )
    return best


def yolo_line(ann, width, height, names):
    x, y, w, h = ann["bbox"]
    if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width + 1 or y + h > height + 1:
        raise ValueError(f"Invalid bounding box: {ann['bbox']}")
    x2, y2 = min(x + w, width), min(y + h, height)
    return f"{names.index(ann['class'])} {(x + x2) / 2 / width:.8f} {(y + y2) / 2 / height:.8f} {(x2 - x) / width:.8f} {(y2 - y) / height:.8f}"


def prepare(path, destination, split_map=None):
    if destination.exists():
        raise ValueError(
            "Destination exists. Choose a new directory; existing data is never overwritten."
        )
    records = load_archive(path)
    mapping = split_map or choose_splits(records)
    if set(mapping) != {r["group"] for r in records} or set(mapping.values()) != {
        "train",
        "valid",
        "test",
    }:
        raise ValueError("Split map must assign every recording to train, valid or test.")
    hashes = {}
    prepared = []
    with zipfile.ZipFile(path) as archive:
        for index, rec in enumerate(records):
            payload = archive.read(rec["member"])
            digest = hashlib.sha256(payload).hexdigest()
            split = mapping[rec["group"]]
            if digest in hashes:
                if hashes[digest] != split:
                    raise ValueError(
                        "Exact duplicate occurs in different recording splits; review source grouping."
                    )
                continue
            hashes[digest] = split
            filename = f"{index:06d}{Path(rec['file_name']).suffix.lower()}"
            # Generate all labels before writing so invalid geometry cannot be silently clipped.
            labels = {
                view: [
                    yolo_line(a, rec["width"], rec["height"], names)
                    for a in rec["annotations"]
                    if a["class"] in names
                ]
                for view, names in CLASSES.items()
            }
            for view in CLASSES:
                image_dir = destination / view / "images" / split
                label_dir = destination / view / "labels" / split
                image_dir.mkdir(parents=True, exist_ok=True)
                label_dir.mkdir(parents=True, exist_ok=True)
                (image_dir / filename).write_bytes(payload)
                (label_dir / Path(filename).with_suffix(".txt")).write_text(
                    "\n".join(labels[view]), encoding="utf-8"
                )
            prepared.append(
                {
                    "source": rec["member"],
                    "group": rec["group"],
                    "split": split,
                    "sha256": digest,
                    "name": filename,
                }
            )
    for view, names in CLASSES.items():
        data = {
            "path": str((destination / view).resolve()),
            "train": "images/train",
            "val": "images/valid",
            "test": "images/test",
            "names": dict(enumerate(names)),
        }
        (destination / view / "data.yaml").write_text(
            yaml.safe_dump(data, sort_keys=False), encoding="utf-8"
        )
    (destination / "split-manifest.json").write_text(
        json.dumps(
            {
                "recordings": mapping,
                "images": prepared,
                "warning": "Recording-disjoint filenames do not establish independent people or sessions; review source provenance.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "images": len(prepared),
        "split_recordings": mapping,
        "splits": dict(Counter(r["split"] for r in prepared)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["audit", "prepare"])
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-map", type=Path)
    args = parser.parse_args()
    try:
        result = (
            audit(args.archive, args.output)
            if args.action == "audit"
            else prepare(
                args.archive,
                args.output,
                json.loads(args.split_map.read_text()) if args.split_map else None,
            )
        )
        print(json.dumps(result, indent=2))
    except (ValueError, OSError, zipfile.BadZipFile) as exc:
        parser.exit(1, str(exc) + "\n")
