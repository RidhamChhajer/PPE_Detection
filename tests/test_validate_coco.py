"""Regression checks for dataset corruption and non-destructive normalization."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.validate_coco import validate_split, check_independent_splits


class ValidateCocoTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cv2.imwrite(str(self.root / "frame.jpg"), np.zeros((20, 30, 3), dtype=np.uint8))
        self.document = {
            "images": [{"id": 0, "file_name": "frame.jpg", "width": 30, "height": 20}],
            "categories": [{"id": 0, "name": "PPE-Googles-and-Boots"},
                           {"id": 7, "name": "goggles on face"}],
            "annotations": [{"id": 1, "image_id": 0, "category_id": 7,
                             "bbox": [20, 15, 10, 5], "area": 50, "iscrowd": 0}],
        }

    def test_normalization_preserves_original_and_edge_box(self):
        before = json.dumps(self.document)
        clean, stats, _, _ = validate_split(self.document, self.root)
        self.assertEqual(before, json.dumps(self.document))
        self.assertEqual(clean["categories"], [{"id": 1, "name": "goggles", "supercategory": "ppe"}])
        self.assertEqual(clean["annotations"][0]["category_id"], 1)
        self.assertEqual(clean["annotations"][0]["bbox"], [20, 15, 10, 5])
        self.assertEqual(stats["annotations"], 1)

    def test_invalid_annotations_rejected(self):
        for field, value in [("image_id", 99), ("category_id", 99), ("category_id", 0),
                             ("bbox", [-1, 0, 10, 5]), ("bbox", [29, 0, 10, 5]),
                             ("bbox", [0, 19, 10, 5]), ("bbox", [0, 0, 0, 5]),
                             ("bbox", [0, 0, float("nan"), 5]), ("area", 0),
                             ("ignore", True), ("iscrowd", 2)]:
            with self.subTest(field=field, value=value):
                document = deepcopy(self.document)
                document["annotations"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_split(document, self.root)

    def test_missing_unsafe_or_misreported_image_rejected(self):
        for field, value in [("file_name", "missing.jpg"), ("file_name", "../frame.jpg"),
                             ("width", 29), ("height", 21)]:
            with self.subTest(field=field, value=value):
                document = deepcopy(self.document)
                document["images"][0][field] = value
                with self.assertRaises(ValueError):
                    validate_split(document, self.root)

    def test_duplicate_ids_rejected(self):
        for kind in ("images", "annotations", "categories"):
            with self.subTest(kind=kind):
                document = deepcopy(self.document)
                document[kind].append(deepcopy(document[kind][0]))
                with self.assertRaisesRegex(ValueError, "Duplicate"):
                    validate_split(document, self.root)

    def test_unknown_category_not_silently_removed(self):
        self.document["categories"].append({"id": 9, "name": "boots"})
        with self.assertRaisesRegex(ValueError, "Cannot remove"):
            validate_split(self.document, self.root)

    def test_corrupt_image_rejected(self):
        (self.root / "frame.jpg").write_bytes(b"not an image")
        with self.assertRaisesRegex(ValueError, "cannot decode"):
            validate_split(self.document, self.root)


class IndependentSplitTests(unittest.TestCase):
    def documents(self):
        return {split: {'images': [{'id': 0, 'source_group': split+'-a'},
                                   {'id': 1, 'source_group': split+'-b'}],
                        'annotations': [{'image_id': 0}]}
                for split in ('train', 'valid', 'test')}

    def test_independent_groups_and_negatives(self):
        self.assertEqual(check_independent_splits(self.documents())['train'], (2, 1))

    def test_shared_source_rejected(self):
        documents = self.documents()
        documents['test']['images'][0]['source_group'] = 'train-a'
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            check_independent_splits(documents)

    def test_missing_provenance_rejected(self):
        documents = self.documents()
        del documents['train']['images'][0]['source_group']
        with self.assertRaisesRegex(ValueError, 'metadata is required'):
            check_independent_splits(documents)

    def test_absent_negatives_rejected(self):
        documents = self.documents()
        documents['valid']['annotations'].append({'image_id': 1})
        with self.assertRaisesRegex(ValueError, 'negative images'):
            check_independent_splits(documents)


if __name__ == "__main__":
    unittest.main()
