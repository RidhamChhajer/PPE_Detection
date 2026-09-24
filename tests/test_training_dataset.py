"""Tests for task-neutral COCO validation."""

from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from tools.validate_training_dataset import validate_document


class TrainingDatasetValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        cv2.imwrite(str(self.root / "shoe.jpg"), np.zeros((20, 30, 3), dtype=np.uint8))
        self.document = {
            "images": [{"id": 1, "file_name": "shoe.jpg", "width": 30, "height": 20}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 2,
                             "bbox": [1, 2, 10, 5], "area": 50, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "SAFETY_SHOE"},
                           {"id": 2, "name": "NOT_SAFETY_SHOE"}],
        }

    def test_two_class_document(self):
        stats, _ = validate_document(
            self.document, self.root, ("SAFETY_SHOE", "NOT_SAFETY_SHOE"))
        self.assertEqual(stats["annotations"], 1)
        self.assertEqual(stats["class_counts"], {"NOT_SAFETY_SHOE": 1})

    def test_wrong_class_order_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected categories"):
            validate_document(
                self.document, self.root, ("NOT_SAFETY_SHOE", "SAFETY_SHOE"))

    def test_out_of_bounds_box_rejected(self):
        self.document["annotations"][0]["bbox"] = [25, 2, 10, 5]
        with self.assertRaisesRegex(ValueError, "outside image"):
            validate_document(
                self.document, self.root, ("SAFETY_SHOE", "NOT_SAFETY_SHOE"))


if __name__ == "__main__":
    unittest.main()
