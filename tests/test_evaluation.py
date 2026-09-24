import unittest
import numpy as np
from tools.evaluate import match_counts


class DetectionMetricsTests(unittest.TestCase):
    def test_duplicate_prediction_is_false_positive(self):
        box = np.array([0, 0, 10, 10])
        self.assertEqual(match_counts({1: [(box, .9), (box, .8)]}, {1: [box]}), (1, 1, 0))

    def test_missed_and_negative_images(self):
        box = np.array([0, 0, 10, 10])
        self.assertEqual(match_counts({2: [(box, .9)]}, {1: [box]}), (0, 1, 1))

    def test_low_confidence_and_disjoint_boxes(self):
        box = np.array([0, 0, 10, 10])
        other = np.array([20, 20, 30, 30])
        self.assertEqual(match_counts({1: [(box, .2), (other, .9)]}, {1: [box]}), (0, 1, 1))


if __name__ == '__main__':
    unittest.main()
