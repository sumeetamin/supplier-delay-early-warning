import unittest

from benchmark import blend_probabilities, capacity_metrics, metrics


class MetricTests(unittest.TestCase):
    def test_probability_and_threshold_metrics(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.8])
        self.assertAlmostEqual(result["average_precision"], 1.0)
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["precision_at_0_5"], 1.0)
        self.assertAlmostEqual(result["recall_at_0_5"], 1.0)
        self.assertEqual(result["predicted_positive_at_0_5"], 2)

    def test_blend_combines_only_aligned_probabilities(self):
        [self.assertAlmostEqual(value, 0.5) for value in blend_probabilities([0.8, 0.2], [0.4, 0.6], 0.25)]
        with self.assertRaises(ValueError):
            blend_probabilities([0.5], [0.5, 0.5], 0.5)
        with self.assertRaises(ValueError):
            blend_probabilities([0.5], [0.5], 1.5)

    def test_capacity_metrics_use_score_rank_and_ceiling(self):
        result = capacity_metrics([0, 0, 1, 0, 1, 1], [0.1, 0.7, 0.8, 0.2, 0.6, 0.3], shares=(0.33, 0.67))
        self.assertEqual(result["33%"]["rows_reviewed"], 2)
        self.assertAlmostEqual(result["33%"]["precision"], 0.5)
        self.assertAlmostEqual(result["33%"]["recall"], 1 / 3)
        self.assertEqual(result["67%"]["rows_reviewed"], 5)
        self.assertEqual(result["67%"]["true_positives"], 3)

    def test_rejects_bad_capacity(self):
        with self.assertRaises(ValueError):
            capacity_metrics([0, 1], [0.1, 0.9], shares=(0,))


if __name__ == "__main__":
    unittest.main()
