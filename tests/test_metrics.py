import unittest

from benchmark import metrics


class MetricTests(unittest.TestCase):
    def test_probability_and_threshold_metrics(self):
        result = metrics([0, 0, 1, 1], [0.1, 0.4, 0.6, 0.8])
        self.assertAlmostEqual(result["average_precision"], 1.0)
        self.assertAlmostEqual(result["roc_auc"], 1.0)
        self.assertAlmostEqual(result["precision_at_0_5"], 1.0)
        self.assertAlmostEqual(result["recall_at_0_5"], 1.0)
        self.assertEqual(result["predicted_positive_at_0_5"], 2)


if __name__ == "__main__":
    unittest.main()
