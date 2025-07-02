import unittest
import numpy as np
import os
from utils import plotLearning


class TestUtils(unittest.TestCase):
    def test_plotLearning(self):
        scores = np.random.randn(200).tolist()
        filename = "test_plot.png"
        plotLearning(scores, filename, window=10)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)


if __name__ == "__main__":
    unittest.main()
