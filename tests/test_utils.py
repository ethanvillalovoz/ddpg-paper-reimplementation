import unittest
import numpy as np
import os
from utils import plotLearning


class TestUtils(unittest.TestCase):
    def test_plotLearning(self):
        """
        Test that plotLearning creates a plot file and removes it after the test.
        """
        scores = np.random.randn(200).tolist()  # Generate random scores for plotting
        filename = "test_plot.png"              # Temporary filename for plot
        plotLearning(scores, filename, window=10)  # Create the plot
        self.assertTrue(os.path.exists(filename))   # Check that the file was created
        os.remove(filename)                        # Clean up by removing the file


if __name__ == "__main__":
    unittest.main()
