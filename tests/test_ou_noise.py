import unittest
import numpy as np
from ou_noise import OUActionNoise


class TestOUActionNoise(unittest.TestCase):
    def test_noise_shape(self):
        """
        Test that the OUActionNoise returns a sample of the correct shape.
        """
        noise = OUActionNoise(mu=np.zeros(2))  # Initialize noise process with mean vector of zeros
        sample = noise()                       # Generate a noise sample
        self.assertEqual(sample.shape, (2,))   # Check that the sample shape matches the mean vector


if __name__ == "__main__":
    unittest.main()
