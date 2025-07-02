import unittest
import numpy as np
from ou_noise import OUActionNoise


class TestOUActionNoise(unittest.TestCase):
    def test_noise_shape(self):
        noise = OUActionNoise(mu=np.zeros(2))
        sample = noise()
        self.assertEqual(sample.shape, (2,))


if __name__ == "__main__":
    unittest.main()
