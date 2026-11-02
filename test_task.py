import unittest

import numpy as np


class TestTask(unittest.TestCase):
    def test_matvec_product(self):
        # 1. matrices/vecteurs A, x et B

        A = np.array([[2.0, 0.0], [0.0, 2.0]])
        x = np.array([1.0, 3.0])
        B = np.array([2.0, 6.0])

        # 2. Calcule A * x
        Ax = np.dot(A, x)  # ou A @ x

        # 3. Vérifiez que Ax est égal à B avec une tolérance numérique
        np.testing.assert_allclose(Ax, B)


if __name__ == "__main__":
    unittest.main()
