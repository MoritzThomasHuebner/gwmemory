from gwmemory import harmonics as gw_harmonics
from NRSur7dq2 import harmonics as nr_harmonics
import numpy as np
import unittest


class TestHarmonics(unittest.TestCase):

    def test_factorial(self):
        for n in range(10):
            self.assertEqual(nr_harmonics.fac(n), gw_harmonics.fac(n))

    def test_c_slm(self):
        s = -2
        for l in range(0, 5):
            for m in range(-l, l + 1):
                try:
                    self.assertEqual(np.nan_to_num(nr_harmonics.Cslm(s, l, m)),
                                     np.nan_to_num(gw_harmonics.Cslm(s, l, m)))
                except ZeroDivisionError as e:
                    print(e)
                    continue

    def test_s_lambda_lm(self):
        x = 0.1
        s = -2
        for l in range(0, 5):
            for m in range(-l, l + 1):
                try:
                    self.assertEqual(np.nan_to_num(nr_harmonics.s_lambda_lm(s, l, m, x)),
                                     np.nan_to_num(gw_harmonics.s_lambda_lm(s, l, m, x)))
                except ZeroDivisionError as e:
                    print(e)
                    continue

    def test_sYlm(self):
        theta = 0.1
        phi = 0.1
        s = -2
        for l in range(0, 5):
            for m in range(-l, l + 1):
                try:
                    self.assertEqual(np.nan_to_num(nr_harmonics.sYlm(s, l, m, theta, phi)),
                                     np.nan_to_num(gw_harmonics.sYlm(s, l, m, theta, phi)))
                except ZeroDivisionError as e:
                    print(e)
                    continue
