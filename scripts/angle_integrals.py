#!/home/colm.talbot/virtualenvironents/py-2.7/bin/python
from __future__ import division, print_function
import numpy as np
import pandas as pd
import sys
from gwmemory.angles import gamma

"""
Script to calculate the spherical harmonic decomposition of the output memory.

Colm Talbot
"""

delta_m = int(sys.argv[1])
# delta_m = 0
ells = np.arange(2, 5, 1)

coefficients = pd.DataFrame()

coefficients['l'] = np.arange(2, 21, 1)

for ell1 in ells:
    for ell2 in ells:
        # if ell2 > ell1:
            # continue
        for m1 in np.arange(-ell1, ell1+1, 1):
            m2 = m1 - delta_m
            if (m1 < -ell1) or (m1 > ell1) or (m2 < -ell2) or (m2 > ell2):
                continue
            lm1 = str(ell1) + str(m1)
            lm2 = str(ell2) + str(m2)

            print(lm1+lm2)

            coefficients[lm1+lm2] = np.real(gamma(lm1, lm2))

out_file = "data/gamma_coefficients_delta_m_{}.dat".format(delta_m)
print("Saving to {}".format(out_file))
coefficients.to_csv(out_file, sep='\t', index=False)
