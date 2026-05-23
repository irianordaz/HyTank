"""Sanity tests for the methane (LNG) property surrogates."""

import unittest

import numpy as np

from hytank.CH4_properties import MethaneProperties
from hytank.utilities.constants import MOLEC_WEIGHT_CH4


class TestMethaneProperties(unittest.TestCase):
    def setUp(self):
        self.m = MethaneProperties()

    def test_molec_weight(self):
        self.assertEqual(self.m.MOLEC_WEIGHT, MOLEC_WEIGHT_CH4)

    def test_saturated_pressure_at_112K(self):
        # NIST/REFPROP: methane saturation pressure at 112 K is ~1.04 bar.
        P = float(self.m.lh2_P(112.0))
        self.assertAlmostEqual(P, 1.04e5, delta=3e3)

    def test_saturated_liquid_density_at_112K(self):
        # NIST/REFPROP: methane liquid density at 112 K is ~422 kg/m^3.
        rho = float(self.m.lh2_rho(112.0))
        self.assertAlmostEqual(rho, 422.0, delta=2.0)

    def test_inverse_saturation_round_trip(self):
        # sat_gh2_T at 1.013e5 Pa should return ~111.7 K.
        T = float(self.m.sat_gh2_T(1.013e5))
        self.assertAlmostEqual(T, 111.7, delta=0.5)

    def test_real_gas_density(self):
        # Methane vapor at 1 bar / 120 K is roughly 1.65 kg/m^3 (ideal gas
        # estimate ~1.6 kg/m^3, real gas slightly higher).
        rho = self.m.gh2_rho(np.array([1e5]), np.array([120.0]))
        self.assertAlmostEqual(float(rho[0]), 1.65, delta=0.1)


if __name__ == "__main__":
    unittest.main()
