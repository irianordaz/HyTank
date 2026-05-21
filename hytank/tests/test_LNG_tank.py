"""Build-and-run sanity test for the LNG tank wrapper."""

import unittest

import numpy as np
import openmdao.api as om

from hytank import LNGTank


class TestLNGTank(unittest.TestCase):
    def test_build_and_run(self):
        nn = 11  # HyTank's integrator requires odd num_nodes (bdf3 stencil)

        p = om.Problem(reports=False)
        p.model.add_subsystem(
            "tank",
            LNGTank(num_nodes=nn, fill_level_init=0.95),
            promotes=["*"],
        )
        p.model.nonlinear_solver = om.NewtonSolver(
            solve_subsystems=True, maxiter=10, iprint=0
        )
        p.model.linear_solver = om.DirectSolver()
        p.setup()

        # Modestly-sized LNG tank, integrated for one hour with a steady
        # extraction rate. Geometry chosen so the tank doesn't approach
        # empty during the integration (LiquidHeight is ill-conditioned
        # near 0%/100% fill).
        p.set_val("thermals.boil_off.integ.duration", 3600.0, units="s")
        p.set_val("radius", 1.0, units="m")
        p.set_val("length", 4.0, units="m")
        p.set_val("N_layers", 30)
        p.set_val("vacuum_gap", 5.0, units="cm")
        p.set_val("environment_design_pressure", 1.0, units="bar")
        p.set_val("max_expected_operating_pressure", 3.0, units="bar")
        p.set_val("m_dot_liq_out", np.full(nn, 0.05), units="kg/s")
        p.set_val("m_dot_gas_out", np.zeros(nn), units="kg/s")
        p.set_val("P_heater", np.zeros(nn), units="W")
        p.set_val("T_env", np.full(nn, 300.0), units="K")

        p.run_model()

        m_liq = p.get_val("m_liq", units="kg")
        fill = p.get_val("fill_level")

        # Tank should have liquid in it
        self.assertTrue(np.all(m_liq > 0), "Liquid mass went non-positive")
        # All outputs finite
        self.assertTrue(np.all(np.isfinite(m_liq)))
        self.assertTrue(np.all(np.isfinite(fill)))
        # Fill level should be monotonically decreasing under steady extraction
        self.assertLess(fill[-1], fill[0], "Fill level did not decrease")
        # And it stays in (0, 1)
        self.assertTrue(np.all((fill > 0) & (fill < 1)))


if __name__ == "__main__":
    unittest.main()
