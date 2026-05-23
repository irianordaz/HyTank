"""Convenience wrappers for using HyTank as an LNG (methane) tank model.

These classes subclass :class:`LH2Tank` and :class:`LH2TankThermals`
purely to shift the default initial conditions from LH2 values (~20 K)
to LNG values (~111 K) and to set ``propellant='LNG'``. The underlying
physics components (``BoilOff``, ``LH2BoilOffODE``, ``HeatTransferVacuumTank``,
``VacuumTankWeight``) are unchanged: they read the propellant choice
from their OpenMDAO option and pull the right thermophysical
surrogate from the lazy registry in ``hytank.boil_off.get_propellant``.

Defaults below pick a methane saturation point a hair above 1 atm
(110 K liquid, 112 K ullage at ~1.5 bar) which puts the model in a
well-conditioned state on first solve. The user can still override any
of these via the same option names as the LH2 classes.
"""

from hytank.LH2_tank import LH2Tank, LH2TankThermals


class LNGTank(LH2Tank):
    """Liquid methane (LNG) tank model.

    Same architecture and call surface as :class:`LH2Tank`; only the
    default initial conditions and ``propellant`` choice differ.
    """

    def initialize(self):
        super().initialize()
        # At P = 1.5 bar, methane saturation T ~ 116.7 K. Picking
        # T_gas just above and T_liq just below saturation matches
        # the convention HyTank uses for LH2 (ullage warmer than sat,
        # liquid cooler) so the boil-off solver starts stable.
        self.options["ullage_T_init"] = 118.0
        self.options["ullage_P_init"] = 1.5e5
        self.options["liquid_T_init"] = 113.0
        self.options["propellant"] = "LNG"


class LNGTankThermals(LH2TankThermals):
    """Liquid methane thermal/boil-off model.

    Same architecture and call surface as :class:`LH2TankThermals`;
    only the default initial conditions and ``propellant`` choice
    differ.
    """

    def initialize(self):
        super().initialize()
        # At P = 1.5 bar, methane saturation T ~ 116.7 K. Picking
        # T_gas just above and T_liq just below saturation matches
        # the convention HyTank uses for LH2 (ullage warmer than sat,
        # liquid cooler) so the boil-off solver starts stable.
        self.options["ullage_T_init"] = 118.0
        self.options["ullage_P_init"] = 1.5e5
        self.options["liquid_T_init"] = 113.0
        self.options["propellant"] = "LNG"
