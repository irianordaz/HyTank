"""LNG (liquefied natural gas / methane) cryogenic tank model.

Provides methane thermophysical properties via CoolProp and an
LNG-specific boil-off ODE that mirrors the LH2 implementation but
uses :class:`LNGProperties` for all equation-of-state lookups.
"""

# ==============================================================================
# Standard Python modules
# ==============================================================================
import functools

# ==============================================================================
# External Python modules
# ==============================================================================
import numpy as np
import openmdao.api as om

# ==============================================================================
# Extension modules
# ==============================================================================
from hytank.heat_leak import HeatTransferVacuumTank
from hytank.utilities import Integrator

# ---------------------------------------------------------------------------
# Methane thermophysical properties (CoolProp backend)
# ---------------------------------------------------------------------------


def _ensure_1d_args(method):
    """Promote a property method's numeric arguments to >=1-D arrays.

    CoolProp's ``PropsSI`` raises "max() iterable argument is empty"
    when handed a 0-D numpy array. A 0-D array arises naturally here:
    passing a single-element array into ``PropsSI`` makes it return a
    0-D array, which is then fed straight into the next lookup (e.g.
    ``sat_vap_rho(sat_T(P))``). This bites the boil-off IVP guess
    solver, which evaluates the ODE one node at a time. Promoting every
    argument to at least 1-D keeps CoolProp on its array code path.
    """

    @functools.wraps(method)
    def wrapper(self, *args):
        coerced = [np.atleast_1d(np.asarray(a, dtype=float)) for a in args]
        return method(self, *coerced)

    return wrapper


def _saturated_T_arg(method):
    """Promote T to >=1-D and clamp it to the valid two-phase band.

    CoolProp's two-phase (Q=0/1) lookups RAISE a ``ValueError`` at or
    above the critical point and are undefined below the triple point.
    The boil-off solver can transiently evaluate states outside this
    band (e.g. a Newton/IVP step that overshoots toward the critical
    temperature). A raised exception aborts the whole ``compute`` and
    defeats the NaN/physical guards downstream, so clamp the saturation
    temperature to the dome and let those guards take over instead.
    """

    @functools.wraps(method)
    def wrapper(self, T):
        T = np.atleast_1d(np.asarray(T, dtype=float))
        lo, hi = self._saturation_T_bounds()
        return method(self, np.clip(T, lo, hi))

    return wrapper


class LNGProperties:
    """Methane thermophysical property wrapper around CoolProp.

    Provides the same interface pattern as :class:`HydrogenProperties`
    so that :class:`LNGBoilOffODE` can swap in without modification.

    All methods accept numpy arrays and return numpy arrays of the
    same shape.  Scalars are accepted and returned as scalars (or
    0-D arrays depending on CoolProp internals).
    """

    def __init__(self):
        try:
            from CoolProp.CoolProp import PropsSI  # noqa: F401
        except ImportError:  # pragma: no cover
            raise ImportError(
                'CoolProp is required for LNG tank properties. '
                'Install it with `pixi add coolprop`.'
            )
        self._fluid = 'Methane'
        self._sat_T_bounds = None

    def _saturation_T_bounds(self):
        """Return the (lower, upper) valid two-phase temperature band (K).

        Bounded just inside the triple and critical points so that a
        Q-flash always lands strictly within the saturation dome.
        """
        if self._sat_T_bounds is None:
            from CoolProp.CoolProp import PropsSI

            t_triple = PropsSI('Ttriple', self._fluid)
            t_crit = PropsSI('Tcrit', self._fluid)
            self._sat_T_bounds = (t_triple + 1e-3, t_crit - 1e-3)
        return self._sat_T_bounds

    # -- saturated liquid properties (input = T_sat in K) ------------------

    @_saturated_T_arg
    def sat_liq_P(self, T):
        """Saturated liquid pressure (Pa)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('P', 'T', T, 'Q', 0, self._fluid)

    @_saturated_T_arg
    def sat_liq_rho(self, T):
        """Saturated liquid density (kg/m³)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('D', 'T', T, 'Q', 0, self._fluid)

    @_saturated_T_arg
    def sat_liq_h(self, T):
        """Saturated liquid specific enthalpy (J/kg)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('H', 'T', T, 'Q', 0, self._fluid)

    @_saturated_T_arg
    def sat_liq_u(self, T):
        """Saturated liquid specific internal energy (J/kg)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('U', 'T', T, 'Q', 0, self._fluid)

    @_saturated_T_arg
    def sat_liq_cp(self, T):
        """Saturated liquid specific heat at constant pressure (J/(kg·K))."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('C', 'T', T, 'Q', 0, self._fluid)

    @_saturated_T_arg
    def sat_liq_k(self, T):
        """Saturated liquid thermal conductivity (W/(m·K))."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('L', 'T', T, 'Q', 0, self._fluid)

    @_saturated_T_arg
    def sat_liq_viscosity(self, T):
        """Saturated liquid dynamic viscosity (Pa·s)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('V', 'T', T, 'Q', 0, self._fluid)

    @_saturated_T_arg
    def sat_liq_beta(self, T):
        """Saturated liquid thermal expansion coefficient (1/K)."""
        from CoolProp.CoolProp import PropsSI

        dT = 1e-6
        rho0 = PropsSI('D', 'T', T, 'Q', 0, self._fluid)
        rho1 = PropsSI('D', 'T', T + dT, 'Q', 0, self._fluid)
        return -(rho1 - rho0) / rho0 / dT

    # -- saturated vapour properties (input = T_sat in K) ------------------

    @_saturated_T_arg
    def sat_vap_P(self, T):
        """Saturated vapour pressure (Pa)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('P', 'T', T, 'Q', 1, self._fluid)

    @_saturated_T_arg
    def sat_vap_rho(self, T):
        """Saturated vapour density (kg/m³)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('D', 'T', T, 'Q', 1, self._fluid)

    @_saturated_T_arg
    def sat_vap_h(self, T):
        """Saturated vapour specific enthalpy (J/kg)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('H', 'T', T, 'Q', 1, self._fluid)

    @_saturated_T_arg
    def sat_vap_cp(self, T):
        """Saturated vapour specific heat at constant pressure (J/(kg·K))."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('C', 'T', T, 'Q', 1, self._fluid)

    @_saturated_T_arg
    def sat_vap_k(self, T):
        """Saturated vapour thermal conductivity (W/(m·K))."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('L', 'T', T, 'Q', 1, self._fluid)

    @_saturated_T_arg
    def sat_vap_viscosity(self, T):
        """Saturated vapour dynamic viscosity (Pa·s)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('V', 'T', T, 'Q', 1, self._fluid)

    @_saturated_T_arg
    def sat_vap_beta(self, T):
        """Saturated vapour thermal expansion coefficient (1/K)."""
        from CoolProp.CoolProp import PropsSI

        dT = 1e-6
        rho0 = PropsSI('D', 'T', T, 'Q', 1, self._fluid)
        rho1 = PropsSI('D', 'T', T + dT, 'Q', 1, self._fluid)
        return -(rho1 - rho0) / rho0 / dT

    # -- real-gas properties (inputs = P in Pa, T in K) -------------------

    @_ensure_1d_args
    def gh2_rho(self, P, T):
        """Real-gas density (kg/m³)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('D', 'P', P, 'T', T, self._fluid)

    @_ensure_1d_args
    def gh2_P(self, rho, T):
        """Real-gas pressure (Pa)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('P', 'D', rho, 'T', T, self._fluid)

    @_ensure_1d_args
    def gh2_cv(self, P, T):
        """Specific heat at constant volume (J/(kg·K))."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('Cvmass', 'P', P, 'T', T, self._fluid)

    @_ensure_1d_args
    def gh2_cp(self, P, T):
        """Specific heat at constant pressure (J/(kg·K))."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('Cpmass', 'P', P, 'T', T, self._fluid)

    @_ensure_1d_args
    def gh2_u(self, P, T):
        """Specific internal energy (J/kg)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('U', 'P', P, 'T', T, self._fluid)

    @_ensure_1d_args
    def gh2_h(self, P, T):
        """Specific enthalpy (J/kg)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('H', 'P', P, 'T', T, self._fluid)

    # -- misc -------------------------------------------------------------

    @_ensure_1d_args
    def sat_T(self, P):
        """Saturation temperature at pressure P (K)."""
        from CoolProp.CoolProp import PropsSI

        return PropsSI('T', 'P', P, 'Q', 0, self._fluid)


# ---------------------------------------------------------------------------
# LNG-specific boil-off ODE
# ---------------------------------------------------------------------------


class LNGBoilOffODE(om.ExplicitComponent):
    """Compute derivatives of the state values for the LNG boil-off
    process given the current states and other related inputs.  The
    states are the mass of gaseous and liquid methane, the temperature
    of the gas and liquid, and the volume of gas (ullage volume).

    This portion of the code leans on much of the work from Eugina
    Mendez Ramos's thesis (http://hdl.handle.net/1853/64797).  The
    thermophysical properties are replaced with methane-equivalent
    lookups via :class:`LNGProperties`.

    Inputs
    ------
    m_gas : float
        Mass of the gaseous methane in the tank ullage (vector, kg)
    m_liq : float
        Mass of liquid methane in the tank (vector, kg)
    T_gas : float
        Temperature of the gaseous methane in the ullage (vector, K)
    T_liq : float
        Temperature of the bulk liquid methane (vector, K)
    V_gas : float
        Volume of the ullage (vector, m³)
    m_dot_gas_out : float
        Mass flow rate of gaseous methane out of the ullage
        (vector, kg/s)
    m_dot_liq_out : float
        Mass flow rate of liquid methane out of the tank
        (vector, kg/s)
    Q_gas : float
        Heat flow rate through the tank walls into the ullage
        (vector, W)
    Q_liq : float
        Heat flow rate through the tank walls into the bulk liquid
        (vector, W)
    Q_add : float
        Additional heat added directly to the bulk liquid by a heater
        (vector, W)
    A_interface : float
        Area of the liquid-vapour interface (vector, m²)
    L_interface : float
        Characteristic length of the interface (vector, m)

    Outputs
    -------
    m_dot_gas : float
        Rate of change of ullage gas mass (vector, kg/s)
    m_dot_liq : float
        Rate of change of bulk liquid mass (vector, kg/s)
    T_dot_gas : float
        Rate of change of ullage gas temperature (vector, K/s)
    T_dot_liq : float
        Rate of change of bulk liquid temperature (vector, K/s)
    V_dot_gas : float
        Rate of change of ullage volume (vector, m³/s)
    P_gas : float
        Pressure in the ullage (vector, Pa)

    Options
    -------
    num_nodes : int
        Number of analysis points to run (scalar, dimensionless)
    heater_boil_frac : float
        Fraction of heater heat that directly induces boil-off
        (0-1). By default 0.1.
    heat_transfer_C_gas_const : float
        Multiplier on the Nusselt number for ullage convection.
        By default 0.27 / 4.
    heat_transfer_n_gas_const : float
        Exponent on the Prandtl-Grashof product for ullage
        convection. By default 0.25.
    heat_transfer_C_liq_const : float
        Multiplier on the Nusselt number for liquid convection.
        By default 0.13.
    heat_transfer_n_liq_const : float
        Exponent on the Prandtl-Grashof product for liquid
        convection. By default 0.33.
    sigmoid_fac : float
        Sigmoid smoothing factor for the liquid/vapour split of
        heater heat. By default 100.
    heater_rate_const : float or str
        Constant in the heater thermal-inertia ODE (1/s), or the
        string ``"input"`` to make it an input. By default 1e-3.
    """

    def initialize(self):
        self.options.declare('num_nodes', default=1, desc='Number of nodes')
        self.options.declare(
            'heater_boil_frac',
            default=0.1,
            desc='Fraction of heater heat inducing boil-off',
        )
        self.options.declare(
            'heat_transfer_C_gas_const',
            default=0.27 / 4,
            desc='Nusselt multiplier for ullage convection',
        )
        self.options.declare(
            'heat_transfer_n_gas_const',
            default=0.25,
            desc='Prandtl-Grashof exponent for ullage',
        )
        self.options.declare(
            'heat_transfer_C_liq_const',
            default=0.13,
            desc='Nusselt multiplier for liquid convection',
        )
        self.options.declare(
            'heat_transfer_n_liq_const',
            default=0.33,
            desc='Prandtl-Grashof exponent for liquid',
        )
        self.options.declare(
            'sigmoid_fac',
            default=100,
            desc='Sigmoid smoothing factor for heater split',
        )
        self.options.declare(
            'heater_rate_const',
            default=1e-3,
            types=(float, str),
            desc="Heater thermal-inertia constant (1/s) or 'input'",
        )

    def setup(self):
        nn = self.options['num_nodes']

        self.add_input('m_gas', shape=(nn,), val=1.0, units='kg')
        self.add_input('m_liq', shape=(nn,), val=1000.0, units='kg')
        self.add_input('T_gas', shape=(nn,), val=20.0, units='K')
        self.add_input('T_liq', shape=(nn,), val=20.0, units='K')
        self.add_input('V_gas', shape=(nn,), val=1.0, units='m**3')
        self.add_input(
            'm_dot_gas_out',
            shape=(nn,),
            val=0.0,
            units='kg/s',
        )
        self.add_input(
            'm_dot_liq_out',
            shape=(nn,),
            val=0.0,
            units='kg/s',
        )
        self.add_input('Q_gas', shape=(nn,), val=0.0, units='W')
        self.add_input('Q_liq', shape=(nn,), val=0.0, units='W')
        self.add_input('Q_add', shape=(nn,), val=0.0, units='W')
        self.add_input('A_interface', shape=(nn,), val=1.0, units='m**2')
        self.add_input('L_interface', shape=(nn,), val=1.0, units='m')

        self.add_output(
            'm_dot_gas',
            shape=(nn,),
            val=0.0,
            lower=-1e3,
            upper=1e3,
            units='kg/s',
        )
        self.add_output(
            'm_dot_liq',
            shape=(nn,),
            val=0.0,
            lower=-1e3,
            upper=1e3,
            units='kg/s',
        )
        self.add_output(
            'T_dot_gas',
            shape=(nn,),
            val=0.0,
            lower=-1e3,
            upper=1e3,
            units='K/s',
        )
        self.add_output(
            'T_dot_liq',
            shape=(nn,),
            val=0.0,
            lower=-1e3,
            upper=1e3,
            units='K/s',
        )
        self.add_output(
            'V_dot_gas',
            shape=(nn,),
            val=0.0,
            lower=-1e3,
            upper=1e3,
            units='m**3/s',
        )
        self.add_output(
            'P_gas',
            shape=(nn,),
            val=1e5,
            lower=1e3,
            upper=1e8,
            units='Pa',
        )

        # Every output at a node depends only on the inputs at that same
        # node, so the sub-Jacobians are diagonal. CoolProp property
        # lookups are not complex-safe, so the Jacobian is built by
        # central finite difference rather than complex step. Supplying
        # the diagonal sparsity lets OpenMDAO colour the FD so the whole
        # Jacobian costs only one extra compute per input. An absolute
        # step is used because several inputs (Q_add, the mass flows) are
        # zero in the no-heater/ground cases, where a relative step would
        # collapse to zero and zero out their derivatives.
        arng = np.arange(nn)
        self.declare_partials(
            [
                'm_dot_gas',
                'm_dot_liq',
                'T_dot_gas',
                'T_dot_liq',
                'V_dot_gas',
                'P_gas',
            ],
            [
                'm_gas',
                'm_liq',
                'T_gas',
                'T_liq',
                'V_gas',
                'm_dot_gas_out',
                'm_dot_liq_out',
                'Q_gas',
                'Q_liq',
                'Q_add',
                'A_interface',
                'L_interface',
            ],
            method='fd',
            form='central',
            step=1e-6,
            rows=arng,
            cols=arng,
        )
        if self.options['heater_rate_const'] == 'input':
            self.add_input(
                'heater_rate_const',
                val=1e-3,
                units='1/s',
            )
            self.declare_partials(
                'T_dot_gas',
                'heater_rate_const',
                method='fd',
                form='central',
                step=1e-6,
            )

        self._prop = LNGProperties()

    def compute(self, inputs, outputs):
        prop = self._prop

        # Suppress numpy warnings during grid refinement where extreme
        # intermediate values (zero mass, huge volumes) are expected.
        np.seterr(all='ignore')

        m_gas = inputs['m_gas']
        m_liq = inputs['m_liq']
        T_gas = inputs['T_gas']
        T_liq = inputs['T_liq']
        V_gas = inputs['V_gas']
        m_dot_gas_out = inputs['m_dot_gas_out']
        m_dot_liq_out = inputs['m_dot_liq_out']
        Q_gas = inputs['Q_gas']
        Q_liq = inputs['Q_liq']
        Q_add = inputs['Q_add']
        A_int = inputs['A_interface']
        L_int = inputs['L_interface']

        nn = m_gas.shape[0]
        sigmoid_fac = self.options['sigmoid_fac']
        C_gas = self.options['heat_transfer_C_gas_const']
        n_gas = self.options['heat_transfer_n_gas_const']
        C_liq = self.options['heat_transfer_C_liq_const']
        n_liq = self.options['heat_transfer_n_liq_const']
        C_heater = self.options['heater_rate_const']

        # --- Density -------------------------------------------------------
        rho_gas = m_gas / V_gas

        # --- Pressure ------------------------------------------------------
        # Real-gas equation of state via CoolProp
        P_gas = np.zeros(nn)
        for i in range(nn):
            try:
                P_gas[i] = prop.gh2_P(rho_gas[i], T_gas[i])
            except (ValueError, TypeError):
                # Fallback to ideal gas law
                R = 8.3145 / 16.0425  # J/(kg·K) for CH₄
                P_gas[i] = rho_gas[i] * R * T_gas[i]

        # --- Saturation temperature ----------------------------------------
        T_sat = prop.sat_T(P_gas)
        # CoolProp returns NaN when P exceeds critical pressure (~46 bar).
        T_sat = np.where(np.isnan(T_sat), 150.0, T_sat)

        # --- Thermophysical properties at saturation -----------------------
        rho_liq = prop.sat_liq_rho(T_liq)
        rho_liq = np.where(np.isnan(rho_liq), 350.0, rho_liq)
        rho_vap_sat = prop.sat_vap_rho(T_sat)
        rho_vap_sat = np.where(np.isnan(rho_vap_sat), 1.0, rho_vap_sat)
        h_liq = prop.sat_liq_h(T_liq)
        h_liq = np.where(np.isnan(h_liq), 0.0, h_liq)
        u_liq = prop.sat_liq_u(T_liq)
        u_liq = np.where(np.isnan(u_liq), 0.0, u_liq)
        h_vap_sat = prop.sat_vap_h(T_sat)
        h_vap_sat = np.where(np.isnan(h_vap_sat), 5e5, h_vap_sat)
        cp_liq = prop.sat_liq_cp(T_liq)
        cp_liq = np.where(np.isnan(cp_liq), 3500.0, cp_liq)
        cp_vap = prop.sat_vap_cp(T_sat)
        cp_vap = np.where(np.isnan(cp_vap), 2200.0, cp_vap)
        k_liq = prop.sat_liq_k(T_liq)
        k_liq = np.where(np.isnan(k_liq), 0.03, k_liq)
        k_vap = prop.sat_vap_k(T_sat)
        k_vap = np.where(np.isnan(k_vap), 0.02, k_vap)
        mu_liq = prop.sat_liq_viscosity(T_liq)
        mu_liq = np.where(np.isnan(mu_liq), 1e-4, mu_liq)
        mu_vap = prop.sat_vap_viscosity(T_sat)
        mu_vap = np.where(np.isnan(mu_vap), 7e-6, mu_vap)
        beta_liq = prop.sat_liq_beta(T_liq)
        beta_liq = np.where(np.isnan(beta_liq), 0.004, beta_liq)
        beta_vap = prop.sat_vap_beta(T_sat)
        beta_vap = np.where(np.isnan(beta_vap), 0.03, beta_vap)

        # --- Latent heat of vaporization -----------------------------------
        h_fg = h_vap_sat - h_liq
        h_fg = np.where(np.isnan(h_fg), 5e5, h_fg)
        h_fg = np.clip(h_fg, 1e3, 1e6)  # avoid division by zero & overflow

        # --- Heat-transfer coefficients ------------------------------------
        # Ullage (natural convection above hot surface)
        cp_vap_compute = prop.gh2_cp(P_gas, T_gas)
        cp_vap_compute = np.where(
            np.isnan(cp_vap_compute), 2200.0, cp_vap_compute
        )
        Gr_gas = (
            beta_vap
            * (T_gas - T_liq)
            * L_int**3
            / (mu_vap / cp_vap_compute * mu_vap)
        )
        Pr_gas = cp_vap_compute * mu_vap / k_vap
        # Clip to zero when buoyancy is negative (T_gas < T_liq) to avoid
        # NaN from a fractional power of a negative number.
        Ra_gas = np.maximum(0.0, Pr_gas * Gr_gas)
        Nu_gas = C_gas * Ra_gas**n_gas
        h_conv_gas = Nu_gas * k_vap / L_int

        # Liquid (natural convection below hot surface)
        cp_liq_compute = prop.sat_liq_cp(T_liq)
        Gr_liq = (
            beta_liq
            * (T_gas - T_liq)
            * L_int**3
            / (mu_liq / cp_liq_compute * mu_liq)
        )
        Pr_liq = cp_liq_compute * mu_liq / k_liq
        Ra_liq = np.maximum(0.0, Pr_liq * Gr_liq)
        Nu_liq = C_liq * Ra_liq**n_liq
        h_conv_liq = Nu_liq * k_liq / L_int

        # --- Heater heat split ---------------------------------------------
        if sigmoid_fac == 0:
            Q_boil = Q_add
            Q_heat_liq = 0.0
        else:
            # Numerically stable sigmoid: clip argument to [-500, 500]
            # so exp never overflows.  sigmoid_fac * (T_liq - T_sat) can
            # easily exceed 700 during grid-refinement iterations.
            arg = np.clip(-sigmoid_fac * (T_liq - T_sat), -500.0, 500.0)
            sigmoid = 1.0 / (1.0 + np.exp(arg))
            Q_boil = (
                sigmoid_fac * sigmoid * (1 - sigmoid) * Q_add
                + self.options['heater_boil_frac'] * Q_add
            )
            Q_heat_liq = (1 - self.options['heater_boil_frac']) * Q_add

        # --- Mass balance --------------------------------------------------
        dmdt_gas = (
            -m_dot_gas_out
            + Q_gas / h_fg
            + Q_boil / h_fg
            - h_conv_gas * A_int * (T_gas - T_sat) / h_fg
        )
        dmdt_liq = (
            -m_dot_liq_out
            - Q_liq / h_fg
            - Q_heat_liq / h_fg
            + h_conv_liq * A_int * (T_gas - T_liq) / h_fg
        )

        # --- Energy balance ------------------------------------------------
        # Clip mass to avoid division by zero during grid refinement.
        m_gas_safe = np.maximum(m_gas, 1e-6)
        m_liq_safe = np.maximum(m_liq, 1e-6)
        cp_gas = prop.gh2_cp(P_gas, T_gas)
        cp_gas = np.maximum(cp_gas, 1e-3)  # avoid zero denominator

        # Energy convected out with the leaving streams is the flow-work
        # term (h - u): the enthalpy carried out minus the internal
        # energy lost from the control volume. Use the real internal
        # energy from CoolProp, NOT T*cp (which is not internal energy
        # and, because CoolProp enthalpy carries a reference offset,
        # produces an O(1e5 J/kg) spurious sink that drives the liquid
        # below its freezing point and stalls the Newton solve).
        h_gas = prop.gh2_h(P_gas, T_gas)
        h_gas = np.where(np.isnan(h_gas), h_vap_sat, h_gas)
        u_gas = prop.gh2_u(P_gas, T_gas)
        u_gas = np.where(np.isnan(u_gas), h_gas, u_gas)

        T_dot_gas = (
            Q_gas
            + h_conv_gas * A_int * (T_sat - T_gas)
            + m_dot_gas_out * (h_gas - u_gas)
        ) / (m_gas_safe * cp_gas)

        cp_liq_safe = np.maximum(cp_liq, 1e-3)
        T_dot_liq = (
            Q_liq
            + Q_heat_liq
            + h_conv_liq * A_int * (T_gas - T_liq)
            + m_dot_liq_out * (h_liq - u_liq)
        ) / (m_liq_safe * cp_liq_safe)

        # --- Ullage volume rate --------------------------------------------
        # Clip rho_vap_sat to avoid division by zero when ullage density
        # approaches zero (large V_gas during grid refinement).
        rho_vap_sat_safe = np.maximum(rho_vap_sat, 1e-6)
        dVdt_gas = dmdt_gas / rho_vap_sat_safe

        # --- Clip to avoid unphysical values -------------------------------
        dmdt_gas = np.where(np.isnan(dmdt_gas), 0.0, dmdt_gas)
        dmdt_liq = np.where(np.isnan(dmdt_liq), 0.0, dmdt_liq)
        T_dot_gas = np.where(np.isnan(T_dot_gas), 0.0, T_dot_gas)
        T_dot_liq = np.where(np.isnan(T_dot_liq), 0.0, T_dot_liq)
        dVdt_gas = np.where(np.isnan(dVdt_gas), 0.0, dVdt_gas)
        P_gas = np.where(np.isnan(P_gas), 1e5, P_gas)

        outputs['m_dot_gas'] = dmdt_gas
        outputs['m_dot_liq'] = dmdt_liq
        outputs['T_dot_gas'] = T_dot_gas
        outputs['T_dot_liq'] = T_dot_liq
        outputs['V_dot_gas'] = dVdt_gas
        # Clamp pressure to physically meaningful range.
        outputs['P_gas'] = np.clip(P_gas, 1e3, 1e8)


class LNGFullODE(om.Group):
    """Group combining ODE with geometry computations for LNG tanks.

    Mirrors :class:`FullODE` but uses :class:`LNGBoilOffODE` instead
    of :class:`LH2BoilOffODE`.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes',
            default=1,
            desc='Number of design points',
        )
        self.options.declare(
            'end_cap_depth_ratio',
            lower=0.0,
            upper=1.0,
            default=1.0,
            desc='End cap depth / cylinder radius',
        )

    def setup(self):
        nn = self.options['num_nodes']
        depth_ratio = self.options['end_cap_depth_ratio']

        from hytank.boil_off import (
            BoilOffFillLevelCalc,
            BoilOffGeometry,
            LiquidHeight,
        )

        self.add_subsystem(
            'level_calc',
            BoilOffFillLevelCalc(
                num_nodes=nn,
                end_cap_depth_ratio=depth_ratio,
            ),
            promotes_inputs=['radius', 'length', 'V_gas'],
            promotes_outputs=['fill_level'],
        )

        self.add_subsystem(
            'liq_height_calc',
            LiquidHeight(
                num_nodes=nn,
                end_cap_depth_ratio=depth_ratio,
            ),
            promotes_inputs=['radius', 'length'],
        )
        self.add_subsystem(
            'interface_params',
            BoilOffGeometry(
                num_nodes=nn,
                end_cap_depth_ratio=depth_ratio,
            ),
            promotes_inputs=['radius', 'length'],
        )
        self.connect('fill_level', 'liq_height_calc.fill_level')
        self.connect(
            'liq_height_calc.h_liq_frac', 'interface_params.h_liq_frac'
        )

        from hytank.boil_off import HeaterODE

        self.add_subsystem(
            'heater_ode',
            HeaterODE(num_nodes=nn),
            promotes_inputs=['P_heater', 'Q_add'],
            promotes_outputs=['Q_add_dot'],
        )

        self.add_subsystem(
            'boil_off_ode',
            LNGBoilOffODE(num_nodes=nn),
            promotes_inputs=[
                'm_dot_gas_out',
                'm_dot_liq_out',
                'Q_gas',
                'Q_liq',
                'Q_add',
                'm_gas',
                'm_liq',
                'T_gas',
                'T_liq',
                'V_gas',
            ],
            promotes_outputs=[
                'm_dot_gas',
                'm_dot_liq',
                'T_dot_gas',
                'T_dot_liq',
                'V_dot_gas',
                'P_gas',
            ],
        )
        self.connect('interface_params.A_interface', 'boil_off_ode.A_interface')
        self.connect('interface_params.L_interface', 'boil_off_ode.L_interface')

        self.set_input_defaults('radius', 1.0, units='m')
        self.set_input_defaults('length', 0.5, units='m')


class LNGInitialTankStateModification(om.ExplicitComponent):
    """Modify initial state values for LNG (methane) tanks.

    Mirrors :class:`InitialTankStateModification` from
    ``hytank.boil_off`` but uses methane properties via CoolProp
    instead of hydrogen surrogate models.
    """

    def __init__(self, **kwargs):
        """Initialize with methane property backend."""
        self._prop = LNGProperties()
        super().__init__(**kwargs)

    def initialize(self):
        self.options.declare(
            'num_nodes', default=1, desc='Number of design points'
        )
        self.options.declare(
            'fill_level_init', default=0.95, desc='Initial fill level'
        )
        self.options.declare(
            'ullage_T_init',
            default=111.0,
            desc='Initial ullage temp (K)',
        )
        self.options.declare(
            'ullage_P_init',
            default=1.5e5,
            desc='Initial ullage pressure (Pa)',
        )
        self.options.declare(
            'liquid_T_init',
            default=111.0,
            desc='Initial bulk liquid temp (K)',
        )
        self.options.declare(
            'end_cap_depth_ratio',
            lower=0.0,
            upper=1.0,
            default=1.0,
            desc='End cap depth / cylinder radius',
        )

    def setup(self):
        nn = self.options['num_nodes']
        prop = self._prop

        r_default = 1.0
        L_default = 0.5
        self.add_input('radius', val=r_default, units='m')
        self.add_input('length', val=L_default, units='m')

        self.add_input('delta_m_gas', shape=(nn,), units='kg', val=0.0)
        self.add_input('delta_m_liq', shape=(nn,), units='kg', val=0.0)
        self.add_input('delta_T_gas', shape=(nn,), units='K', val=0.0)
        self.add_input('delta_T_liq', shape=(nn,), units='K', val=0.0)
        self.add_input('delta_V_gas', shape=(nn,), units='m**3', val=0.0)

        self.add_input('fill_level_init', val=self.options['fill_level_init'])
        self.add_input(
            'ullage_T_init',
            val=self.options['ullage_T_init'],
            units='K',
        )
        self.add_input(
            'ullage_P_init',
            val=self.options['ullage_P_init'],
            units='Pa',
        )
        self.add_input(
            'liquid_T_init',
            val=self.options['liquid_T_init'],
            units='K',
        )

        # Reasonable defaults for methane states
        defaults = self._compute_initial_states(
            r_default,
            L_default,
            self.options,
        )
        self.add_output(
            'm_gas',
            shape=(nn,),
            units='kg',
            lower=1e-6,
            val=defaults['m_gas_init'],
            upper=1e5,
        )
        self.add_output(
            'm_liq',
            shape=(nn,),
            units='kg',
            lower=1e-2,
            val=defaults['m_liq_init'],
            upper=1e7,
        )
        self.add_output(
            'T_gas',
            shape=(nn,),
            units='K',
            lower=90,
            val=defaults['T_gas_init'],
            upper=300,
        )
        self.add_output(
            'T_liq',
            shape=(nn,),
            units='K',
            lower=90,
            val=defaults['T_liq_init'],
            upper=150,
        )
        self.add_output(
            'V_gas',
            shape=(nn,),
            units='m**3',
            lower=1e-5,
            val=defaults['V_gas_init'],
            upper=1e4,
        )

        arng = np.arange(nn)
        self.declare_partials(
            'V_gas',
            'delta_V_gas',
            rows=arng,
            cols=arng,
            val=np.ones(nn),
        )
        self.declare_partials(
            'm_liq',
            'delta_m_liq',
            rows=arng,
            cols=arng,
            val=np.ones(nn),
        )
        self.declare_partials(
            'm_gas',
            'delta_m_gas',
            rows=arng,
            cols=arng,
            val=np.ones(nn),
        )
        self.declare_partials(
            'T_gas',
            'delta_T_gas',
            rows=arng,
            cols=arng,
            val=np.ones(nn),
        )
        self.declare_partials(
            'T_liq',
            'delta_T_liq',
            rows=arng,
            cols=arng,
            val=np.ones(nn),
        )
        self.declare_partials(
            ['V_gas', 'm_gas', 'm_liq'],
            ['radius', 'length', 'fill_level_init'],
            rows=arng,
            cols=np.zeros(nn),
        )
        self.declare_partials(
            'T_gas',
            'ullage_T_init',
            rows=arng,
            cols=np.zeros(nn),
            val=np.ones(nn),
        )
        self.declare_partials(
            'T_liq',
            'liquid_T_init',
            rows=arng,
            cols=np.zeros(nn),
            val=np.ones(nn),
        )
        self.declare_partials(
            'm_gas',
            ['ullage_T_init', 'ullage_P_init'],
            rows=arng,
            cols=np.zeros(nn),
        )
        self.declare_partials(
            'm_liq',
            'liquid_T_init',
            rows=arng,
            cols=np.zeros(nn),
        )

    def compute(self, inputs, outputs):
        init_states = self._compute_initial_states(
            inputs['radius'],
            inputs['length'],
            inputs,
        )
        outputs['V_gas'] = inputs['delta_V_gas'] + init_states['V_gas_init']
        outputs['m_gas'] = inputs['delta_m_gas'] + init_states['m_gas_init']
        outputs['m_liq'] = inputs['delta_m_liq'] + init_states['m_liq_init']
        outputs['T_gas'] = inputs['delta_T_gas'] + init_states['T_gas_init']
        outputs['T_liq'] = inputs['delta_T_liq'] + init_states['T_liq_init']

    def compute_partials(self, inputs, J):
        from CoolProp.CoolProp import PropsSI

        r = inputs['radius']
        L = inputs['length']
        fill_init = inputs['fill_level_init']
        T_gas_init = inputs['ullage_T_init']
        T_liq_init = inputs['liquid_T_init']
        P_init = inputs['ullage_P_init']
        d_end = self.options['end_cap_depth_ratio']
        nn = int(J['V_gas', 'delta_V_gas'].shape[0])

        V_tank = 4 / 3 * np.pi * r**3 * d_end + np.pi * r**2 * L
        Vtank_r = 4 * np.pi * r**2 * d_end + 2 * np.pi * r * L
        Vtank_L = np.pi * r**2

        V_gas_frac = 1 - fill_init
        J['V_gas', 'radius'] = Vtank_r * V_gas_frac
        J['V_gas', 'length'] = Vtank_L * V_gas_frac

        # Methane gas density via CoolProp
        rho_gas = PropsSI('D', 'P', P_init, 'T', T_gas_init, 'Methane')
        J['m_gas', 'radius'] = rho_gas * J['V_gas', 'radius']
        J['m_gas', 'length'] = rho_gas * J['V_gas', 'length']

        # Methane liquid density via CoolProp
        rho_liq = PropsSI('D', 'T', T_liq_init, 'Q', 0, 'Methane')
        V_liq_r = Vtank_r - J['V_gas', 'radius']
        V_liq_L = Vtank_L - J['V_gas', 'length']
        J['m_liq', 'radius'] = V_liq_r * rho_liq
        J['m_liq', 'length'] = V_liq_L * rho_liq

        J['V_gas', 'fill_level_init'] = -V_tank
        J['m_gas', 'fill_level_init'] = rho_gas * J['V_gas', 'fill_level_init']

        # Finite-difference density derivatives
        dP = max(P_init * 1e-6, 1.0)
        dT = 1e-6
        rho_gas_dP_plus = PropsSI(
            'D',
            'P',
            P_init + dP,
            'T',
            T_gas_init,
            'Methane',
        )
        rho_gas_dP_minus = PropsSI(
            'D',
            'P',
            P_init - dP,
            'T',
            T_gas_init,
            'Methane',
        )
        drho_dP = (rho_gas_dP_plus - rho_gas_dP_minus) / (2 * dP)
        V_gas_init = V_tank * V_gas_frac
        drho_dT = (
            PropsSI('D', 'P', P_init, 'T', T_gas_init + dT, 'Methane')
            - PropsSI('D', 'P', P_init, 'T', T_gas_init - dT, 'Methane')
        ) / (2 * dT)
        J['m_gas', 'ullage_T_init'] = drho_dT * V_gas_init
        J['m_gas', 'ullage_P_init'] = drho_dP * V_gas_init

        J['m_liq', 'fill_level_init'] = -J['V_gas', 'fill_level_init'] * rho_liq
        rho_liq_dT = (
            PropsSI('D', 'T', T_liq_init + dT, 'Q', 0, 'Methane')
            - PropsSI('D', 'T', T_liq_init - dT, 'Q', 0, 'Methane')
        ) / (2 * dT)
        J['m_liq', 'liquid_T_init'] = V_tank * fill_init * rho_liq_dT

    def _compute_initial_states(self, radius, length, init_vals):
        """Compute initial methane state values from tank geometry."""
        from CoolProp.CoolProp import PropsSI

        fill_init = init_vals['fill_level_init']
        T_gas_init = init_vals['ullage_T_init']
        T_liq_init = init_vals['liquid_T_init']
        P_init = init_vals['ullage_P_init']
        d_end = self.options['end_cap_depth_ratio']

        V_tank = 4 / 3 * np.pi * radius**3 * d_end + np.pi * radius**2 * length
        V_gas_init = V_tank * (1 - fill_init)

        res = {
            'T_liq_init': T_liq_init,
            'T_gas_init': T_gas_init,
            'V_gas_init': V_gas_init,
            'm_gas_init': PropsSI(
                'D',
                'P',
                P_init,
                'T',
                T_gas_init,
                'Methane',
            )
            * V_gas_init,
            'm_liq_init': (V_tank - V_gas_init)
            * PropsSI('D', 'T', T_liq_init, 'Q', 0, 'Methane'),
        }
        return res


class LNGBoilOff(om.Group):
    """Time-integrated boil-off model for LNG (methane) tanks.

    Mirrors :class:`BoilOff` but uses :class:`LNGFullODE` instead of
    :class:`FullODE`.  All initial-condition defaults are set to LNG
    saturation values (~111 K at ~1 atm).

    See :class:`BoilOff` for full input/output/option documentation.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes',
            default=1,
            desc='Number of design points',
        )
        self.options.declare(
            'fill_level_init',
            default=0.95,
            desc='Initial fill level',
        )
        self.options.declare(
            'ullage_T_init',
            default=111.0,
            desc='Initial ullage temp (K)',
        )
        self.options.declare(
            'ullage_P_init',
            default=1.5e5,
            desc='Initial ullage pressure (Pa)',
        )
        self.options.declare(
            'liquid_T_init',
            default=111.0,
            desc='Initial liquid temp (K)',
        )
        self.options.declare(
            'end_cap_depth_ratio',
            lower=0.0,
            upper=1.0,
            default=1.0,
            desc='End cap depth / cylinder radius',
        )
        self.options.declare(
            'heater_Q_add_init',
            default=0.0,
            types=float,
            desc='Initial heat input from heater (W)',
        )

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(
            'ode',
            LNGFullODE(
                num_nodes=nn,
                end_cap_depth_ratio=self.options['end_cap_depth_ratio'],
            ),
            promotes_inputs=[
                'radius',
                'length',
                'P_heater',
                'm_dot_gas_out',
                'm_dot_liq_out',
                'Q_gas',
                'Q_liq',
                'm_gas',
                'm_liq',
                'T_gas',
                'T_liq',
            ],
            promotes_outputs=['fill_level', 'P_gas'],
        )

        integ = self.add_subsystem(
            'integ',
            Integrator(
                num_nodes=nn,
                diff_units='s',
                time_setup='duration',
                method='bdf3',
            ),
        )
        integ.add_integrand(
            'delta_m_gas',
            rate_name='m_dot_gas',
            units='kg',
            val=0,
            start_val=0,
        )
        integ.add_integrand(
            'delta_m_liq',
            rate_name='m_dot_liq',
            units='kg',
            val=0,
            start_val=0,
        )
        integ.add_integrand(
            'delta_T_gas',
            rate_name='T_dot_gas',
            units='K',
            val=0,
            start_val=0,
        )
        integ.add_integrand(
            'delta_T_liq',
            rate_name='T_dot_liq',
            units='K',
            val=0,
            start_val=0,
        )
        integ.add_integrand(
            'delta_V_gas',
            rate_name='V_dot_gas',
            units='m**3',
            val=0,
            start_val=0,
        )
        integ.add_integrand(
            'Q_add',
            rate_name='Q_add_dot',
            units='W',
            val=0,
            start_val=self.options['heater_Q_add_init'],
        )

        self.add_subsystem(
            'add_init_state_values',
            LNGInitialTankStateModification(
                num_nodes=nn,
                fill_level_init=self.options['fill_level_init'],
                ullage_T_init=self.options['ullage_T_init'],
                ullage_P_init=self.options['ullage_P_init'],
                liquid_T_init=self.options['liquid_T_init'],
                end_cap_depth_ratio=self.options['end_cap_depth_ratio'],
            ),
            promotes_inputs=['radius', 'length'],
            promotes_outputs=['m_liq', 'm_gas', 'T_liq', 'T_gas'],
        )

        self.connect('integ.delta_m_gas', 'add_init_state_values.delta_m_gas')
        self.connect('integ.delta_m_liq', 'add_init_state_values.delta_m_liq')
        self.connect('integ.delta_T_gas', 'add_init_state_values.delta_T_gas')
        self.connect('integ.delta_T_liq', 'add_init_state_values.delta_T_liq')
        self.connect('integ.delta_V_gas', 'add_init_state_values.delta_V_gas')
        self.connect('integ.Q_add', 'ode.Q_add')

        self.connect('ode.m_dot_gas', 'integ.m_dot_gas')
        self.connect('ode.m_dot_liq', 'integ.m_dot_liq')
        self.connect('ode.T_dot_gas', 'integ.T_dot_gas')
        self.connect('ode.T_dot_liq', 'integ.T_dot_liq')
        self.connect('ode.V_dot_gas', 'integ.V_dot_gas')
        self.connect('ode.Q_add_dot', 'integ.Q_add_dot')
        self.connect('add_init_state_values.V_gas', 'ode.V_gas')

        self.linear_solver = om.DirectSolver()
        self.nonlinear_solver = om.NewtonSolver()
        self.nonlinear_solver.options['solve_subsystems'] = False
        self.nonlinear_solver.options['err_on_non_converge'] = True
        self.nonlinear_solver.options['restart_from_successful'] = True
        self.nonlinear_solver.options['maxiter'] = 30
        self.nonlinear_solver.options['iprint'] = 2
        self.nonlinear_solver.options['atol'] = 1e-9
        self.nonlinear_solver.options['rtol'] = 1e-12
        self.nonlinear_solver.linesearch = om.ArmijoGoldsteinLS(
            bound_enforcement='scalar',
            alpha=1.0,
            iprint=0,
            print_bound_enforce=False,
        )

        # Flag to track whether guess_nonlinear is called by this group's solver
        self._in_my_solve_nl = False

    def _solve_nonlinear(self):
        """Track when this group's solver calls guess_nonlinear."""
        self._in_my_solve_nl = True
        try:
            super()._solve_nonlinear()
        finally:
            self._in_my_solve_nl = False

    def _mpi_print_stuff(self, text):
        """Print with the Newton solver prefix."""
        if self.nonlinear_solver.options['iprint'] > 0:
            prefix = self.nonlinear_solver._solver_info.prefix
            print(f'{prefix}{text}')

    def guess_nonlinear(self, inputs, outputs, resids):
        """Generate initial guesses via SciPy IVP solver for methane.

        Mirrors :meth:`BoilOff.guess_nonlinear` but uses
        :class:`LNGFullODE` and :class:`LNGProperties`.
        """
        import scipy.integrate
        import scipy.interpolate

        # If already converged, bail early
        norm = resids.get_norm()
        zero_resids = np.all(resids.asarray() < 1e-14)
        if norm < 1e-2 and not zero_resids:
            return

        # Initial tank properties from options
        r = inputs['ode.level_calc.radius'].item()
        L = inputs['ode.level_calc.length'].item()
        fill_init = self.options['fill_level_init']
        T_gas_init = self.options['ullage_T_init']
        P_gas_init = self.options['ullage_P_init']
        T_liq_init = self.options['liquid_T_init']
        d_end = self.options['end_cap_depth_ratio']

        # Compute initial masses using methane properties
        V_tank = 4 / 3 * np.pi * r**3 * d_end + np.pi * r**2 * L
        V_gas_init = V_tank * (1 - fill_init)
        prop = LNGProperties()
        m_gas_init = prop.gh2_rho(P_gas_init, T_gas_init) * V_gas_init
        m_liq_init = (V_tank - V_gas_init) * prop.sat_liq_rho(T_liq_init)

        # Guard against NaN/inf in initial mass computation
        if np.any(np.isnan(m_gas_init)) or m_gas_init < 1e-6:
            m_gas_init = max(1.0, m_gas_init)
        if np.any(np.isnan(m_liq_init)) or m_liq_init < 1e-6:
            m_liq_init = max(100.0, m_liq_init)

        def get_ode_problem(num_nodes=1):
            """Build a standalone LNGFullODE problem for IVP evaluation."""
            p = om.Problem(reports=False)
            p.model = LNGFullODE(
                num_nodes=num_nodes,
                end_cap_depth_ratio=d_end,
            )
            ode_opts = self.ode.boil_off_ode.options
            heater_opts = self.ode.heater_ode.options
            p.model_options['*'] = {
                'heater_boil_frac': ode_opts['heater_boil_frac'],
                'heat_transfer_C_gas_const': (
                    ode_opts['heat_transfer_C_gas_const']
                ),
                'heat_transfer_n_gas_const': (
                    ode_opts['heat_transfer_n_gas_const']
                ),
                'heat_transfer_C_liq_const': (
                    ode_opts['heat_transfer_C_liq_const']
                ),
                'heat_transfer_n_liq_const': (
                    ode_opts['heat_transfer_n_liq_const']
                ),
                'sigmoid_fac': ode_opts['sigmoid_fac'],
                'heater_rate_const': heater_opts['heater_rate_const'],
            }
            p.setup()

            # Configure liq_height_calc Newton solver
            p.model.liq_height_calc.linear_solver = om.DirectSolver()
            p.model.liq_height_calc.nonlinear_solver = om.NewtonSolver()
            p.model.liq_height_calc.nonlinear_solver.options[
                'solve_subsystems'
            ] = False
            p.model.liq_height_calc.nonlinear_solver.options[
                'err_on_non_converge'
            ] = True
            p.model.liq_height_calc.nonlinear_solver.options['maxiter'] = 5
            p.model.liq_height_calc.nonlinear_solver.options['iprint'] = 0
            p.model.liq_height_calc.nonlinear_solver.options['atol'] = 1e-9
            p.model.liq_height_calc.nonlinear_solver.options['rtol'] = 1e-12
            p.model.liq_height_calc.nonlinear_solver.linesearch = (
                om.ArmijoGoldsteinLS(
                    bound_enforcement='scalar',
                    alpha=1.0,
                    iprint=0,
                    print_bound_enforce=False,
                )
            )

            p.set_val('radius', r, units='m')
            p.set_val('length', L, units='m')
            return p

        # ======================================================================
        # Try solving an initial value problem to get guesses for the states
        # ======================================================================
        # Run the initial value problem solver only when this guess_nonlinear
        # is called by this group's own solver
        if self._in_my_solve_nl:
            self._mpi_print_stuff(
                'Solving boil-off IVP to generate initial guesses...',
            )
            p = get_ode_problem(num_nodes=1)

        u_init = np.array(
            [
                m_gas_init + inputs['integ.delta_m_gas_initial'].item(),
                m_liq_init + inputs['integ.delta_m_liq_initial'].item(),
                T_gas_init + inputs['integ.delta_T_gas_initial'].item(),
                T_liq_init + inputs['integ.delta_T_liq_initial'].item(),
                V_gas_init + inputs['integ.delta_V_gas_initial'].item(),
                inputs['integ.Q_add_initial'].item(),
            ]
        )

        # Build splines for control inputs
        input_splines = {}
        t_span = (0, inputs['integ.duration'].item())
        t = np.linspace(*t_span, self.options['num_nodes'])
        for i_name in [
            'm_dot_gas_out',
            'm_dot_liq_out',
            'Q_liq',
            'Q_gas',
            'P_heater',
        ]:
            input_splines[i_name] = scipy.interpolate.Akima1DInterpolator(
                t,
                inputs[i_name],
            )

        def state_deriv_function(t_val, u):
            """Compute state derivatives at time t_val."""
            p.set_val('m_gas', u[0], units='kg')
            p.set_val('m_liq', u[1], units='kg')
            p.set_val('T_gas', u[2], units='K')
            p.set_val('T_liq', u[3], units='K')
            p.set_val('V_gas', u[4], units='m**3')
            p.set_val('Q_add', u[5], units='W')

            p.set_val(
                'm_dot_gas_out',
                input_splines['m_dot_gas_out'](t_val),
                units='kg/s',
            )
            p.set_val(
                'm_dot_liq_out',
                input_splines['m_dot_liq_out'](t_val),
                units='kg/s',
            )
            p.set_val(
                'Q_gas',
                input_splines['Q_gas'](t_val),
                units='W',
            )
            p.set_val(
                'Q_liq',
                input_splines['Q_liq'](t_val),
                units='W',
            )
            p.set_val(
                'P_heater',
                input_splines['P_heater'](t_val),
                units='W',
            )
            p.run_model()
            vals = np.array(
                [
                    p.get_val('m_dot_gas', units='kg/s').item(),
                    p.get_val('m_dot_liq', units='kg/s').item(),
                    p.get_val('T_dot_gas', units='K/s').item(),
                    p.get_val('T_dot_liq', units='K/s').item(),
                    p.get_val('V_dot_gas', units='m**3/s').item(),
                    p.get_val('Q_add_dot', units='W/s').item(),
                ]
            )
            # Replace NaN/inf derivatives with zero to prevent IVP
            # solver from diverging during grid refinement.
            return np.where(np.isfinite(vals), vals, 0.0)

        # Let SciPy's BDF build and adaptively step its own internal
        # Jacobian. Passing an explicit Jacobian here (the FD
        # compute_totals of the ODE) made BDF's corrector overshoot
        # toward the methane critical point and produced garbage guess
        # trajectories; the internal estimate is better scaled and
        # integrates this stiff draining problem cleanly.
        try:
            sol = scipy.integrate.solve_ivp(
                state_deriv_function,
                t_span,
                u_init,
                method='BDF',
                t_eval=t,
                rtol=1e-3,
                atol=1e-6,
                max_step=inputs['integ.duration'].item() / 100,
            )
            success = sol.success
            # Also check for NaN in the solution
            if success:
                for row in sol.y:
                    if not np.all(np.isfinite(row)):
                        success = False
                        break
        except (ValueError, RuntimeError):
            success = False

        if success:
            self._mpi_print_stuff('    ...succeeded')

            outputs['m_gas'] = m_gas = sol.y[0]
            outputs['m_liq'] = m_liq = sol.y[1]
            outputs['T_gas'] = T_gas = sol.y[2]
            outputs['T_liq'] = T_liq = sol.y[3]
            outputs['add_init_state_values.V_gas'] = V_gas = sol.y[4]
            outputs['integ.Q_add'] = Q_add = sol.y[5]

            outputs['integ.delta_m_gas'] = m_gas - m_gas_init
            outputs['integ.delta_m_liq'] = m_liq - m_liq_init
            outputs['integ.delta_T_gas'] = T_gas - T_gas_init
            outputs['integ.delta_T_liq'] = T_liq - T_liq_init
            outputs['integ.delta_V_gas'] = V_gas - V_gas_init

            outputs['integ.delta_m_gas_final'] = outputs['integ.delta_m_gas'][
                -1
            ]
            outputs['integ.delta_m_liq_final'] = outputs['integ.delta_m_liq'][
                -1
            ]
            outputs['integ.delta_T_gas_final'] = outputs['integ.delta_T_gas'][
                -1
            ]
            outputs['integ.delta_T_liq_final'] = outputs['integ.delta_T_liq'][
                -1
            ]
            outputs['integ.delta_V_gas_final'] = outputs['integ.delta_V_gas'][
                -1
            ]
            outputs['integ.Q_add_final'] = Q_add[-1]

            outputs['fill_level'] = 1 - V_gas / V_tank
            prop = LNGProperties()
            outputs['P_gas'] = prop.gh2_P(m_gas / V_gas, T_gas)

            # Geometric properties at all nodes
            p_geo = get_ode_problem(num_nodes=self.options['num_nodes'])
            p_geo.set_val('m_gas', m_gas, units='kg')
            p_geo.set_val('m_liq', m_liq, units='kg')
            p_geo.set_val('T_gas', T_gas, units='K')
            p_geo.set_val('T_liq', T_liq, units='K')
            p_geo.set_val('V_gas', V_gas, units='m**3')
            p_geo.set_val('Q_add', Q_add, units='W')
            p_geo.set_val(
                'm_dot_gas_out',
                input_splines['m_dot_gas_out'](t),
                units='kg/s',
            )
            p_geo.set_val(
                'm_dot_liq_out',
                input_splines['m_dot_liq_out'](t),
                units='kg/s',
            )
            p_geo.set_val(
                'Q_gas',
                input_splines['Q_gas'](t),
                units='W',
            )
            p_geo.set_val(
                'Q_liq',
                input_splines['Q_liq'](t),
                units='W',
            )
            p_geo.set_val(
                'P_heater',
                input_splines['P_heater'](t),
                units='W',
            )
            p_geo.run_model()

            outputs['ode.interface_params.A_interface'] = p_geo.get_val(
                'interface_params.A_interface'
            )
            outputs['ode.interface_params.L_interface'] = p_geo.get_val(
                'interface_params.L_interface'
            )
            outputs['ode.interface_params.A_dry'] = p_geo.get_val(
                'interface_params.A_dry'
            )
            outputs['ode.interface_params.A_wet'] = p_geo.get_val(
                'interface_params.A_wet'
            )
            outputs['ode.liq_height_calc.h_liq_frac'] = p_geo.get_val(
                'liq_height_calc.h_liq_frac'
            )
            outputs['ode.boil_off_ode.m_dot_gas'] = p_geo.get_val(
                'boil_off_ode.m_dot_gas'
            )
            outputs['ode.boil_off_ode.m_dot_liq'] = p_geo.get_val(
                'boil_off_ode.m_dot_liq'
            )
            outputs['ode.boil_off_ode.T_dot_gas'] = p_geo.get_val(
                'boil_off_ode.T_dot_gas'
            )
            outputs['ode.boil_off_ode.T_dot_liq'] = p_geo.get_val(
                'boil_off_ode.T_dot_liq'
            )
            outputs['ode.boil_off_ode.V_dot_gas'] = p_geo.get_val(
                'boil_off_ode.V_dot_gas'
            )
            outputs['ode.heater_ode.Q_add_dot'] = p_geo.get_val(
                'heater_ode.Q_add_dot'
            )
            return

        self._mpi_print_stuff('    ...IVP failed, falling back to')
        self._mpi_print_stuff('    ...constant initial-state guess')

        # Fallback: constant initial-state guess
        self._mpi_print_stuff('Skipping boil-off initial guess process')
        outputs['m_gas'] = (
            m_gas_init + inputs['integ.delta_m_gas_initial'].item()
        )
        outputs['m_liq'] = (
            m_liq_init + inputs['integ.delta_m_liq_initial'].item()
        )
        outputs['T_gas'] = (
            T_gas_init + inputs['integ.delta_T_gas_initial'].item()
        )
        outputs['T_liq'] = (
            T_liq_init + inputs['integ.delta_T_liq_initial'].item()
        )
        outputs['add_init_state_values.V_gas'] = (
            V_gas_init + inputs['integ.delta_V_gas_initial'].item()
        )
        outputs['integ.delta_m_gas'] = inputs[
            'integ.delta_m_gas_initial'
        ].item()
        outputs['integ.delta_m_liq'] = inputs[
            'integ.delta_m_liq_initial'
        ].item()
        outputs['integ.delta_T_gas'] = inputs[
            'integ.delta_T_gas_initial'
        ].item()
        outputs['integ.delta_T_liq'] = inputs[
            'integ.delta_T_liq_initial'
        ].item()
        outputs['integ.delta_V_gas'] = inputs[
            'integ.delta_V_gas_initial'
        ].item()


class LNGTankThermals(om.Group):
    """Thermal model of an LNG (liquefied methane) storage tank.

    Combines heat-leak computation through vacuum MLI insulation with
    a time-integrated boil-off model using methane thermophysical
    properties.  The tank geometry is a cylinder with hemispherical
    end caps.

          |--- length ---|
         . -------------- .         ---
      ,'                    `.       | radius
     /                        \\      |
    |                          |    ---
     \\                        /
      `.                    ,'
         ` -------------- '

    Inputs
    ------
    radius : float
        Inner radius of the cylinder (scalar, m).
    length : float
        Length of the cylindrical section (scalar, m).
    P_heater : float
        Power added to the resistive heater (vector, W).
    m_dot_gas_out : float
        Gaseous methane outflow rate (vector, kg/s).
    m_dot_liq_out : float
        Liquid methane extraction rate (vector, kg/s).
    T_env : float
        External environment temperature (vector, K).
    N_layers : float
        Number of MLI reflective shield layers (scalar).

    Outputs
    -------
    m_gas : float
        Mass of gaseous methane in the ullage (vector, kg).
    m_liq : float
        Mass of liquid methane in the tank (vector, kg).
    T_gas : float
        Ullage gas temperature (vector, K).
    T_liq : float
        Bulk liquid temperature (vector, K).
    P : float
        Ullage pressure (vector, Pa).
    fill_level : float
        Fraction of tank volume filled with liquid.

    Options
    -------
    num_nodes : int
        Number of analysis points (scalar).
    fill_level_init : float
        Initial fill level (0-1). Default 0.95.
    ullage_T_init : float
        Initial ullage temperature (K). Default 111.0 (LNG saturation).
    ullage_P_init : float
        Initial ullage pressure (Pa). Default 1.5e5 (1.5 bar).
    liquid_T_init : float
        Initial liquid temperature (K). Default 111.0.
    heat_multiplier : float
        Multiplier for heat leak (supports, connections). Default 2.0.
    end_cap_depth_ratio : float
        End cap depth / cylinder radius. 1 = hemisphere. Default 1.0.
    heater_Q_add_init : float
        Initial heater heat input (W). Default 0.0.
    """

    def initialize(self):
        self.options.declare(
            'num_nodes',
            default=1,
            desc='Number of design points',
        )
        self.options.declare(
            'fill_level_init',
            default=0.95,
            desc='Initial fill level',
        )
        self.options.declare(
            'ullage_T_init',
            default=111.0,
            desc='Initial ullage temp (K)',
        )
        self.options.declare(
            'ullage_P_init',
            default=1.5e5,
            desc='Initial ullage pressure (Pa)',
        )
        self.options.declare(
            'liquid_T_init',
            default=111.0,
            desc='Initial liquid temp (K)',
        )
        self.options.declare(
            'heat_multiplier',
            default=2.0,
            desc='Heat leak multiplier',
        )
        self.options.declare(
            'end_cap_depth_ratio',
            lower=0.0,
            upper=1.0,
            default=1.0,
            desc='End cap depth / cylinder radius',
        )
        self.options.declare(
            'heater_Q_add_init',
            default=0.0,
            types=float,
            desc='Initial heat input from heater (W)',
        )

    def setup(self):
        nn = self.options['num_nodes']

        self.add_subsystem(
            'heat_leak',
            HeatTransferVacuumTank(
                num_nodes=nn,
                heat_multiplier=self.options['heat_multiplier'],
            ),
            promotes_inputs=[
                'T_env',
                'N_layers',
                'T_liq',
                'T_gas',
            ],
        )

        self.add_subsystem(
            'boil_off',
            LNGBoilOff(
                num_nodes=nn,
                fill_level_init=self.options['fill_level_init'],
                ullage_T_init=self.options['ullage_T_init'],
                ullage_P_init=self.options['ullage_P_init'],
                liquid_T_init=self.options['liquid_T_init'],
                end_cap_depth_ratio=self.options['end_cap_depth_ratio'],
                heater_Q_add_init=self.options['heater_Q_add_init'],
            ),
            promotes_inputs=[
                'radius',
                'length',
                'm_dot_gas_out',
                'm_dot_liq_out',
                'P_heater',
            ],
            promotes_outputs=[
                'm_gas',
                'm_liq',
                'T_gas',
                'T_liq',
                ('P_gas', 'P'),
                'fill_level',
            ],
        )

        self.connect('heat_leak.Q_gas', 'boil_off.Q_gas')
        self.connect('heat_leak.Q_liq', 'boil_off.Q_liq')
        self.connect('boil_off.ode.interface_params.A_wet', 'heat_leak.A_wet')
        self.connect('boil_off.ode.interface_params.A_dry', 'heat_leak.A_dry')

        self.set_input_defaults('radius', 1.0, units='m')
        self.set_input_defaults('N_layers', 20)
        self.set_input_defaults('P_heater', np.zeros(nn), units='W')
        self.set_input_defaults('T_env', np.full(nn, 300), units='K')
        self.set_input_defaults('T_liq', np.full(nn, 111), units='K')
        self.set_input_defaults('T_gas', np.full(nn, 111), units='K')


if __name__ == '__main__':
    import openmdao.api as om

    duration = 10.0  # hr
    nn = 11  # odd for BDF3

    p = om.Problem(reports=False)
    p.model.add_subsystem(
        'tank',
        LNGTankThermals(num_nodes=nn),
        promotes=['*'],
    )
    p.model.nonlinear_solver = om.NewtonSolver(iprint=2)
    p.model.linear_solver = om.DirectSolver()

    p.setup(force_alloc_complex=True)

    p.set_val('boil_off.integ.duration', duration, units='h')
    p.set_val('radius', 1.5, units='m')
    p.set_val('length', 6.0, units='m')
    p.set_val('P_heater', 0.0, units='W')
    p.set_val('m_dot_gas_out', 0.0, units='kg/s')
    p.set_val('m_dot_liq_out', 0.0, units='kg/s')
    p.set_val('T_env', 300, units='K')
    p.set_val('N_layers', 30)
    p.set_val('environment_design_pressure', 1, units='atm')
    p.set_val('max_expected_operating_pressure', 5, units='bar')
    p.set_val('vacuum_gap', 5, units='cm')

    p.run_model()

    print(f'm_liq start: {p.get_val("m_liq", units="kg")[0]:.1f} kg')
    print(f'm_liq end:   {p.get_val("m_liq", units="kg")[-1]:.1f} kg')
    print(f'P start:     {p.get_val("P", units="bar")[0]:.2f} bar')
    print(f'P end:       {p.get_val("P", units="bar")[-1]:.2f} bar')
    print(f'T_liq start: {p.get_val("T_liq", units="K")[0]:.2f} K')
    print(f'T_liq end:   {p.get_val("T_liq", units="K")[-1]:.2f} K')
    print(f'T_gas start: {p.get_val("T_gas", units="K")[0]:.2f} K')
    print(f'T_gas end:   {p.get_val("T_gas", units="K")[-1]:.2f} K')
