"""Generate methane (LNG) property data files from CoolProp.

This script mirrors the NIST tab-separated schema used by
``H2_property_data/saturated_properties.txt`` and
``H2_property_data/<P>_bar_properties.txt`` but populates the values
with methane (CH4) data from CoolProp's HEOS backend. Output files
are written next to this script.

Run once after CoolProp is installed:

    pixi run python HyTank/hytank/CH4_property_data/generate_from_coolprop.py

Re-running overwrites existing files. Delete
``saturated_property_surrogate_models.pkl`` and
``real_gas_property_surrogate_models.pkl`` (in this directory) after
regenerating data so ``MethaneProperties`` retrains its surrogates.
"""

import os

import numpy as np

# CoolProp is a hard dependency of this generator (not of the runtime
# surrogate class), so the import lives here.
from CoolProp.CoolProp import PropsSI


FLUID = "Methane"

# Saturation curve: from above the triple point (~90.694 K) up to a
# point safely below the critical temperature (190.564 K).
SAT_T_MIN = 91.0
SAT_T_MAX = 190.0
SAT_N_POINTS = 200

# Real-gas pressure ladder. Each value is a pressure in bar; for each
# one we sweep a temperature range that spans the liquid side (below
# saturation) and the vapor side (above saturation) so the parser's
# liquid/vapor split mirrors the H2 schema.
BAR_PRESSURES = [1.0, 2.0, 5.0, 10.0, 20.0, 30.0, 40.0]
BAR_T_MIN = 95.0
BAR_T_MAX = 350.0
BAR_N_POINTS = 200

# Tab character used to delimit columns.
SEP = "\t"

# Column header for saturated_properties.txt. The names are kept
# byte-identical to the H2 NIST schema so the existing data parser
# and surrogate code can be reused without modification.
SAT_COLUMNS = [
    "Temperature (K)",
    "Pressure (MPa)",
    "Density (l, kg/m3)",
    "Volume (l, m3/kg)",
    "Internal Energy (l, kJ/kg)",
    "Enthalpy (l, kJ/kg)",
    "Entropy (l, J/g*K)",
    "Cv (l, J/g*K)",
    "Cp (l, J/g*K)",
    "Sound Spd. (l, m/s)",
    "Joule-Thomson (l, K/MPa)",
    "Viscosity (l, Pa*s)",
    "Therm. Cond. (l, W/m*K)",
    "Surf. Tension (l, N/m)",
    "Thermal Expansion Coefficient (l, 1/K)",
    "Density (v, kg/m3)",
    "Volume (v, m3/kg)",
    "Internal Energy (v, kJ/kg)",
    "Enthalpy (v, kJ/kg)",
    "Entropy (v, J/g*K)",
    "Cv (v, J/g*K)",
    "Cp (v, J/g*K)",
    "Sound Spd. (v, m/s)",
    "Joule-Thomson (v, K/MPa)",
    "Viscosity (v, Pa*s)",
    "Therm. Cond. (v, W/m*K)",
    "Thermal Expansion Coefficient (v, 1/K)",
]

# Column header for <P>_bar_properties.txt. Trailing "Phase" column
# carries the literal strings "liquid" or "vapor", matching the H2
# file format (which the parser uses to split sides).
BAR_COLUMNS = [
    "Temperature (K)",
    "Pressure (MPa)",
    "Density (kg/m3)",
    "Volume (m3/kg)",
    "Internal Energy (kJ/kg)",
    "Enthalpy (kJ/kg)",
    "Entropy (J/g*K)",
    "Cv (J/g*K)",
    "Cp (J/g*K)",
    "Sound Spd. (m/s)",
    "Joule-Thomson (K/MPa)",
    "Viscosity (Pa*s)",
    "Therm. Cond. (W/m*K)",
    "Phase",
]


def _safe(call, *args):
    """Call PropsSI and return NaN instead of raising on a failure."""
    try:
        return PropsSI(*call, *args)
    except Exception:
        return float("nan")


def _sat_row(T):
    """Build one saturated_properties.txt row for a given temperature."""
    # Saturation pressure (Pa) — same for liquid and vapor at this T.
    P_sat = _safe(("P", "T", T, "Q", 0, FLUID))
    P_MPa = P_sat * 1e-6

    row = [T, P_MPa]

    for Q in (0, 1):  # 0 = saturated liquid, 1 = saturated vapor
        rho = _safe(("D", "T", T, "Q", Q, FLUID))
        v = 1.0 / rho if rho and rho > 0 else float("nan")
        u = _safe(("U", "T", T, "Q", Q, FLUID)) * 1e-3  # kJ/kg
        h = _safe(("H", "T", T, "Q", Q, FLUID)) * 1e-3  # kJ/kg
        s = _safe(("S", "T", T, "Q", Q, FLUID)) * 1e-3  # J/g/K (== kJ/kg/K)
        cv = _safe(("O", "T", T, "Q", Q, FLUID)) * 1e-3  # J/g/K
        cp = _safe(("C", "T", T, "Q", Q, FLUID)) * 1e-3  # J/g/K
        a = _safe(("A", "T", T, "Q", Q, FLUID))  # m/s
        jt = _safe(("d(T)/d(P)|H", "T", T, "Q", Q, FLUID)) * 1e6  # K/MPa
        mu = _safe(("V", "T", T, "Q", Q, FLUID))  # Pa·s
        k = _safe(("L", "T", T, "Q", Q, FLUID))  # W/m/K

        if Q == 0:
            sigma = _safe(("surface_tension", "T", T, "Q", 0, FLUID))  # N/m
            beta_l = _safe(("isobaric_expansion_coefficient", "T", T, "Q", 0, FLUID))
            row += [rho, v, u, h, s, cv, cp, a, jt, mu, k, sigma, beta_l]
        else:
            beta_v = _safe(("isobaric_expansion_coefficient", "T", T, "Q", 1, FLUID))
            row += [rho, v, u, h, s, cv, cp, a, jt, mu, k, beta_v]

    return row


def _bar_row(T, P_bar):
    """Build one row for a <P>_bar_properties.txt file."""
    P_Pa = P_bar * 1e5
    P_MPa = P_bar * 0.1

    rho = _safe(("D", "T", T, "P", P_Pa, FLUID))
    v = 1.0 / rho if rho and rho > 0 else float("nan")
    u = _safe(("U", "T", T, "P", P_Pa, FLUID)) * 1e-3
    h = _safe(("H", "T", T, "P", P_Pa, FLUID)) * 1e-3
    s = _safe(("S", "T", T, "P", P_Pa, FLUID)) * 1e-3
    cv = _safe(("O", "T", T, "P", P_Pa, FLUID)) * 1e-3
    cp = _safe(("C", "T", T, "P", P_Pa, FLUID)) * 1e-3
    a = _safe(("A", "T", T, "P", P_Pa, FLUID))
    jt = _safe(("d(T)/d(P)|H", "T", T, "P", P_Pa, FLUID)) * 1e6
    mu = _safe(("V", "T", T, "P", P_Pa, FLUID))
    k = _safe(("L", "T", T, "P", P_Pa, FLUID))

    # Phase label by comparing T to saturation T at this pressure. Above
    # the critical pressure CoolProp can't compute Tsat, so default to
    # vapor.
    try:
        T_sat = PropsSI("T", "P", P_Pa, "Q", 0, FLUID)
        phase = "liquid" if T < T_sat else "vapor"
    except Exception:
        phase = "vapor"

    return [T, P_MPa, rho, v, u, h, s, cv, cp, a, jt, mu, k, phase]


def _format_row(row):
    formatted = []
    for x in row:
        if isinstance(x, str):
            formatted.append(x)
        elif x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
            formatted.append("nan")
        else:
            formatted.append(repr(float(x)))
    return SEP.join(formatted)


def write_saturated(out_dir):
    path = os.path.join(out_dir, "saturated_properties.txt")
    temps = np.linspace(SAT_T_MIN, SAT_T_MAX, SAT_N_POINTS)
    with open(path, "w") as f:
        f.write(SEP.join(SAT_COLUMNS) + "\n")
        for T in temps:
            row = _sat_row(T)
            if any(isinstance(x, float) and (np.isnan(x) or np.isinf(x)) for x in row):
                continue
            f.write(_format_row(row) + "\n")
    return path


def write_bar(out_dir, P_bar):
    if float(P_bar).is_integer():
        name = f"{int(P_bar)}_bar_properties.txt"
    else:
        name = f"{P_bar}_bar_properties.txt"
    path = os.path.join(out_dir, name)
    temps = np.linspace(BAR_T_MIN, BAR_T_MAX, BAR_N_POINTS)
    rows_liquid = []
    rows_vapor = []
    for T in temps:
        row = _bar_row(T, P_bar)
        if any(isinstance(x, float) and (np.isnan(x) or np.isinf(x)) for x in row):
            continue
        if row[-1] == "liquid":
            rows_liquid.append(row)
        else:
            rows_vapor.append(row)
    with open(path, "w") as f:
        f.write(SEP.join(BAR_COLUMNS) + "\n")
        for row in rows_liquid + rows_vapor:
            f.write(_format_row(row) + "\n")
    return path


def main():
    out_dir = os.path.dirname(os.path.abspath(__file__))
    written = []

    print(f"Writing methane saturated properties for T in [{SAT_T_MIN}, {SAT_T_MAX}] K...")
    written.append(write_saturated(out_dir))

    for P_bar in BAR_PRESSURES:
        print(f"  Writing methane properties at {P_bar} bar...")
        written.append(write_bar(out_dir, P_bar))

    # Remove any stale surrogate caches so MethaneProperties retrains.
    for cache in (
        "saturated_property_surrogate_models.pkl",
        "real_gas_property_surrogate_models.pkl",
    ):
        path = os.path.join(out_dir, cache)
        if os.path.exists(path):
            os.remove(path)
            print(f"  Removed stale cache {cache}")

    print("\nWrote files:")
    for p in written:
        print(f"  {os.path.relpath(p, os.path.dirname(out_dir))}")


if __name__ == "__main__":
    main()
