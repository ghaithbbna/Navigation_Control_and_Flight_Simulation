"""Unit conversions: the ONLY place where ft <-> m and deg <-> rad happen.

Convention used across the project
----------------------------------
* The longitudinal plant (plant.py) works in feet, feet/second and radians,
  because that is how the model was identified in the course.
* Geodesy, missions and FlightGear work in metres and radians.
* Conversion happens only at the boundary, through the functions below.
"""

import numpy as np

M_PER_FT = 0.3048            # exact, by international definition (1959)
FT_PER_M = 1.0 / M_PER_FT

G0_SI = 9.80665              # standard gravity, m/s^2 (exact, by definition)
G0_FT = G0_SI / M_PER_FT     # = 32.17405 ft/s^2


def ft2m(x):
    """Feet -> metres (works on scalars and arrays)."""
    return np.asarray(x, dtype=float) * M_PER_FT


def m2ft(x):
    """Metres -> feet."""
    return np.asarray(x, dtype=float) * FT_PER_M


def deg2rad(x):
    return np.deg2rad(x)


def rad2deg(x):
    return np.rad2deg(x)
