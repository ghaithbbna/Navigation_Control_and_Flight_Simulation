"""WGS84 geodesy: geodetic (lat, lon, h) <-> ECEF <-> local NED.

Angles in degrees, lengths in metres. NED = North, East, Down (Down positive).
"""

import numpy as np

A = 6378137.0                    # semi-major axis, m
F = 1.0 / 298.257223563          # flattening
E2 = F * (2.0 - F)               # first eccentricity squared


def llh_to_ecef(lat, lon, h):
    """Geodetic (deg, deg, m) -> ECEF position (m)."""
    p, l = np.radians(lat), np.radians(lon)
    N = A / np.sqrt(1.0 - E2 * np.sin(p) ** 2)
    return np.array([(N + h) * np.cos(p) * np.cos(l),
                     (N + h) * np.cos(p) * np.sin(l),
                     (N * (1.0 - E2) + h) * np.sin(p)])


def ecef_to_llh(P, passes=10):
    """ECEF position (m) -> geodetic (deg, deg, m), by fixed-point iteration."""
    x, y, z = P
    r = np.hypot(x, y)
    lat = np.arctan2(z, r * (1.0 - E2))                 # first guess
    for _ in range(passes):
        N = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
        h = r / np.cos(lat) - N
        lat = np.arctan2(z, r * (1.0 - E2 * N / (N + h)))
    N = A / np.sqrt(1.0 - E2 * np.sin(lat) ** 2)
    h = r / np.cos(lat) - N
    return np.degrees(lat), np.degrees(np.arctan2(y, x)), h


def ned_rotation(lat, lon):
    """Rotation R with [N, E, D] = R @ (P_ecef - P0_ecef) for an origin at (lat, lon)."""
    p, l = np.radians(lat), np.radians(lon)
    return np.array([[-np.sin(p) * np.cos(l), -np.sin(p) * np.sin(l),  np.cos(p)],
                     [-np.sin(l),              np.cos(l),              0.0],
                     [-np.cos(p) * np.cos(l), -np.cos(p) * np.sin(l), -np.sin(p)]])


def llh_to_ned(llh, origin):
    """Geodetic point -> [N, E, D] (m) relative to the origin (lat, lon, h)."""
    R = ned_rotation(origin[0], origin[1])
    return R @ (llh_to_ecef(*llh) - llh_to_ecef(*origin))


def ned_to_llh(ned, origin):
    """[N, E, D] (m) relative to the origin -> geodetic (lat, lon, h)."""
    R = ned_rotation(origin[0], origin[1])
    return ecef_to_llh(llh_to_ecef(*origin) + R.T @ np.asarray(ned, dtype=float))