"""M3.2: print the geodesy numbers quoted in the theory note."""

from navigation_control_flight_simulation.geodesy import llh_to_ned, ned_to_llh

origin = (48.1374, 11.5755, 3048.0)               # Munich-like, 10 000 ft

d = 1e-3
north = llh_to_ned((origin[0] + d, origin[1], origin[2]), origin)[0] / d
east = llh_to_ned((origin[0], origin[1] + d, origin[2]), origin)[1] / d
print(f"1 deg latitude  = {north / 1000:.2f} km")
print(f"1 deg longitude = {east / 1000:.2f} km")

lat, lon, h = ned_to_llh([0.0, 8230.0, 0.0], origin)
print(f"8230 m due east at D = 0: lon = {lon:.4f}  ({lon - origin[1]:+.4f} deg)")
print(f"  height above the origin height: {h - origin[2]:.2f} m = {(h - origin[2]) / 0.3048:.1f} ft")