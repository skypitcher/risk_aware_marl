import math
from typing import Union

import astropy.units as u
import numpy as np
from astropy.coordinates import EarthLocation

GM_EARTH = 3.9857606e14  # Earth gravitational parameter (GM) in m^3/s^2
EARTH_R_KM = 6371.0  # Earth radius in kilometers
LIGHT_SPEED_MS = 299.792  # Speed of light in km/ms

def distance_between(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points in 3D space.

    Args:
        p1: First point as a 3D vector
        p2: Second point as a 3D vector

    Returns:
        Distance in kilometers
    """
    return np.sqrt(np.sum(np.square(p1 - p2)))


def calculate_orbital_period(altitude: float) -> float:
    """
    Calculate the orbital period using Kepler's third law for a circular orbit.

    Args:
        altitude: Satellite orbital altitude in kilometers

    Returns:
        Orbital period in seconds
    """
    # Calculate semi-major axis in meters
    a_m = (EARTH_R_KM + altitude) * 1000

    # Calculate the orbital period using Kepler's third law
    T = 2 * math.pi * math.sqrt(math.pow(a_m, 3) / GM_EARTH)

    return T


def calculate_orbital_velocity(altitude: float) -> float:
    """
    Calculate the orbital velocity for a circular orbit.

    Args:
        altitude: Satellite orbital altitude in kilometers

    Returns:
        Orbital velocity in kilometers per second
    """

    # Calculate semi-major axis in meters
    a_m = (EARTH_R_KM + altitude) * 1000

    # Calculate the orbital velocity
    v = math.sqrt(GM_EARTH / a_m)

    # Convert to km/s
    v_kms = v / 1000

    return v_kms


def calculate_elevation_angle(ground_position: np.ndarray, satellite_position: np.ndarray) -> float:
    """
    Calculate an elevation angle from ground station to satellite.

    Args:
        ground_position: Ground station position as a 3D vector
        satellite_position: Satellite position as a 3D vector

    Returns:
        Elevation angle in degrees
    """
    # Vector from ground station to satellite
    gs_to_sat = satellite_position - ground_position

    # Normalized vector from Earth center to ground station (local zenith direction)
    zenith_direction = ground_position / np.linalg.norm(ground_position)

    # Normalize the ground-to-satellite vector
    gs_to_sat_norm = gs_to_sat / np.linalg.norm(gs_to_sat)

    # Compute dot product to get cosine of the complement of an elevation angle
    cos_angle = np.dot(zenith_direction, gs_to_sat_norm)

    # Convert to elevation angle in degrees (clamp to avoid numerical issues)
    elevation_deg = 90 - np.degrees(np.arccos(max(-1.0, min(1.0, cos_angle))))

    return elevation_deg


def calculate_slant_range(elevation_angle_deg: float, sat_altitude_km: float) -> float:
    """
    Calculate the slant range (distance) from a ground user to a satellite with the given elevation angle and altitude.

    Args:
        elevation_angle_deg: Elevation angle in degrees (0° to 90°)
        sat_altitude_km: Satellite altitude above Earth's surface in kilometers

    Returns:
        Slant range in kilometers
    """
    R_e = EARTH_R_KM

    # Convert an elevation angle from degrees to radians
    theta_rad = math.radians(elevation_angle_deg)

    # Calculate sine of the elevation angle
    sin_theta = math.sin(theta_rad)

    # Compute the term under the square root
    term_under_sqrt = (R_e**2 * sin_theta**2) + (2 * R_e * sat_altitude_km) + (sat_altitude_km**2)

    # Calculate slant range using the derived formula
    d = -R_e * sin_theta + math.sqrt(term_under_sqrt)

    return d


def calculate_maximum_inter_satellite_range(altitude_km: float) -> float:
    """
    Calculate the maximum visible range between two satellites at the same altitude.

    Args:
        altitude_km: Altitude of the satellites above Earth's surface in kilometers

    Returns:
        Maximum visible range in kilometers
    """
    # Earth's mean radius in kilometers
    R_e = 6371.0

    # Calculate the maximum visible range
    d_max = 2 * math.sqrt(2 * R_e * altitude_km + altitude_km**2)

    return d_max


def get_position_eci(
    altitude: float, raan: float, inclination: float, true_anomaly: float, timestamp: Union[np.ndarray, float]
) -> np.ndarray:
    """
    Calculates the position of a satellite in the Earth-Centered Inertial (ECI) coordinate system.

    Args:
        altitude: Satellite altitude above Earth's surface in kilometers
        raan: Right ascension of ascending node in degree
        inclination: Orbital inclination in degrees
        true_anomaly: True anomaly in degree
        timestamp: Time offset in milliseconds from reference epoch

    Returns:
        Cartesian coordinates (x, y, z) in the ECI frame
    """
    orbit_cycle = calculate_orbital_period(altitude)
    a_km = EARTH_R_KM + altitude
    theta = (true_anomaly + (360 / orbit_cycle) * timestamp / 1000) % 360  # true anomaly of each time point
    theta_rad = np.deg2rad(theta)
    raan_rad = np.deg2rad(raan)
    inc_rad = np.deg2rad(inclination)

    x_eci = a_km * (np.cos(raan_rad) * np.cos(theta_rad) - np.sin(raan_rad) * np.sin(theta_rad) * np.cos(inc_rad))
    y_eci = a_km * (np.sin(raan_rad) * np.cos(theta_rad) + np.cos(raan_rad) * np.sin(theta_rad) * np.cos(inc_rad))
    z_eci = a_km * np.sin(theta_rad) * np.sin(inc_rad)

    if np.size(timestamp) <= 1:
        return np.array([x_eci, y_eci, z_eci])
    else:
        return np.column_stack([x_eci, y_eci, z_eci])


def get_position_ecef(
    altitude: float, raan: float, inclination: float, true_anomaly: float, timestamp: Union[np.ndarray, float]
) -> np.ndarray:
    """
    Calculates the position of a satellite in the Earth-Centered Earth-Fixed (ECEF) coordinate system.

    Args:
        altitude: Satellite altitude above Earth's surface in kilometers
        raan: Right ascension of ascending node in degree
        inclination: Orbital inclination in degrees
        true_anomaly: True anomaly in degree
        timestamp: Time offset in milliseconds from reference epoch

    Returns:
        Cartesian coordinates (x, y, z) in the ECEF frame
    """
    pos_eci = get_position_eci(altitude, raan, inclination, true_anomaly, timestamp)

    # Set Earth rotation parameters
    omega_earth = 7.2921150e-5  # Earth rotation rate (rad/s)
    theta_earth = omega_earth * timestamp / 1000  # Accumulated rotation angle

    if np.size(timestamp) <= 1:
        # Rotation matrix from ECI to ECEF (about Z-axis)
        x_eci = pos_eci[0]
        y_eci = pos_eci[1]
        z_eci = pos_eci[2]

        x_ecef = x_eci * np.cos(theta_earth) + y_eci * np.sin(theta_earth)
        y_ecef = -x_eci * np.sin(theta_earth) + y_eci * np.cos(theta_earth)
        z_ecef = z_eci
        return np.array([x_ecef, y_ecef, z_ecef])
    else:
        x_eci = pos_eci[..., 0]
        y_eci = pos_eci[..., 1]
        z_eci = pos_eci[..., 2]

        x_ecef = x_eci * np.cos(theta_earth) + y_eci * np.sin(theta_earth)
        y_ecef = -x_eci * np.sin(theta_earth) + y_eci * np.cos(theta_earth)
        z_ecef = z_eci
        return np.column_stack([x_ecef, y_ecef, z_ecef])


def geo_to_ecef_position(lat: float, lon: float, alt: float = 0.0) -> np.ndarray:
    """
    Convert geodetic coordinates (latitude, longitude, altitude) to ECEF position.

    Args:
        lat: Latitude in degrees
        lon: Longitude in degrees
        alt: Altitude above Earth's surface in kilometers

    Returns:
        3D position vector in ECEF (Earth-Centered, Earth-Fixed) coordinates
    """
    # Convert from degrees to radians
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)

    # Calculate position
    R = EARTH_R_KM + alt
    x = R * np.cos(lat_rad) * np.cos(lon_rad)
    y = R * np.cos(lat_rad) * np.sin(lon_rad)
    z = R * np.sin(lat_rad)

    return np.array([x, y, z])


def get_projected_position(x: np.ndarray | float, y: np.ndarray | float, z: np.ndarray | float):
    """
    Calculate the projected position (latitude, longitude) of a point (ECEF coordinates)
    on the surface of the Earth.

    Args:
        x: ECEF x-coordinate(s) in kilometers.
        y: ECEF y-coordinate(s) in kilometers.
        z: ECEF z-coordinate(s) in kilometers.

    Returns:
        Tuple[np.ndarray | float, np.ndarray | float]: Longitude(x) and Latitude(y) in degrees.
    """
    location = EarthLocation.from_geocentric(x * u.km, y * u.km, z * u.km)

    # Return longitude(x) and latitude(y) in degrees
    return location.lon.deg, location.lat.deg


def calculate_delay(pos1: np.ndarray, pos2: np.ndarray) -> float:
    """Calculate the propagation delay between two nodes based on their positions."""
    distance = float(np.linalg.norm(pos1 - pos2))
    delay = float(distance / LIGHT_SPEED_MS)
    return delay


def normalize_longitude(lon):
    """Normalize longitude to [-180, 180] range"""
    while lon > 180:
        lon -= 360
    while lon < -180:
        lon += 360
    return lon


def angular_distance_longitude(lon1, lon2):
    """
    Calculate the shortest angular distance between two longitudes
    handling wrap-around at ±180°
    
    Args:
        lon1, lon2: Longitude values in degrees [-180, 180]
    
    Returns:
        Shortest angular distance in degrees [0, 180]
    """
    # Normalize both longitudes
    lon1 = normalize_longitude(lon1)
    lon2 = normalize_longitude(lon2)
    
    # Calculate direct difference
    diff = abs(lon2 - lon1)
    
    # Check if going the other way around is shorter
    # The other way would be 360 - diff
    return min(diff, 360 - diff)


def angular_distance_latitude(lat1, lat2):
    """
    Calculate angular distance between two latitudes
    No wrap-around needed since latitude is bounded [-90, 90]
    
    Args:
        lat1, lat2: Latitude values in degrees [-90, 90]
    
    Returns:
        Angular distance in degrees [0, 180]
    """
    return abs(lat2 - lat1)


def separate_angular_distances(lon1, lat1, lon2, lat2):
    """
    Calculate separate angular distances in longitude and latitude directions
    
    Returns:
        tuple: (longitude_distance, latitude_distance) in degrees
    """
    lon_dist = angular_distance_longitude(lon1, lon2)
    lat_dist = angular_distance_latitude(lat1, lat2)
    return lon_dist, lat_dist


def great_circle_distance(lon1, lat1, lon2, lat2):
    """
    Calculate the shortest angular distance between two points on a sphere
    using the great circle distance formula (haversine formula)
    
    Args:
        lon1, lat1: First point coordinates in degrees
        lon2, lat2: Second point coordinates in degrees
    
    Returns:
        Great circle distance in degrees [0, 180]
    """
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = math.sin(dlat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Convert back to degrees
    return math.degrees(c)
