"""Stage 2 - convert sonar pixel positions to latitude/longitude.

Assumptions (standard waterfall image):
  * x axis = across-track, vessel track (nadir) at the image centre, starboard on the right
  * y axis = along-track, row 0 = start of the survey line
  * (lat, lon) = vessel position at row 0
  * heading = direction of travel in degrees clockwise from north
"""
import math
from dataclasses import dataclass

M_PER_DEG_LAT = 111_320.0


@dataclass
class SonarMeta:
    lat: float
    lon: float
    heading_deg: float = 0.0
    swath_m: float = 100.0  # total across-track width covered by the image


def pixel_to_latlon(x, y, meta, img_w):
    scale = meta.swath_m / img_w  # metres per pixel (square pixels assumed)
    across = (x - img_w / 2) * scale
    along = y * scale
    h = math.radians(meta.heading_deg)
    east = along * math.sin(h) + across * math.cos(h)
    north = along * math.cos(h) - across * math.sin(h)
    lat = meta.lat + north / M_PER_DEG_LAT
    lon = meta.lon + east / (M_PER_DEG_LAT * math.cos(math.radians(meta.lat)))
    return lat, lon
