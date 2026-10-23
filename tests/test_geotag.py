from src.geotag import SonarMeta, pixel_to_latlon


def test_centre_pixel_row0_is_start_point():
    lat, lon = pixel_to_latlon(50, 0, SonarMeta(10.0, 20.0, 0, 100), 100)
    assert abs(lat - 10.0) < 1e-9 and abs(lon - 20.0) < 1e-9


def test_heading_north_moves_north_along_track():
    lat, lon = pixel_to_latlon(50, 100, SonarMeta(10.0, 20.0, 0, 100), 100)  # 100 m along track
    assert lat > 10.0 and abs(lon - 20.0) < 1e-9


def test_heading_north_starboard_is_east():
    lat, lon = pixel_to_latlon(100, 0, SonarMeta(10.0, 20.0, 0, 100), 100)  # 50 m starboard
    assert lon > 20.0 and abs(lat - 10.0) < 1e-9


def test_heading_east_starboard_is_south():
    lat, lon = pixel_to_latlon(100, 0, SonarMeta(10.0, 20.0, 90, 100), 100)
    assert lat < 10.0
