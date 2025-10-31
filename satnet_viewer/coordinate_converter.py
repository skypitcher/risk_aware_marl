from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from satnet_viewer.renderer import SatNetViewer


class CoordinateConverter:
    def __init__(self, renderer: "SatNetViewer"):
        """ renderer is a reference to the SatelliteMapRenderer instance """
        self.renderer = renderer

    def lat_lon_to_screen(self, lat, lon):
        renderer = self.renderer
        norm_x = (lon + 180.0) / 360.0
        norm_y = (90.0 - lat) / 180.0

        norm_x = (norm_x - 0.5) * renderer.zoom_level + 0.5
        norm_y = (norm_y - 0.5) * renderer.zoom_level + 0.5

        screen_x = norm_x * renderer.map_width
        screen_y = norm_y * renderer.map_height

        screen_x += renderer.pan_offset_x
        screen_y += renderer.pan_offset_y
        return screen_x, screen_y

    def screen_to_lat_lon(self, x, y):
        renderer = self.renderer
        x_no_pan = x - renderer.pan_offset_x
        y_no_pan = y - renderer.pan_offset_y

        norm_x = (x_no_pan / renderer.map_width - 0.5) / renderer.zoom_level + 0.5
        norm_y = (y_no_pan / renderer.map_height - 0.5) / renderer.zoom_level + 0.5

        lon = norm_x * 360.0 - 180.0
        lat = 90.0 - norm_y * 180.0
        return lat, lon
