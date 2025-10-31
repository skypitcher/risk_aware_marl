import json
import os
import time
from datetime import timedelta

import geopandas as gpd
import glfw
import imgui
import imgui.integrations.glfw
import numpy as np
from OpenGL.GL import *
from PIL import Image
from shapely.geometry import Polygon, MultiPolygon

from sat_net import SatelliteNetwork, Satellite, GroundStation, ms2str
from satnet_viewer.coordinate_converter import CoordinateConverter
from satnet_viewer.texture_manager import TextureManager


# Helper function to convert RGBA tuples to ImGui color format
def rgba_to_imgui_color(r, g, b, a=1.0):
    """Convert RGBA values (0-1 range) to ImGui color format (32-bit packed integer)"""
    r = int(r * 255) & 0xFF
    g = int(g * 255) & 0xFF
    b = int(b * 255) & 0xFF
    a = int(a * 255) & 0xFF
    return (a << 24) | (b << 16) | (g << 8) | r


def _get_lon_lat(pos: np.ndarray) -> tuple[float, float]:
    norm_pos = pos / np.linalg.norm(pos)
    lat = np.degrees(np.arcsin(norm_pos[2]))
    lon = np.degrees(np.arctan2(norm_pos[1], norm_pos[0]))
    return lon, lat


def _normalize_longitude(lon):  # Helper for normalization
    return (lon + 180) % 360 - 180


class SatNetViewer:
    SETTINGS_FILENAME = "viewer_settings.json"
    _instance = None  # Class attribute to hold the instance for the callback

    def __init__(self, config_name, network: SatelliteNetwork, time_step=timedelta(seconds=1), width=1800, height=900):
        self.config_name = config_name
        self.network = network
        self.time_step = time_step
        self.time_step_ms = int(time_step.total_seconds() * 1000)
        self.current_time = 0
        self.width = width
        self.height = height

        # Define project root and settings file path
        # Assumes renderer.py is in net_viewer, so project root is one level up.
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.settings_file_path = os.path.join(self.project_root, self.SETTINGS_FILENAME)

        self.assets_dir = os.path.join(self.project_root, "assets")

        self._init_glfw()

        imgui.create_context()
        self.io = imgui.get_io()

        if hasattr(imgui, "get_version"):
            print(f"ImGui Version: {imgui.get_version()}")
        else:
            print("ImGui version could not be retrieved via get_version().")

        self.impl = imgui.integrations.glfw.GlfwRenderer(self.window)

        self.map_width = int(width * 0.75)
        self.map_height = height
        # self.view_mode = "2d_fixed" # Removed

        self.zoom_level = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 10.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.is_dragging = False
        self.drag_start_x = 0.0
        self.drag_start_y = 0.0
        self.last_mouse_x = 0.0
        self.last_mouse_y = 0.0

        self.colors = {
            "background": (0.0, 0.0, 0.2, 1.0),  # Sea blue
            "grid_major": (1.0, 1.0, 1.0, 0.5),
            "grid_minor": (0.7, 0.7, 0.7, 0.3),
            "equator": (0.7, 0.7, 0.9, 0.5),
            "sat": (1.0, 0.0, 0.0, 1.0),
            "sat_outline": (1.0, 1.0, 1.0, 1.0),
            "sat_selected": (0.0, 0.0, 1.0, 1.0),
            "gs": (0.2, 1.0, 0.2, 1.0),
            "gs_outline": (0.1, 0.1, 0.1, 1.0),
            "link_isl": (0.1, 0.8, 0.8, 1.0),
            "link_gsl": (0.2, 1.0, 0.2, 1.0),
            "link_idl": (1.0, 0.5, 0.0, 1.0),
            "text": (1.0, 1.0, 1.0, 1.0),
            "panel": (0.1, 0.1, 0.2, 0.8),
            "gs_tracking_circle": (0.3, 0.7, 1.0, 0.9),  # Light blue, semi-transparent
            "orbit_trajectory": (0.5, 0.5, 0.5, 0.7),  # Grey, semi-transparent
            "link_isl_hovered": (1.0, 1.0, 0.0, 1.0),  # Yellow, for hovered ISLs
        }

        self.show_grid = True
        self.show_satellites = True
        self.show_ground_stations = True
        self.show_isls = True
        self.show_gsls = True
        self.show_sat_labels = True
        self.show_gs_labels = True

        self.show_gs_tracking_circles = False  # New setting
        self.show_orbit_trajectories = True  # New setting for orbit trajectories
        self.satellite_size = 6.0
        self.ground_station_size = 6.0
        self.link_thickness = 1.0
        self.animation_speed = 1.0
        self.animation_speed_max = 200.0

        self.satellite_visibility = {sat_id: True for sat_id in network.satellites}
        self.gsl_draw_visibility = {}
        for link_id, link in network.links.items():
            if (isinstance(link.source, Satellite) and isinstance(link.sink, GroundStation)) or (
                    isinstance(link.source, GroundStation) and isinstance(link.sink, Satellite)
            ):
                self.gsl_draw_visibility[link_id] = True

        self.hovered_satellite_id = None  # For hovered satellite trajectory
        self.render_fps = 0.0
        self.last_frame_time = time.time()
        self.frame_count = 0
        self.last_fps_update = time.time()

        self.paused = False

        self.coord_converter = CoordinateConverter(self)

        self.texture_manager = TextureManager(assets_dir=self.assets_dir)

        self.land_geometries = None
        self._load_land_geometries()

        self.land_color = rgba_to_imgui_color(1.0, 1.0, 0, 1.0)

        self._load_settings()  # Load settings after defaults are set

    @staticmethod
    def _framebuffer_size_callback(_window, width, height):
        if SatNetViewer._instance:  # Check if instance exists
            renderer_instance = SatNetViewer._instance
            renderer_instance.width = width
            renderer_instance.height = height
            # OpenGL viewport update
            glViewport(0, 0, width, height)
            # Mark that debug message should show resize
            renderer_instance.debug_message = f"Window Resized: {width}x{height}"
            renderer_instance.debug_time = time.time()

    def _init_glfw(self):
        if not glfw.init():
            raise RuntimeError("Could not initialize GLFW")
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, GL_TRUE)
        self.window = glfw.create_window(self.width, self.height, "Satellite Network Simulation", None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Could not create GLFW window")
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)
        SatNetViewer._instance = self  # Store the instance on the class
        glfw.set_framebuffer_size_callback(self.window, self._framebuffer_size_callback)
        self.debug_message = ""
        self.debug_time = time.time()
        self.last_pan_debug = time.time()

    def _draw_grid(self):
        draw_list = imgui.get_window_draw_list()
        grid_color = rgba_to_imgui_color(*self.colors["grid_major"])
        text_color = rgba_to_imgui_color(*self.colors["text"])
        for lat_deg in range(-90, 91, 15):
            x1, y1 = self.coord_converter.lat_lon_to_screen(lat_deg, -180)
            x2, y2 = self.coord_converter.lat_lon_to_screen(lat_deg, 180)
            draw_list.add_line(x1, y1, x2, y2, grid_color, 1.0)
            if lat_deg % 30 == 0 and lat_deg != 0:
                label = f"{abs(lat_deg)}°{'N' if lat_deg > 0 else 'S'}"
                draw_list.add_text(x2 - 30, y1, text_color, label)
        for lon_deg in range(-180, 181, 15):
            x1, y1 = self.coord_converter.lat_lon_to_screen(-90, lon_deg)
            x2, y2 = self.coord_converter.lat_lon_to_screen(90, lon_deg)
            draw_list.add_line(x1, y1, x2, y2, grid_color, 1.0)
            if lon_deg % 30 == 0 and lon_deg != 0:
                label = f"{abs(lon_deg)}°{'E' if lon_deg > 0 else 'W'}"
                draw_list.add_text(x1, y1 - 15, text_color, label)
        equator_color = rgba_to_imgui_color(*self.colors["equator"])
        x1, y1 = self.coord_converter.lat_lon_to_screen(0, -180)
        x2, y2 = self.coord_converter.lat_lon_to_screen(0, 180)
        draw_list.add_line(x1, y1, x2, y2, equator_color, 1.5)
        x1, y1 = self.coord_converter.lat_lon_to_screen(-90, 0)
        x2, y2 = self.coord_converter.lat_lon_to_screen(90, 0)
        draw_list.add_line(x1, y1, x2, y2, equator_color, 1.5)

    def _draw_satellites(self):
        draw_list = imgui.get_window_draw_list()
        satellite_color = rgba_to_imgui_color(*self.colors["sat"])
        satellite_outline = rgba_to_imgui_color(*self.colors["sat_outline"])
        text_color = rgba_to_imgui_color(1.0, 0.647, 0.0, 1.0)
        for sat_id, sat in self.network.satellites.items():
            if not self.satellite_visibility.get(sat_id, True):
                continue
            lon, lat = _get_lon_lat(sat.position)
            x, y = self.coord_converter.lat_lon_to_screen(lat, lon)
            outline_size = self.satellite_size + 1
            draw_list.add_circle_filled(x, y, outline_size, satellite_outline)
            draw_list.add_circle_filled(x, y, self.satellite_size, satellite_color)
            if self.show_sat_labels:
                draw_list.add_text(x + 5, y + 5, text_color, f"Sat-{sat_id}")

    def _draw_ground_stations(self):
        draw_list = imgui.get_window_draw_list()
        gs_color = rgba_to_imgui_color(*self.colors["gs"])
        gs_outline = rgba_to_imgui_color(*self.colors["gs_outline"])
        text_color = rgba_to_imgui_color(1.0, 0.647, 0.0, 1.0)
        for gs_id, gs in self.network.ground_stations.items():
            x, y = self.coord_converter.lat_lon_to_screen(gs.latitude, gs.longitude)
            size = self.ground_station_size
            outline_size = size + 1.5
            draw_list.add_triangle_filled(
                x, y - outline_size, x - outline_size, y + outline_size, x + outline_size, y + outline_size, gs_outline
            )
            draw_list.add_triangle_filled(x, y - size, x - size, y + size, x + size, y + size, gs_color)
            if self.show_gs_labels:
                draw_list.add_text(x + size + 2, y, text_color, f"{gs.name} (GS-{gs_id})")

    def _draw_links(self):
        draw_list = imgui.get_window_draw_list()
        isl_color = rgba_to_imgui_color(*self.colors["link_isl"])
        gsl_color = rgba_to_imgui_color(*self.colors["link_gsl"])
        isl_hovered_color = rgba_to_imgui_color(*self.colors["link_isl_hovered"])

        for link_id, link in self.network.links.items():
            if not link.is_connected:
                continue

            src_lon, src_lat = _get_lon_lat(link.source.position)
            dst_lon, dst_lat = _get_lon_lat(link.sink.position)

            is_isl = isinstance(link.source, Satellite) and isinstance(link.sink, Satellite)
            is_gsl = not is_isl

            current_color = isl_color  # Default
            draw_this_link = False

            if is_isl:
                is_hovered_link = False
                if self.hovered_satellite_id is not None and \
                        (link.source.id == self.hovered_satellite_id or link.sink.id == self.hovered_satellite_id):
                    is_hovered_link = True

                if is_hovered_link:
                    current_color = isl_hovered_color
                    draw_this_link = True  # Always draw ISLs for hovered satellite
                elif self.show_isls:
                    draw_this_link = True
                else:
                    # an ISL, not hovered, and show_isls is false
                    continue  # Skip this link

                # Visibility check for non-hovered ISLs (already handled by continue above if show_isls is false)
                # For hovered ISLs, we draw them regardless of individual satellite visibility settings for simplicity,
                # or one might argue they should only be drawn if both connected sats are visible.
                # For now, let's draw if hovered. If show_isls is true for non-hovered, then check sat visibility.
                if not is_hovered_link and self.show_isls:  # This implies it's a normal ISL
                    source_sat_id = link.source.id
                    sink_sat_id = link.sink.id
                    if not (self.satellite_visibility.get(source_sat_id, True) and self.satellite_visibility.get(
                            sink_sat_id, True)):
                        continue  # Skip if participating satellites are not visible

            elif is_gsl:
                if self.show_gsls:
                    gsl_is_individually_visible = self.gsl_draw_visibility.get(link_id, True)
                    if gsl_is_individually_visible:
                        draw_this_link = True
                        current_color = gsl_color
                    else:
                        continue  # GSL is individually hidden
                else:
                    continue  # GSLs are globally hidden

            if not draw_this_link:
                continue

            # if not (src_lat is not None and src_lon is not None and dst_lat is not None and dst_lon is not None):
            #     continue # This check was originally here, but lat/lon are derived just before this block

            # At this point, current_color is set and we've decided to draw the link.
            thickness = self.link_thickness
            self._draw_line_IDL(draw_list, src_lat, src_lon, dst_lat, dst_lon, current_color, thickness, truncate=True)

    def _draw_legend(self):
        draw_list = imgui.get_window_draw_list()
        text_color = rgba_to_imgui_color(1.0, 1.0, 1.0, 1.0)
        gs_color = rgba_to_imgui_color(*self.colors["gs"])
        sat_color = rgba_to_imgui_color(*self.colors["sat"])
        legend_bg_color = rgba_to_imgui_color(*self.colors["panel"])
        legend_width, legend_height, margin = 200, 120, 10
        x, y = self.map_width - legend_width - margin, margin
        draw_list.add_rect_filled(x, y, x + legend_width, y + legend_height, legend_bg_color)
        draw_list.add_rect(x, y, x + legend_width, y + legend_height, text_color)
        draw_list.add_text(x + 10, y + 10, text_color, "Legend")

        icon_size, item_height = 10, 20
        item_y = y + 30
        draw_list.add_circle_filled(x + 20, item_y + icon_size / 2, icon_size / 2, sat_color)
        draw_list.add_text(x + 40, item_y, text_color, "Satellite")

        item_y += item_height
        draw_list.add_triangle_filled(x + 20, item_y, x + 15, item_y + icon_size, x + 25, item_y + icon_size, gs_color)
        draw_list.add_text(x + 40, item_y, text_color, "Ground Station")

        item_y += item_height
        link_color = rgba_to_imgui_color(*self.colors["link_isl"])
        draw_list.add_line(x + 10, item_y + icon_size / 2, x + 30, item_y + icon_size / 2, link_color, 2.0)
        draw_list.add_text(x + 40, item_y, text_color, "ISL")

        item_y += item_height
        link_color = rgba_to_imgui_color(*self.colors["link_gsl"])
        draw_list.add_line(x + 10, item_y + icon_size / 2, x + 30, item_y + icon_size / 2, link_color, 2.0)
        draw_list.add_text(x + 40, item_y, text_color, "GSL")

        fps_text = f"FPS: {self.render_fps:.1f}"
        draw_list.add_text(x, self.map_height - 20, text_color, fps_text)

    def _update_fps(self):
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            self.render_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

    def _draw_debug_info(self):
        draw_list = imgui.get_window_draw_list()
        text_color = rgba_to_imgui_color(1, 1, 1, 1)
        bg_color = rgba_to_imgui_color(0, 0, 0, 0.7)
        highlight_color = rgba_to_imgui_color(1, 0.9, 0.2, 1)
        debug_panel_x, debug_panel_y, padding, panel_width, panel_height = 10, 10, 5, 450, 120
        draw_list.add_rect_filled(
            debug_panel_x - padding, debug_panel_y - padding, debug_panel_x + panel_width, debug_panel_y + panel_height,
            bg_color
        )
        draw_list.add_rect(
            debug_panel_x - padding,
            debug_panel_y - padding,
            debug_panel_x + panel_width,
            debug_panel_y + panel_height,
            rgba_to_imgui_color(0.5, 0.5, 0.5, 1),
            1,
        )
        is_recent_input = time.time() - self.debug_time < 2.0
        input_color = highlight_color if is_recent_input else text_color
        draw_list.add_text(debug_panel_x, debug_panel_y, input_color, f"Input: {self.debug_message}")
        draw_list.add_text(debug_panel_x, debug_panel_y + 20, text_color, f"View Mode: Interactive")
        current_y_offset = debug_panel_y + 40
        draw_list.add_text(debug_panel_x, current_y_offset, text_color, f"Zoom: {self.zoom_level:.2f}x")
        current_y_offset += 20
        pan_color = (
            highlight_color
            if is_recent_input and ("Pan" in self.debug_message or "Panning" in self.debug_message)
            else text_color
        )
        draw_list.add_text(
            debug_panel_x, current_y_offset, pan_color, f"Pan: ({self.pan_offset_x:.1f}, {self.pan_offset_y:.1f})"
        )
        current_y_offset += 20
        controls_text = "Controls: Mouse wheel to zoom, Right-click drag to pan"
        mouse_x, mouse_y = glfw.get_cursor_pos(self.window)
        if mouse_x < self.map_width:
            lat, lon = self.coord_converter.screen_to_lat_lon(mouse_x, mouse_y)
            draw_list.add_text(debug_panel_x, current_y_offset, text_color,
                               f"Cursor: {lat:.1f}°N/S, {abs(lon):.1f}°E/W")
        else:
            draw_list.add_text(debug_panel_x, current_y_offset, text_color, "Cursor: Outside map")
        current_y_offset += 20
        draw_list.add_text(debug_panel_x, current_y_offset, text_color, controls_text)

    def _handle_imgui_scroll(self):
        wheel_delta = self.io.mouse_wheel
        if abs(wheel_delta) < 0.01:
            return
        mouse_x, mouse_y = self.io.mouse_pos.x, self.io.mouse_pos.y
        if mouse_x >= self.map_width:
            return
        self.debug_message = f"Scroll: {wheel_delta:.2f} at ({mouse_x:.1f}, {mouse_y:.1f}) - Zooming"
        self.debug_time = time.time()
        lat_at_mouse, lon_at_mouse = self.coord_converter.screen_to_lat_lon(mouse_x, mouse_y)
        old_zoom = self.zoom_level
        zoom_factor = 1.1 if wheel_delta > 0 else (1.0 / 1.1)
        self.zoom_level = max(self.min_zoom, min(self.max_zoom, old_zoom * zoom_factor))
        if abs(old_zoom - self.zoom_level) < 0.001:
            return
        screen_x_new_zoom, screen_y_new_zoom = self.coord_converter.lat_lon_to_screen(lat_at_mouse, lon_at_mouse)
        self.pan_offset_x += mouse_x - screen_x_new_zoom
        self.pan_offset_y += mouse_y - screen_y_new_zoom
        self._apply_pan_restrictions()

    def _handle_imgui_pan(self):
        is_right_mouse_down = self.io.mouse_down[1]
        mouse_x, mouse_y = self.io.mouse_pos.x, self.io.mouse_pos.y
        if is_right_mouse_down and not self.is_dragging:
            if mouse_x < self.map_width:
                self.is_dragging = True
                self.drag_start_x, self.drag_start_y = mouse_x, mouse_y
                self.last_mouse_x, self.last_mouse_y = mouse_x, mouse_y
                self.debug_message = f"Pan Start: ({mouse_x:.1f}, {mouse_y:.1f})"
                self.debug_time = time.time()
        elif self.is_dragging:
            if not is_right_mouse_down:
                self.is_dragging = False
                self.debug_message = f"Pan End: ({mouse_x:.1f}, {mouse_y:.1f})"
                self.debug_time = time.time()
            else:
                if mouse_x < self.map_width:
                    dx, dy = mouse_x - self.last_mouse_x, mouse_y - self.last_mouse_y
                    self.pan_offset_x += dx
                    self.pan_offset_y += dy
                    self.debug_message = f"Panning: dx={dx:.1f}, dy={dy:.1f}"
                    self.debug_time = time.time()
                self.last_mouse_x, self.last_mouse_y = mouse_x, mouse_y
        self._apply_pan_restrictions()

    def _apply_pan_restrictions(self):
        """Clamps pan_offset_x and pan_offset_y to keep view within world bounds."""

        if self.zoom_level <= 1.0:
            # If zoomed out (or at 1x zoom), content is smaller or equal to view.
            # Center the content, so no panning.
            self.pan_offset_x = 0.0
            self.pan_offset_y = 0.0
        else:
            # Content is larger than the view, allow panning within bounds.

            # Vertical pan restriction (latitude: 90N to 90S)
            # max_permissible_pan_y allows 90N (normalized y=0) to be at screen y=0
            max_permissible_pan_y = 0.5 * self.map_height * (self.zoom_level - 1.0)
            # min_permissible_pan_y allows -90S (normalized y=1) to be at screen y=map_height
            min_permissible_pan_y = 0.5 * self.map_height * (1.0 - self.zoom_level)
            self.pan_offset_y = max(min_permissible_pan_y, min(self.pan_offset_y, max_permissible_pan_y))

            # Horizontal pan restriction (longitude: -180W to 180E)
            # max_permissible_pan_x allows -180W (normalized x=0) to be at screen x=0
            max_permissible_pan_x = 0.5 * self.map_width * (self.zoom_level - 1.0)
            # min_permissible_pan_x allows 180E (normalized x=1) to be at screen x=map_width
            min_permissible_pan_x = 0.5 * self.map_width * (1.0 - self.zoom_level)
            self.pan_offset_x = max(min_permissible_pan_x, min(self.pan_offset_x, max_permissible_pan_x))

    def _draw_polygon_ring(self, draw_list, ring, color, thickness):
        """Draws a single polygon ring (exterior or interior), handling dateline crossing."""
        if ring is None or ring.is_empty:
            return
        coords = list(ring.coords)  # List of (lon, lat) tuples
        if len(coords) < 3:  # Need at least 3 points for a ring/polygon
            return

        # Iterate through line segments, including the closing one
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]
            self._draw_line_IDL(draw_list, lat1, lon1, lat2, lon2, color, thickness)

    def _draw_land_mass(self):
        """Draws landmass using loaded vector data."""
        if not self.land_geometries or self.coord_converter is None:  # Check converter too
            return

        draw_list = imgui.get_window_draw_list()
        color = self.land_color
        thickness = 1.0
        for geom in self.land_geometries:
            if geom is None or geom.is_empty:
                continue

            if isinstance(geom, Polygon):
                # Draw exterior
                self._draw_polygon_ring(draw_list, geom.exterior, color, thickness)
            elif isinstance(geom, MultiPolygon):
                for polygon in geom.geoms:  # Iterate through polygons in the MultiPolygon
                    if polygon is None or polygon.is_empty:
                        continue
                    self._draw_polygon_ring(draw_list, polygon.exterior, color, thickness)
            else:
                pass  # Handle other geometry types if necessary, though land data is usually Polygon/MultiPolygon

    def _draw_map_panel(self):
        map_panel_width = self.width - int(self.width * 0.25)
        imgui.set_next_window_position(0, 0)
        imgui.set_next_window_size(map_panel_width, self.height)
        imgui.begin(
            "Satellite Map",
            False,
            imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_SCROLLBAR,
        )
        self.map_width, self.map_height = map_panel_width, self.height

        # Draw a basic background color first if outlines are shown
        bg_color = rgba_to_imgui_color(*self.colors["background"])
        draw_list = imgui.get_window_draw_list()  # Get a draw list if needed here
        draw_list.add_rect_filled(0, 0, self.map_width, self.map_height, bg_color)

        # Draw landmass
        self._draw_land_mass()

        if self.show_grid:
            self._draw_grid()

        # Draw orbit trajectories after grid and land, before satellites
        if self.show_orbit_trajectories:
            self._draw_orbit_trajectory_of_sat()

        if self.show_satellites:
            self._draw_satellites()

        if self.show_ground_stations:
            self._draw_ground_stations()

        if self.show_gs_tracking_circles:
            self._draw_gs_tracking_circles()

        self._draw_links()

        self._draw_legend()

        self._draw_debug_info()

        imgui.end()

    def _draw_settings_panel(self):
        settings_width = int(self.width * 0.25)
        map_panel_width = self.width - settings_width
        imgui.set_next_window_position(map_panel_width, 0)
        imgui.set_next_window_size(settings_width, self.height)
        imgui.begin("Settings Panel", False, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_COLLAPSE)
        if imgui.begin_tab_bar("SettingsTabs"):
            if imgui.begin_tab_item("General Settings")[0]:
                imgui.text("Network Info")
                imgui.indent()
                imgui.text(f"Cfg: {self.config_name}")
                imgui.text(f"Orbits: {self.network.num_orbits} x {self.network.num_sats_per_orbit} sats")
                imgui.text(f"Alt: {self.network.altitude}km, Inc: {self.network.inclination}°")
                imgui.text(f"Min Elev: {self.network.min_elevation_angle_deg}°")
                imgui.text(f"Period: {self.network.orbit_cycle:.2f}s")
                imgui.unindent()

                imgui.separator()
                imgui.text(f"Time: {ms2str(self.current_time)}")
                imgui.text(f"FPS: {self.render_fps:.1f}")
                imgui.text(f"Satellites: {len(self.network.satellites)}")
                imgui.text(f"Ground Stations: {len(self.network.ground_stations)}")
                imgui.text(f"Links: {sum(1 for l in self.network.links.values() if l.is_connected)}/{len(self.network.links)}")

                imgui.separator()
                if self.paused:
                    if imgui.button("Resume"):
                        self.paused = False
                else:
                    if imgui.button("Pause"):
                        self.paused = True
                imgui.same_line()
                if imgui.button("Step"):
                    self.current_time += self.time_step_ms
                    self.network.update_topology(self.current_time)
                imgui.same_line()
                if imgui.button("Reset"):
                    self.current_time = 0
                    self.network.update_topology(self.current_time)

                _, self.animation_speed = imgui.slider_float("Speed", self.animation_speed, 0.1,
                                                             self.animation_speed_max, "%.1fx")

                imgui.separator()
                _, self.show_grid = imgui.checkbox("Show Grid", self.show_grid)
                _, self.show_sat_labels = imgui.checkbox("Show Sat Labels", self.show_sat_labels)
                _, self.show_gs_labels = imgui.checkbox("Show GS Labels", self.show_gs_labels)
                _, self.show_satellites = imgui.checkbox("Show Satellites", self.show_satellites)
                _, self.show_ground_stations = imgui.checkbox("Show Ground Stations", self.show_ground_stations)
                _, self.show_isls = imgui.checkbox("Show ISL Links", self.show_isls)
                _, self.show_gsls = imgui.checkbox("Show GSL Links", self.show_gsls)
                _, self.show_gs_tracking_circles = imgui.checkbox("Show GS Tracking Circles",
                                                                  self.show_gs_tracking_circles)
                _, self.show_orbit_trajectories = imgui.checkbox("Show Orbit Trajectories",
                                                                 self.show_orbit_trajectories)

                imgui.separator()
                _, self.satellite_size = imgui.slider_float("Sat Size", self.satellite_size, 1, 10)
                _, self.ground_station_size = imgui.slider_float("GS Size", self.ground_station_size, 1, 15)
                _, self.link_thickness = imgui.slider_float("Link Thick", self.link_thickness, 0.5, 3)

                imgui.separator()
                imgui.text("View Controls")
                if imgui.button("Zoom In"):
                    self.zoom_level = min(self.max_zoom, self.zoom_level * 1.1)
                imgui.same_line()
                if imgui.button("Zoom Out"):
                    self.zoom_level = max(self.min_zoom, self.zoom_level * 0.9)
                _, self.zoom_level = imgui.slider_float("Zoom", self.zoom_level, self.min_zoom, self.max_zoom, "%.2fx")
                if imgui.button("Reset View"):
                    self.zoom_level = 1
                    self.pan_offset_x = 0
                    self.pan_offset_y = 0
                imgui.separator()

                imgui.end_tab_item()
            if imgui.begin_tab_item("Satellite Visibility")[0]:
                imgui.text("Toggle satellite visibility:")
                if imgui.button("All On"):
                    self.satellite_visibility = {k: True for k in self.satellite_visibility}
                imgui.same_line()
                if imgui.button("All Off"):
                    self.satellite_visibility = {k: False for k in self.satellite_visibility}
                if imgui.begin_table("SatVisTable", 5):
                    imgui.table_setup_column("Vis")
                    imgui.table_setup_column("ID")
                    imgui.table_setup_column("Orb")
                    imgui.table_setup_column("Idx")
                    imgui.table_setup_column("LL")
                    imgui.table_headers_row()
                    for sat_id, sat in self.network.satellites.items():
                        imgui.table_next_row()
                        imgui.table_next_column()
                        changed, vis = imgui.checkbox(f"##vis{sat_id}", self.satellite_visibility.get(sat_id, True))
                        if changed:
                            self.satellite_visibility[sat_id] = vis
                        imgui.table_next_column()
                        imgui.text(f"S-{sat_id}")
                        imgui.table_next_column()
                        imgui.text(f"{getattr(sat, 'orbit', 'N/A')}")
                        imgui.table_next_column()
                        imgui.text(f"{getattr(sat, 'index_in_orbit', 'N/A')}")
                        imgui.table_next_column()
                        pos = sat.position
                        norm_pos = pos / np.linalg.norm(pos)
                        lat = np.degrees(np.arcsin(norm_pos[2]))
                        lon = np.degrees(np.arctan2(norm_pos[1], norm_pos[0]))
                        imgui.text(f"{lat:.1f},{lon:.1f}")
                    imgui.end_table()
                imgui.end_tab_item()
            if imgui.begin_tab_item("GSL Debug")[0]:
                imgui.text("Toggle visibility and inspect active GSLs:")
                if imgui.button("Enable All GSL Drawing"):
                    for link_id_key in self.gsl_draw_visibility:
                        self.gsl_draw_visibility[link_id_key] = True
                imgui.same_line()
                if imgui.button("Disable All GSL Drawing"):
                    for link_id_key in self.gsl_draw_visibility:
                        self.gsl_draw_visibility[link_id_key] = False
                imgui.separator()
                cols = 5
                table_flags = (
                        imgui.TABLE_RESIZABLE | imgui.TABLE_SCROLL_X | imgui.TABLE_SCROLL_Y | imgui.TABLE_SIZING_STRETCH_PROP
                )
                if imgui.begin_table("GSLDebugTable", cols, table_flags):
                    imgui.table_setup_column("Draw")
                    imgui.table_setup_column("ID")
                    imgui.table_setup_column("Source")
                    imgui.table_setup_column("Dest")
                    imgui.table_setup_column("Latency")
                    imgui.table_headers_row()
                    for link_id, link_obj in self.network.links.items():
                        if not link_obj.is_connected:
                            continue
                        is_gsl = isinstance(link_obj.sink, GroundStation) or isinstance(link_obj.source, GroundStation)
                        if not is_gsl:
                            continue

                        if link_id not in self.gsl_draw_visibility:
                            self.gsl_draw_visibility[link_id] = True  # Defensive

                        src_name = str(
                            link_obj.source.id
                            if hasattr(link_obj.source, "id")
                            else link_obj.source.name if hasattr(link_obj.source, "name") else "N/A"
                        )
                        dst_name = str(
                            link_obj.sink.id
                            if hasattr(link_obj.sink, "id")
                            else link_obj.sink.name if hasattr(link_obj.sink, "name") else "N/A"
                        )

                        src_lon, src_lat = _get_lon_lat(link_obj.source.position)
                        dst_lon, dst_lat = _get_lon_lat(link_obj.source.position)

                        if not (
                                src_lat is not None and src_lon is not None and dst_lat is not None and dst_lon is not None):
                            continue

                        def normalize_lon(l):
                            return (l + 180) % 360 - 180

                        lon1_n, lon2_n = normalize_lon(src_lon), normalize_lon(dst_lon)
                        delta = lon2_n - lon1_n
                        _crosses = abs(delta) > 180

                        imgui.table_next_row()
                        imgui.table_next_column()
                        initial_visibility = self.gsl_draw_visibility.get(link_id, True)
                        changed, current_vis = imgui.checkbox(f"##draw{link_id}", initial_visibility)
                        if changed:
                            self.gsl_draw_visibility[link_id] = current_vis

                        imgui.table_next_column()
                        imgui.text(str(link_id))
                        imgui.table_next_column()
                        imgui.text(src_name)
                        imgui.table_next_column()
                        imgui.text(dst_name)
                        imgui.table_next_column()
                        imgui.text(f"{link_obj.propagation_delay}ms")
                    imgui.end_table()
                imgui.end_tab_item()
            imgui.end_tab_bar()
        imgui.end()

    def render_frame(self):
        glClearColor(*self.colors["background"])
        glClear(GL_COLOR_BUFFER_BIT)
        self.impl.process_inputs()
        imgui.new_frame()
        self._update_hovered_satellite()  # Update hovered satellite ID
        self._handle_imgui_scroll()
        self._handle_imgui_pan()
        self._draw_map_panel()
        self._draw_settings_panel()
        imgui.render()
        self.impl.render(imgui.get_draw_data())
        glfw.swap_buffers(self.window)
        self._update_fps()

    def update(self):
        if not self.paused:
            self.current_time += int(self.time_step_ms * self.animation_speed)
            self.network.update_topology(self.current_time)

    def run(self):
        while not glfw.window_should_close(self.window):
            glfw.poll_events()
            self.update()
            self.render_frame()

    def cleanup(self):
        self._save_settings()  # Save settings before cleaning up
        self.texture_manager.cleanup()
        try:
            self.impl.shutdown()
            glfw.terminate()
            print("GLFW terminated.")
        except Exception as e:
            print(f"Error during GLFW cleanup: {e}")

    def save_screenshot(self, filename):
        raise NotImplementedError("Not implemented yet.")

    def save_animation(self, filename, fps, duration):
        raise NotImplementedError("Not implemented yet.")

    def _get_settings_to_persist(self):
        """Helper to gather settings into a dictionary for saving."""
        return {
            "show_grid": self.show_grid,
            "show_satellites": self.show_satellites,
            "show_ground_stations": self.show_ground_stations,
            "show_isls": self.show_isls,
            "show_gsls": self.show_gsls,
            "show_sat_labels": self.show_sat_labels,
            "show_gs_labels": self.show_gs_labels,

            "satellite_size": self.satellite_size,
            "ground_station_size": self.ground_station_size,
            "link_thickness": self.link_thickness,
            "animation_speed": self.animation_speed,
            "satellite_visibility": {str(k): v for k, v in self.satellite_visibility.items()},
            "gsl_draw_visibility": {str(k): v for k, v in self.gsl_draw_visibility.items()},
            "show_gs_tracking_circles": self.show_gs_tracking_circles,
            "show_orbit_trajectories": self.show_orbit_trajectories,
        }

    def _load_settings(self):
        try:
            if os.path.exists(self.settings_file_path):
                with open(self.settings_file_path, "r") as f:
                    settings = json.load(f)

                # Apply loaded settings, being careful about type and existence
                self.show_grid = settings.get("show_grid", self.show_grid)
                self.show_satellites = settings.get("show_satellites", self.show_satellites)
                self.show_ground_stations = settings.get("show_ground_stations", self.show_ground_stations)
                self.show_isls = settings.get("show_isls", self.show_isls)
                self.show_gsls = settings.get("show_gsls", self.show_gsls)
                self.show_sat_labels = settings.get("show_sat_labels", self.show_sat_labels)
                self.show_gs_labels = settings.get("show_gs_labels", self.show_gs_labels)

                self.satellite_size = settings.get("satellite_size", self.satellite_size)
                self.ground_station_size = settings.get("ground_station_size", self.ground_station_size)
                self.link_thickness = settings.get("link_thickness", self.link_thickness)
                self.animation_speed = settings.get("animation_speed", self.animation_speed)
                self.show_gs_tracking_circles = settings.get("show_gs_tracking_circles", self.show_gs_tracking_circles)
                self.show_orbit_trajectories = settings.get("show_orbit_trajectories", self.show_orbit_trajectories)

                loaded_sat_vis = settings.get("satellite_visibility")
                if loaded_sat_vis:
                    # Update existing keys, don't overwrite if some sats are new/gone
                    # Convert keys from loaded JSON (str) back to original type (likely int)
                    for sat_id_str, visibility in loaded_sat_vis.items():
                        try:
                            sat_id_original_type = type(next(iter(self.satellite_visibility.keys())))
                            sat_id = sat_id_original_type(sat_id_str)
                            if sat_id in self.satellite_visibility:
                                self.satellite_visibility[sat_id] = visibility
                        except (ValueError, StopIteration):
                            print(f"Warning: Could not process sat_id '{sat_id_str}' from settings.")
                loaded_gsl_vis = settings.get("gsl_draw_visibility")
                if loaded_gsl_vis:
                    # Initialize self.gsl_draw_visibility for all current GSLs first, then update from settings
                    current_gsl_ids = {
                        lid
                        for lid, lk in self.network.links.items()
                        if (isinstance(lk.source, Satellite) and isinstance(lk.sink, GroundStation))
                           or (isinstance(lk.source, GroundStation) and isinstance(lk.sink, Satellite))
                    }
                    for link_id_str, visibility in loaded_gsl_vis.items():
                        # Assume link_id_str from JSON can be directly used if it matches current link_ids.
                        # This might need adjustment if link_ids are not strings or change between sessions significantly
                        if link_id_str in current_gsl_ids:  # Check if this GSL still exists
                            self.gsl_draw_visibility[link_id_str] = visibility
                            # Or, if link_ids are integers, convert: link_id = int(link_id_str)
                        # For robust GSL visibility persistence across network changes, a more stable GSL identifier might be needed.
                print(f"Settings loaded from {self.settings_file_path}")
            else:
                print(f"Settings file not found at {self.settings_file_path}. Using default settings.")
        except Exception as e:
            print(f"Error loading settings from {self.settings_file_path}: {e}. Using default settings.")

    def _save_settings(self):
        try:
            settings_to_save = self._get_settings_to_persist()
            with open(self.settings_file_path, "w") as f:
                json.dump(settings_to_save, f, indent=4)
            print(f"Settings saved to {self.settings_file_path}")
        except Exception as e:
            print(f"Error saving settings to {self.settings_file_path}: {e}")

    def _load_land_geometries(self):
        """Loads land polygon geometries from a shapefile."""
        # Construct path assuming shapefile is in assets/ne_110m_land/
        shapefile_dir = os.path.join(self.assets_dir, "ne_110m_land")
        shapefile_path = os.path.join(shapefile_dir, "ne_110m_land.shp")
        print(f"Attempting to load land geometries from: {shapefile_path}")

        try:
            if not os.path.exists(shapefile_path):
                print(
                    f"Warning: Land shapefile not found at {shapefile_path}.\n"
                    f"Please download Natural Earth 110m land polygons and place them in {shapefile_dir}"
                )
                self.land_geometries = []  # Set to an empty list to avoid errors later
                return

            land_gdf = gpd.read_file(shapefile_path)
            # Ensure it uses WGS84 (lat/lon) coordinates, which Natural Earth typically does.
            # Reproject if necessary, though unlikely for default Natural Earth data.
            if land_gdf.crs and land_gdf.crs.to_epsg() != 4326:
                print(f"Reprojecting land geometries from {land_gdf.crs} to EPSG:4326")
                land_gdf = land_gdf.to_crs(epsg=4326)

            # Store the list of Shapely geometry objects (Polygons, MultiPolygons)
            self.land_geometries = land_gdf.geometry.tolist()
            print(f"Successfully loaded {len(self.land_geometries)} land features.")

        except ImportError:
            print("Warning: geopandas library not found. Cannot load vector land data. Please install it.")
            self.land_geometries = []  # Set to an empty list
        except Exception as e:
            print(f"Error loading land geometries from {shapefile_path}: {e}")
            self.land_geometries = []  # Set to an empty list on error

    def _draw_gs_tracking_circles(self):
        if not self.show_gs_tracking_circles:
            return

        draw_list = imgui.get_window_draw_list()
        circle_color = rgba_to_imgui_color(*self.colors["gs_tracking_circle"])
        thickness = 1.0
        num_segments = 36  # Number of line segments to approximate the circle

        # Constants
        R_e = 6371.0  # Earth radius in km
        h_sat = self.network.altitude  # Satellite altitude in km
        el_min_deg = self.network.min_elevation_angle_deg  # Min elevation angle for tracking satellites on the ground

        if h_sat <= 0:  # Avoid math errors if altitude is not positive
            return

        el_min_rad = np.radians(el_min_deg)
        R_sat = R_e + h_sat

        # Calculate the angle at the satellite between nadir and Earth limb (eta)
        # This is based on sin(eta) = R_e / R_sat
        # However, we need the angle from GS on Earth surface.

        # Calculate angular radius (alpha) on Earth's surface from GS to where the satellite is at el_min
        # angle_CSG = arcsin( (R_e * cos(el_min)) / (R_e + h_sat) )
        # alpha = 90_deg - el_min_deg - angle_CSG_deg
        cos_el_min = np.cos(el_min_rad)
        if (R_e * cos_el_min) / R_sat > 1.0 or (R_e * cos_el_min) / R_sat < -1.0:
            # Argument for arcsin is out of [-1, 1] range, perhaps due to very low el_min and h_sat
            # This means the satellite might always be below horizon or el_min is too high to be seen
            # For simplicity, we can skip drawing or draw a very small circle.
            # Let's skip for now if calculation is problematic.
            print(
                f"Warning: Cannot calculate GS tracking circle for h_sat={h_sat}, el_min={el_min_deg}. Arcsin arg out of range.")
            return

        angle_CSG_rad = np.arcsin((R_e * cos_el_min) / R_sat)
        alpha_rad = (np.pi / 2.0) - el_min_rad - angle_CSG_rad

        if alpha_rad <= 0:  # If the radius is zero or negative, no circle to draw
            return

        # angular_radius_on_surface_km = alpha_rad * R_e # This is the distance on surface

        for gs_id, gs in self.network.ground_stations.items():
            lat1_rad = np.radians(gs.latitude)
            lon1_rad = np.radians(gs.longitude)

            circle_points_geo = []
            for i in range(num_segments + 1):  # +1 to close the circle
                bearing_rad = (i / num_segments) * 2 * np.pi

                # Haversine formula for destination point given starting point, bearing, and distance (alpha_rad)
                lat2_rad = np.arcsin(
                    np.sin(lat1_rad) * np.cos(alpha_rad) + np.cos(lat1_rad) * np.sin(alpha_rad) * np.cos(bearing_rad)
                )
                lon2_rad = lon1_rad + np.arctan2(
                    np.sin(bearing_rad) * np.sin(alpha_rad) * np.cos(lat1_rad),
                    np.cos(alpha_rad) - np.sin(lat1_rad) * np.sin(lat2_rad),
                )

                lat2_deg = np.degrees(lat2_rad)
                lon2_deg = _normalize_longitude(np.degrees(lon2_rad))
                circle_points_geo.append((lat2_deg, lon2_deg))

            # Draw the polyline for the circle
            for i in range(num_segments):
                p1_lat, p1_lon = circle_points_geo[i]
                p2_lat, p2_lon = circle_points_geo[i + 1]
                self._draw_line_IDL(draw_list, p1_lat, p1_lon, p2_lat, p2_lon, circle_color, thickness, truncate=True)

    def _draw_orbit_trajectory_of_sat(self):
        if not self.show_orbit_trajectories or self.hovered_satellite_id is None:
            return

        hovered_sat = self.network.satellites.get(self.hovered_satellite_id)
        if not hovered_sat:
            return

        # Generate timestamps for the next 24 hours from current_time
        # self.current_time is in ms. 24 hours = 24 * 60 * 60 * 1000 ms
        # Let's take a point every 10 minutes (10 * 60 * 1000 ms)
        start_time_ms = self.current_time
        end_time_ms = start_time_ms + int(24 * 60 * 60 * 1000)
        time_step_ms = 60 * 1000

        future_timestamps = np.arange(start_time_ms, end_time_ms + time_step_ms, time_step_ms, dtype=np.int64)
        trajectory_ecef = hovered_sat.predict_position(future_timestamps)
        if trajectory_ecef is None or len(trajectory_ecef) < 2:
            return

        # Convert ECEF points to Lat/Lon
        points_geo = []
        # If predict_position returns a 2D array (N_points x 3_coords)
        if trajectory_ecef.ndim == 2 and trajectory_ecef.shape[1] == 3:
            for point_ecef in trajectory_ecef:
                lon, lat = _get_lon_lat(point_ecef)
                points_geo.append((lat, lon))
        # If it returns a flat array or something else, this might need adjustment
        # For now, assuming (N_points x 3_coords)
        else:
            print(f"Unexpected trajectory_ecef shape: {trajectory_ecef.shape}")
            return

    def _draw_line_IDL(self, draw_list, src_lat, src_lon, dst_lat, dst_lon, color, thickness, truncate: bool = False):
        """Draws a line segment, handling International Date Line (IDL) crossing.

        Args:
            draw_list: ImGui draw list.
            src_lat, src_lon: Latitude and longitude of the source point.
            dst_lat, dst_lon: Latitude and longitude of the destination point.
            color: Color of the line.
            thickness: Thickness of the line.
            truncate: If True, splits the line into two segments at the IDL.
                      If False, draw a single line (which will appear as the "long way"
                      if it crosses the IDL on a 2D map).
        """
        # Use the class's normalize_longitude method
        src_lon_norm = _normalize_longitude(src_lon)
        dst_lon_norm = _normalize_longitude(dst_lon)

        delta_lon = dst_lon_norm - src_lon_norm
        crosses_date_line = abs(delta_lon) > 180.0

        src_x, src_y = self.coord_converter.lat_lon_to_screen(src_lat, src_lon)
        dst_x, dst_y = self.coord_converter.lat_lon_to_screen(dst_lat, dst_lon)

        if crosses_date_line and truncate:
            if delta_lon > 0:  # Crosses westwards (e.g. from -170 lon to 170 lon on a -180 to 180 map)
                # Source is on the West part of the map, Destination on the East part.
                # The line segment logically goes from src_x leftwards, off the map, and re-enters from the right edge.
                p1_x, p1_y = src_x, src_y
                # Effective destination screen X if the map continued past longitude -180
                # (i.e., dst_x on a map shifted one width to the left)
                p2_eff_x = dst_x - self.map_width
                p2_eff_y = dst_y

                # dx_eff is the change in x for the line segment that crosses the IDL the short way
                dx_eff = p2_eff_x - p1_x

                y_at_idl_crossing: float
                # Calculate y-coordinate where the line segment P1 -> P2_eff intersects the West IDL (x=0)
                if abs(dx_eff) < 1e-6:  # Effectively a vertical line segment crossing IDL
                    # This implies P1.x and P2_eff.x are the same (e.g. src_x = 0 and dst_x = self.map_width).
                    # The line segment runs along the IDL. We use p1_y as the y-coordinate for the break.
                    y_at_idl_crossing = p1_y
                else:
                    # Using line equation y = y1 + m * (x_target - x1)
                    # m = (p2_eff_y - p1_y) / dx_eff
                    # x_target = 0.0 (West IDL)
                    y_at_idl_crossing = p1_y + (p2_eff_y - p1_y) * (0.0 - p1_x) / dx_eff

                # Segment 1: From source to West IDL (x=0)
                draw_list.add_line(src_x, src_y, 0.0, y_at_idl_crossing, color, thickness)
                # Segment 2: From East IDL (x=map_width) at same y-coordinate, to destination
                draw_list.add_line(self.map_width, y_at_idl_crossing, dst_x, dst_y, color, thickness)

            else:  # delta_lon < 0. Crosses eastwards (e.g. from 170 lon to -170 lon on a -180 to 180 map)
                # Source is on the East part of the map, Destination on the West part.
                # The line segment logically goes from src_x rightwards, off the map, and re-enters from the left edge.
                p1_x, p1_y = src_x, src_y
                # Effective destination screen X if the map continued past longitude +180
                # (i.e., dst_x on a map shifted one width to the right)
                p2_eff_x = dst_x + self.map_width
                p2_eff_y = dst_y

                dx_eff = p2_eff_x - p1_x

                y_at_idl_crossing: float
                # Calculate y-coordinate where the line segment P1 -> P2_eff intersects the East IDL (x=self.map_width)
                if abs(dx_eff) < 1e-6:  # Effectively a vertical line segment crossing IDL
                    y_at_idl_crossing = p1_y
                else:
                    # Using line equation y = y1 + m * (x_target - x1)
                    # m = (p2_eff_y - p1_y) / dx_eff
                    # x_target = self.map_width (East IDL)
                    y_at_idl_crossing = p1_y + (p2_eff_y - p1_y) * (self.map_width - p1_x) / dx_eff

                # Segment 1: From source to East IDL (x=self.map_width)
                draw_list.add_line(src_x, src_y, self.map_width, y_at_idl_crossing, color, thickness)
                # Segment 2: From West IDL (x=0) at same y-coordinate, to destination
                draw_list.add_line(0.0, y_at_idl_crossing, dst_x, dst_y, color, thickness)
        else:
            # No dateline crossing, OR (crosses_date_line is true BUT truncate is false).
            # In either case, draw a single direct line.
            # If crosses_date_line was true and truncate was false, this single line
            # will appear to go the "long way" across the map.
            draw_list.add_line(src_x, src_y, dst_x, dst_y, color, thickness)

    def _update_hovered_satellite(self):
        self.hovered_satellite_id = None  # Reset each frame
        mouse_x, mouse_y = self.io.mouse_pos.x, self.io.mouse_pos.y

        if mouse_x >= self.map_width or mouse_y >= self.map_height or mouse_x < 0 or mouse_y < 0:
            return  # Mouse is outside the map panel or in the settings panel

        # Define a hover threshold (e.g., satellite size + a small buffer)
        # This threshold is in screen pixels.
        hover_threshold = self.satellite_size + 3.0  # pixels

        for sat_id, sat in self.network.satellites.items():
            if not self.satellite_visibility.get(sat_id, True):
                continue

            lon, lat = _get_lon_lat(sat.position)
            sat_screen_x, sat_screen_y = self.coord_converter.lat_lon_to_screen(lat, lon)

            # Calculate distance on screen
            dist_sq = (mouse_x - sat_screen_x) ** 2 + (mouse_y - sat_screen_y) ** 2

            if dist_sq < hover_threshold ** 2:
                self.hovered_satellite_id = sat_id
                # Optional: If you want to only select the "topmost" or closest satellite
                # if multiple are within threshold, you might need to find the one with min_dist_sq.
                # For simplicity, first one found is fine for now.
                return  # Found a hovered satellite
