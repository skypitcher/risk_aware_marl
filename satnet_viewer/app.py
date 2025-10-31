import argparse
import os
import traceback
from datetime import timedelta

import glfw

# Import from the project root
from sat_net import SatelliteNetwork
from sat_net.util import NamedDict

# Imports from the satnet_viewer package structure
from .renderer import SatNetViewer  # This relative import is fine


def run_satnet_viewer():
    parser = argparse.ArgumentParser(description='Dear ImGui Satellite Map Visualization - Network Viewer')
    parser.add_argument('--config', type=str, default='configs/starlink_dvbs2_train.json',
                      help='Path to environment config file (JSON format)')
    parser.add_argument('--save-screenshot', type=str, default=None,
                      help='Save a screenshot to the specified file and exit')
    parser.add_argument('--save-animation', type=str, default=None,
                      help='Save an animation to the specified file and exit')
    parser.add_argument('--duration-hours', type=float, default=24.0,
                      help='Duration of the animation in hours (default: 24h)')
    parser.add_argument('--fps', type=int, default=30,
                      help='Frames per second for saved animation (default: 30)')
    parser.add_argument('--step-seconds', type=float, default=1.0,
                      help='Time step in seconds (default: 1 second)')
    parser.add_argument('--width', type=int, default=1800,
                      help='Window width (default: 1800)')
    parser.add_argument('--height', type=int, default=900,
                      help='Window height (default: 900)')
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.isfile(args.config):
        print(f"Error: Config file not found at {args.config}")
        print("Please ensure the config path is correct. It should be relative to the project root or absolute.")
        return

    time_step = timedelta(seconds=args.step_seconds)
    duration = timedelta(hours=args.duration_hours)
    
    # Load JSON config and create NamedDict
    try:
        config_obj = NamedDict.load(args.config)
    except Exception as e:
        print(f"Error loading config file: {e}")
        print(f"Please check the config file format at {args.config}")
        print("Full traceback:")
        traceback.print_exc()
        return
    
    # Create satellite network using the network configuration
    try:
        network = SatelliteNetwork(
            ground_stations=config_obj.network.ground_stations,
            altitude=config_obj.network.altitude,
            inclination=config_obj.network.inclination,
            num_orbits=config_obj.network.num_orbits,
            num_sats_per_orbit=config_obj.network.num_sats_per_orbit,
            phasing=config_obj.network.phasing,
            min_elevation_angle_deg=config_obj.network.min_elevation_angle_deg,
            max_gsl_per_gs=config_obj.network.max_gsl_per_gs,
            max_gsl_per_sat=config_obj.network.max_gsl_per_sat,
            node_buffer_size=config_obj.network.node_buffer_size,
            link_buffer_size=config_obj.network.link_buffer_size,
            gsl_data_rate=config_obj.network.gsl_data_rate,
            isl_data_rate=config_obj.network.isl_data_rate,
        )
    except Exception as e:
        print(f"Error creating satellite network: {e}")
        print("Please check your network configuration parameters.")
        print("Full traceback:")
        traceback.print_exc()
        return
    
    # Create renderer instance
    try:
        renderer_instance = SatNetViewer(args.config, network, time_step, args.width, args.height)
    except Exception as e:
        print(f"Error creating renderer: {e}")
        print("Please check if you have the required graphics dependencies installed.")
        print("Full traceback:")
        traceback.print_exc()
        return

    print("Starting interactive satellite network viewer...")
    print("Press ESC to exit, or close the window.")
    renderer_instance.run()

    try:
        if args.save_screenshot:
            if renderer_instance.window:
                 glfw.make_context_current(renderer_instance.window) 
            renderer_instance.render_frame()
            renderer_instance.save_screenshot(args.save_screenshot)
            print(f"Screenshot saved to {args.save_screenshot}")
        elif args.save_animation:
            renderer_instance.save_animation(args.save_animation, args.fps, duration)
            print(f"Animation saved to {args.save_animation}")
        else:
            pass

    except Exception as e:
        print(f"Error running renderer: {e}")
        print("Full traceback:")
        traceback.print_exc()

    finally:
        try:
            renderer_instance.cleanup()
        except Exception as e:
            print(f"Error during cleanup: {e}")
            print("Full traceback:")
            traceback.print_exc()


if __name__ == "__main__":
    print("Running satnet_viewer directly. For correct project structure, run run_satnet_viewer.py from the project root.")
    run_satnet_viewer()