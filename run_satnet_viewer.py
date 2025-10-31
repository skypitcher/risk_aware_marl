#!/usr/bin/env python3
"""
Satellite Network Viewer Launcher

This script launches the interactive satellite network viewer from the project root directory.
It handles proper path setup and imports for the satnet_viewer package.
"""

import sys
from pathlib import Path

# Add the current directory (project root) to Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Now import and run the viewer
from satnet_viewer.app import run_satnet_viewer

if __name__ == "__main__":
    print(f"Starting Satellite Network Viewer from {PROJECT_ROOT}")
    print("=" * 60)
    run_satnet_viewer()
