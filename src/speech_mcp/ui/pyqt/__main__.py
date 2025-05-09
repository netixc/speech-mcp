"""
Entry point for running the PyQt UI directly.

This module allows the PyQt UI to be run directly for testing.
"""

import sys
from speech_mcp.ui.pyqt.pyqt_ui import run_ui

if __name__ == "__main__":
    # Run the UI
    sys.exit(run_ui())