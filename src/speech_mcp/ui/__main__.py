"""
Main UI module for the Speech MCP.

This module provides the main entry point for the UI component of the Speech MCP.
"""

import os
import sys
from speech_mcp.ui.pyqt import run_ui

def main():
    """Run the Speech UI."""
    # Run the PyQt UI
    return run_ui()

if __name__ == "__main__":
    sys.exit(main())