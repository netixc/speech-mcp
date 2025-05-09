"""
UI module for the Speech MCP.

This module provides the UI component of the Speech MCP.
"""

import sys
from speech_mcp.ui.pyqt import run_ui

def main():
    """Run the Speech UI."""
    return run_ui()

# Export the PyQt UI components
from speech_mcp.ui.pyqt import PyQtSpeechUI, run_ui

__all__ = [
    'main',
    'PyQtSpeechUI',
    'run_ui'
]