#!/bin/bash
# Comprehensive uninstall script for speech-mcp
# This script removes all installation artifacts, including global commands

set -e # Exit on error

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
OLD_VENV_DIR="${PROJECT_DIR}/venv_py310"
RUN_SCRIPT="${PROJECT_DIR}/run.sh"
GLOBAL_SCRIPT="${PROJECT_DIR}/speech-mcp-bin"
GLOBAL_COMMAND="speech-mcp"
USER_BIN_DIR="$HOME/bin"
USER_CMD_PATH="$USER_BIN_DIR/$GLOBAL_COMMAND"
LOCAL_BIN_DIR="$HOME/.local/bin"
LOCAL_CMD_PATH="$LOCAL_BIN_DIR/$GLOBAL_COMMAND"

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Helper function for status messages
print_status() {
    echo -e "${BOLD}${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${BOLD}${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${BOLD}${RED}Error:${NC} $1"
}

# Remove global commands
remove_global_commands() {
    print_status "Removing global commands..."

    # Remove from user's bin
    if [ -L "$USER_CMD_PATH" ] || [ -f "$USER_CMD_PATH" ]; then
        rm -f "$USER_CMD_PATH"
        print_status "Removed command from user's bin: $USER_CMD_PATH"
    else
        print_status "Command not found in user's bin: $USER_CMD_PATH"
    fi

    # Remove from .local/bin
    if [ -L "$LOCAL_CMD_PATH" ] || [ -f "$LOCAL_CMD_PATH" ]; then
        rm -f "$LOCAL_CMD_PATH"
        print_status "Removed command from local bin: $LOCAL_CMD_PATH"
    else
        print_status "Command not found in local bin: $LOCAL_CMD_PATH"
    fi

    # Check for any other speech-mcp commands in PATH
    OTHER_COMMANDS=$(find $(echo $PATH | tr ':' ' ') -name "$GLOBAL_COMMAND" 2>/dev/null || true)
    if [ -n "$OTHER_COMMANDS" ]; then
        print_warning "Found other speech-mcp commands in PATH:"
        echo "$OTHER_COMMANDS"
        read -p "Do you want to remove these commands? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            for cmd in $OTHER_COMMANDS; do
                rm -f "$cmd"
                print_status "Removed command: $cmd"
            done
        fi
    fi

    # Remove the standalone script
    if [ -f "$GLOBAL_SCRIPT" ]; then
        rm -f "$GLOBAL_SCRIPT"
        print_status "Removed standalone script: $GLOBAL_SCRIPT"
    else
        print_status "Standalone script not found: $GLOBAL_SCRIPT"
    fi
}

# Remove run script
remove_run_script() {
    print_status "Removing run script..."

    if [ -f "$RUN_SCRIPT" ]; then
        rm -f "$RUN_SCRIPT"
        print_status "Removed run script from $RUN_SCRIPT"
    else
        print_status "Run script not found at $RUN_SCRIPT"
    fi
}

# Remove virtual environments
remove_venvs() {
    print_status "Removing virtual environments..."

    # Remove current venv
    if [ -d "$VENV_DIR" ]; then
        rm -rf "$VENV_DIR"
        print_status "Removed virtual environment from $VENV_DIR"
    else
        print_status "Virtual environment not found at $VENV_DIR"
    fi

    # Remove old venv
    if [ -d "$OLD_VENV_DIR" ]; then
        rm -rf "$OLD_VENV_DIR"
        print_status "Removed old virtual environment from $OLD_VENV_DIR"
    else
        print_status "Old virtual environment not found at $OLD_VENV_DIR"
    fi
}

# Remove build artifacts
remove_build_artifacts() {
    print_status "Removing build artifacts..."

    # Remove build directories
    find "$PROJECT_DIR" -maxdepth 1 -name "build" -type d -exec rm -rf {} \; 2>/dev/null || true
    find "$PROJECT_DIR" -maxdepth 1 -name "dist" -type d -exec rm -rf {} \; 2>/dev/null || true

    # Remove egg-info directories
    find "$PROJECT_DIR" -name "*.egg-info" -type d -exec rm -rf {} \; 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.dist-info" -type d -exec rm -rf {} \; 2>/dev/null || true

    print_status "Removed build artifacts"
}

# Clean up __pycache__ files
clean_pycache() {
    print_status "Cleaning up Python cache files..."

    find "$PROJECT_DIR" -name "__pycache__" -type d -exec rm -rf {} \; 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyc" -type f -delete 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyo" -type f -delete 2>/dev/null || true
    find "$PROJECT_DIR" -name "*.pyd" -type f -delete 2>/dev/null || true
    find "$PROJECT_DIR" -name ".pytest_cache" -type d -exec rm -rf {} \; 2>/dev/null || true
    find "$PROJECT_DIR" -name ".coverage" -type f -delete 2>/dev/null || true
    find "$PROJECT_DIR" -name ".DS_Store" -type f -delete 2>/dev/null || true

    print_status "Cleaned up Python cache files"
}

# Main uninstallation process
main() {
    print_status "Starting speech-mcp uninstallation..."

    remove_global_commands
    remove_run_script
    remove_venvs
    remove_build_artifacts
    clean_pycache

    echo
    echo -e "${BOLD}${GREEN}Uninstallation completed successfully!${NC}"
    echo
    echo "The speech-mcp package has been completely uninstalled from your system."
    echo
    echo "The speech-mcp source code remains in this directory."
}

# Run the main function
main