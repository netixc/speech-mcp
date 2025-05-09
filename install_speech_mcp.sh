#!/bin/bash
# Universal installation script for speech-mcp
# This script handles installation, virtual environment setup, and creates run scripts
# It automatically detects Python 3.10+ and configures the environment accordingly

set -e # Exit on error

# Text formatting
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Helper functions
print_status() {
    echo -e "${BOLD}${GREEN}==>${NC} $1"
}

print_warning() {
    echo -e "${BOLD}${YELLOW}Warning:${NC} $1"
}

print_error() {
    echo -e "${BOLD}${RED}Error:${NC} $1"
}

# Configuration
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/venv"
RUN_SCRIPT="${PROJECT_DIR}/run.sh"
GLOBAL_COMMAND="speech-mcp"
USER_BIN_DIR="$HOME/bin"
GLOBAL_PATH="${USER_BIN_DIR}/${GLOBAL_COMMAND}"
STANDALONE_SCRIPT="${PROJECT_DIR}/speech-mcp-bin"

# Check Python version
check_python() {
    print_status "Checking Python version..."
    if command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
        PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        if [[ $(echo "$PY_VERSION < 3.10" | bc) -eq 1 ]]; then
            print_error "Python 3.10 or higher is required, but you have $PY_VERSION"
            exit 1
        fi
    else
        print_error "Python 3.10 or higher is required but not found"
        exit 1
    fi
    
    PY_VERSION=$(${PYTHON_CMD} -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    print_status "Using $PYTHON_CMD (version $PY_VERSION)"
}

# Setup virtual environment
setup_venv() {
    print_status "Setting up virtual environment..."
    
    if [ -d "$VENV_DIR" ]; then
        print_status "Virtual environment already exists at ${VENV_DIR}"
        read -p "Recreate virtual environment? This will delete the existing one (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
            $PYTHON_CMD -m venv "$VENV_DIR"
            print_status "Created new virtual environment"
        fi
    else
        $PYTHON_CMD -m venv "$VENV_DIR"
        print_status "Created virtual environment at ${VENV_DIR}"
    fi
    
    # Activate the virtual environment
    source "${VENV_DIR}/bin/activate"
    print_status "Activated virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip
    print_status "Upgraded pip to latest version"
}

# Install dependencies and package
install_package() {
    print_status "Installing dependencies and package..."
    
    pip install -e .
    print_status "Installed speech-mcp in development mode"
    
    # Check if OpenAI is installed
    if ! pip show openai &> /dev/null; then
        print_status "Installing OpenAI package..."
        pip install openai
    fi
    
    # Check if PyAudio is installed 
    if ! pip show pyaudio &> /dev/null; then
        print_status "Installing PyAudio package..."
        pip install pyaudio
    fi
    
    # Check if PyQt5 is installed
    if ! pip show PyQt5 &> /dev/null; then
        print_status "Installing PyQt5 package..."
        pip install PyQt5
    fi
    
    print_status "All required packages installed"
}

# Create simple run script
create_run_script() {
    print_status "Creating run script..."
    
    # Create the run script
    cat > "$RUN_SCRIPT" << EOF
#!/bin/bash
# Simple script to run speech-mcp with environment variables

# Get the directory where this script is located
SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="\$SCRIPT_DIR/venv/bin/python"

# Load environment variables from .env file if it exists
if [ -f "\$SCRIPT_DIR/.env" ]; then
    echo "Loading environment variables from .env file..."
    set -o allexport
    source "\$SCRIPT_DIR/.env"
    set +o allexport
else
    echo "Warning: No .env file found. Using default configuration."
    echo "To configure speech-mcp, create a .env file (see .env.example)"
fi

# Check if the virtual environment exists
if [ ! -f "\$VENV_PYTHON" ]; then
    echo "Error: Virtual environment not found at \$VENV_PYTHON"
    echo "Please run ./install_speech_mcp.sh to setup the environment"
    exit 1
fi

# Display configuration information
echo "Starting speech-mcp with configuration:"
echo "  TTS Endpoint: \${OPENAI_TTS_API_BASE_URL:-[Default]}"
echo "  TTS Model: \${SPEECH_MCP_TTS_MODEL:-[Default]}"
echo "  TTS Voice: \${SPEECH_MCP_TTS_VOICE:-[Default]}"
echo "  STT Endpoint: \${OPENAI_STT_API_BASE_URL:-[Default]}"
echo "  STT Model: \${SPEECH_MCP_STT_MODEL:-[Default]}"

# Execute speech-mcp using the virtual environment Python
exec "\$VENV_PYTHON" -m speech_mcp "\$@"
EOF
    
    # Make the run script executable
    chmod +x "$RUN_SCRIPT"
    print_status "Created run script at ${RUN_SCRIPT}"
}

# Setup global command
setup_global_command() {
    print_status "Setting up global command..."
    
    # Create a standalone executable script
    cat > "$STANDALONE_SCRIPT" << EOF
#!/bin/bash
# Standalone executable script for speech-mcp

# Ensure we have a safe working directory
cd "\$HOME" 2>/dev/null || cd /tmp 2>/dev/null || cd / 2>/dev/null

# Constants - use absolute paths to avoid CWD issues
SCRIPT_DIR="${PROJECT_DIR}"
VENV_PYTHON="\${SCRIPT_DIR}/venv/bin/python"

# Set PYTHONPATH to ensure modules can be found
export PYTHONPATH="\${SCRIPT_DIR}:\${PYTHONPATH}"

# Load environment variables from .env file if it exists
if [ -f "\${SCRIPT_DIR}/.env" ]; then
    set -o allexport
    source "\${SCRIPT_DIR}/.env"
    set +o allexport
fi

# Check if the virtual environment exists
if [ ! -f "\${VENV_PYTHON}" ]; then
    echo "Error: Virtual environment not found at \${VENV_PYTHON}"
    echo "Please run ./install_speech_mcp.sh to setup the environment"
    exit 1
fi

# Execute speech-mcp using the virtual environment Python
exec "\${VENV_PYTHON}" -m speech_mcp "\$@"
EOF
    
    chmod +x "$STANDALONE_SCRIPT"
    print_status "Created executable script at ${STANDALONE_SCRIPT}"
    
    # Create user's bin directory if it doesn't exist
    if [ ! -d "$USER_BIN_DIR" ]; then
        print_status "Creating user bin directory at $USER_BIN_DIR..."
        mkdir -p "$USER_BIN_DIR"
    fi
    
    # Create the symlink in user's bin directory
    ln -sf "$STANDALONE_SCRIPT" "$GLOBAL_PATH"
    print_status "Created command in user's bin: ${GLOBAL_PATH}"
    
    # Check if user's bin is in PATH
    if [[ ":$PATH:" != *":$USER_BIN_DIR:"* ]]; then
        print_warning "Your ~/bin directory is not in your PATH"
        print_status "Add it to your PATH with one of these commands:"
        echo "  For Bash: echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.bashrc"
        echo "  For Zsh:  echo 'export PATH=\"\$HOME/bin:\$PATH\"' >> ~/.zshrc"
        print_status "Then restart your terminal or run:"
        echo "  source ~/.bashrc  # or source ~/.zshrc"
    else
        print_status "Your ~/bin directory is already in your PATH"
    fi
}

# Test the installation
test_installation() {
    print_status "Testing installation..."
    source "${VENV_DIR}/bin/activate"
    
    # Test import
    if python -c "import speech_mcp" &> /dev/null; then
        print_status "Package import successful"
    else
        print_error "Package import failed"
        exit 1
    fi
    
    print_status "Installation test completed successfully"
}

# Create .env file from example if it doesn't exist
create_env_file() {
    if [ ! -f "${PROJECT_DIR}/.env" ]; then
        print_status "Creating .env file from example..."
        cp "${PROJECT_DIR}/.env.example" "${PROJECT_DIR}/.env"
        print_status "Created .env file. Please edit it to add your OpenAI API key."
    else
        print_status ".env file already exists. You may want to check its configuration."
    fi
}

# Main installation process
main() {
    print_status "Starting speech-mcp installation..."
    
    check_python
    setup_venv
    install_package
    create_run_script
    setup_global_command
    create_env_file
    test_installation
    
    echo
    echo -e "${BOLD}${GREEN}Installation completed successfully!${NC}"
    echo
    echo "You can now run speech-mcp in multiple ways:"
    echo "  1. Using the global command: speech-mcp"
    echo "  2. Using the run script: ./run.sh"
    echo "  3. Using the standalone script: ./speech-mcp-bin"
    echo
    echo "Before using speech-mcp, make sure to:"
    echo "  1. Edit the .env file to add your OpenAI API key"
    echo "  2. Configure any other settings in the .env file"
    echo
    echo "For Goose integration, the global command should work seamlessly."
    echo
    echo "Enjoy using speech-mcp!"
}

# Run the main function
main
