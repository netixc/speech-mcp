"""
Centralized constants for speech-mcp.

This module provides constants used throughout the speech-mcp extension.
It eliminates duplication by centralizing all shared constants in one place.
"""

import os
import sys
import pyaudio
from pathlib import Path

# File paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRANSCRIPTION_FILE = os.path.join(BASE_DIR, "transcription.txt")
RESPONSE_FILE = os.path.join(BASE_DIR, "response.txt")
COMMAND_FILE = os.path.join(BASE_DIR, "ui_command.txt")

# Log files
SERVER_LOG_FILE = os.path.join(BASE_DIR, "speech-mcp-server.log")
UI_LOG_FILE = os.path.join(BASE_DIR, "speech-mcp-ui.log")
MAIN_LOG_FILE = os.path.join(BASE_DIR, "speech-mcp.log")

# Audio parameters
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

# Audio notification files
AUDIO_DIR = os.path.join(BASE_DIR, "resources", "audio")
START_LISTENING_SOUND = os.path.join(AUDIO_DIR, "start_listening.wav")
STOP_LISTENING_SOUND = os.path.join(AUDIO_DIR, "stop_listening.wav")

# Default speech state has been moved to state_manager.py

# Configuration paths
CONFIG_DIR = os.path.join(str(Path.home()), '.config', 'speech-mcp')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'config.json')

# Environment variable names
ENV_TTS_VOICE = "SPEECH_MCP_TTS_VOICE"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OPENAI_TTS_API_BASE = "OPENAI_TTS_API_BASE_URL"
ENV_OPENAI_STT_API_BASE = "OPENAI_STT_API_BASE_URL"

# Default configuration values, using environment variables without hardcoded defaults
DEFAULT_CONFIG = {
    'tts': {
        'engine': os.environ.get('SPEECH_MCP_TTS_ENGINE'), 
        'voice': os.environ.get('SPEECH_MCP_TTS_VOICE'),
        'model': os.environ.get('SPEECH_MCP_TTS_MODEL'),
        'speed': float(os.environ.get('SPEECH_MCP_TTS_SPEED', '1.0')),  # Keeping default for float conversion
        'lang_code': 'en'
    },
    'stt': {
        'engine': 'openai',  # Only using OpenAI STT
        'model': os.environ.get('SPEECH_MCP_STT_MODEL'),
        'language': os.environ.get('SPEECH_MCP_STT_LANGUAGE')
    },
    'ui': {
        'theme': os.environ.get('SPEECH_MCP_UI_THEME')
    }
}

# UI Commands
CMD_LISTEN = "LISTEN"
CMD_SPEAK = "SPEAK"
CMD_IDLE = "IDLE"
CMD_UI_READY = "UI_READY"
CMD_UI_CLOSED = "UI_CLOSED"

# Speech recognition parameters
SILENCE_THRESHOLD = 0.015  # Threshold for detecting silence (higher = less sensitive)
MAX_SILENCE_DURATION = 1.5  # 1.5 seconds of silence to stop recording
SILENCE_CHECK_INTERVAL = 0.1  # Check every 100ms
SPEECH_TIMEOUT = 600  # 10 minutes timeout for speech recognition

# Streaming transcription parameters
STREAMING_END_SILENCE_DURATION = 1.5  # 1.5 seconds without speech to end streaming (matching example)
STREAMING_INITIAL_WAIT = 0.5  # 0.5 seconds initial wait before first silence check
STREAMING_PROCESSING_INTERVAL = 0.1   # Process streaming audio every 100ms
STREAMING_BUFFER_SIZE = 10  # Number of chunks to buffer before processing (about 0.5 seconds)
STREAMING_MAX_BUFFER_SIZE = 100  # Maximum buffer size to prevent memory issues
STREAMING_MIN_WORDS = 2  # Minimum number of words before considering end of speech