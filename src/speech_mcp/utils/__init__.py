"""
Utility modules for speech-mcp.
"""

from speech_mcp.utils.logger import get_logger, set_log_level, get_log_files

# Import the sound player
try:
    from speech_mcp.utils.sound_player import SoundPlayer
    # Also try to import the direct TTS module
    try:
        from speech_mcp.utils.direct_tts import speak_text, get_tts_instance
        __all__ = ['get_logger', 'set_log_level', 'get_log_files', 'SoundPlayer', 'speak_text', 'get_tts_instance']
    except ImportError:
        __all__ = ['get_logger', 'set_log_level', 'get_log_files', 'SoundPlayer']
except ImportError:
    # Try to import just the direct TTS module
    try:
        from speech_mcp.utils.direct_tts import speak_text, get_tts_instance
        __all__ = ['get_logger', 'set_log_level', 'get_log_files', 'speak_text', 'get_tts_instance']
    except ImportError:
        # If neither module is available, just include the logger functions
        __all__ = ['get_logger', 'set_log_level', 'get_log_files']