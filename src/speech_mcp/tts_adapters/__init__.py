"""
TTS adapters for speech-mcp

This package contains adapters for various text-to-speech engines.
Each adapter implements a common interface defined by BaseTTSAdapter.
"""

import os
from abc import ABC, abstractmethod
from typing import Optional, List, Dict, Any

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE

# Import configuration module
try:
    from speech_mcp.config import get_setting, set_setting, get_env_setting, set_env_setting
except ImportError:
    # Fallback if config module is not available
    def get_setting(section, key, default=None):
        return default
    
    def set_setting(section, key, value):
        return False
    
    def get_env_setting(name, default=None):
        return os.environ.get(name, default)
    
    def set_env_setting(name, value):
        os.environ[name] = value

class BaseTTSAdapter(ABC):
    """
    Base class for all TTS adapters.
    
    This abstract class defines the common interface that all TTS adapters must implement.
    It provides some common functionality and enforces a consistent API across different
    TTS engines.
    """
    
    def __init__(self, voice: str = None, lang_code: str = "en", speed: float = 1.0):
        """
        Initialize the TTS adapter.
        
        Args:
            voice: The voice to use (default determined by implementation)
            lang_code: The language code to use (default: "en" for English)
            speed: The speaking speed (default: 1.0)
        """
        # Get voice preference from config or environment variable if not specified
        if voice is None:
            # First try environment variable
            env_voice = get_env_setting(ENV_TTS_VOICE)
            if env_voice:
                voice = env_voice
            else:
                # Then try config file
                config_voice = get_setting("tts", "voice", None)
                if config_voice:
                    voice = config_voice
        
        self.voice = voice
        self.lang_code = lang_code
        self.speed = speed
        self.is_initialized = False
    
    @abstractmethod
    def speak(self, text: str) -> bool:
        """
        Speak the given text.
        
        Args:
            text: The text to speak
            
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def save_to_file(self, text: str, file_path: str) -> bool:
        """
        Save speech as an audio file.
        
        Args:
            text: The text to convert to speech
            file_path: Path where to save the audio file
        
        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_available_voices(self) -> List[str]:
        """
        Get a list of available voices.
        
        Returns:
            List[str]: List of available voice names
        """
        pass
    
    def set_voice(self, voice: str) -> bool:
        """
        Set the voice to use.
        
        Args:
            voice: The voice to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            old_voice = self.voice
            self.voice = voice
            
            # Save the voice preference to config and environment variable
            try:
                # Save to config file
                set_setting("tts", "voice", voice)
                
                # Save to environment variable
                set_env_setting(ENV_TTS_VOICE, voice)
            except Exception:
                pass
            
            return True
        except Exception:
            return False
    
    def set_speed(self, speed: float) -> bool:
        """
        Set the speaking speed.
        
        Args:
            speed: The speaking speed (1.0 is normal)
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            self.speed = speed
            return True
        except Exception:
            return False


# Try to import available adapters
try:
    from .openai_adapter import OpenAITTS
except ImportError:
    pass

# Import direct adapter (should be more reliable)
try:
    from .direct_adapter import DirectTTS
except ImportError:
    pass
