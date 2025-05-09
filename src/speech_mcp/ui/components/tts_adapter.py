"""
Text-to-speech adapter for the Speech UI.

This module provides a PyQt wrapper around the TTS adapters.
"""

import os
import time
import threading
import random
import math
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="tts")

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE

class TTSAdapter(QObject):
    """
    Text-to-speech adapter for PyQt UI.
    
    This class provides a Qt wrapper around the TTS adapters to integrate with PyQt signals.
    """
    speaking_finished = pyqtSignal()
    speaking_started = pyqtSignal()
    speaking_progress = pyqtSignal(float)  # Progress between 0.0 and 1.0
    audio_level = pyqtSignal(float)  # Audio level for visualization
    
    def __init__(self):
        super().__init__()
        self.tts_engine = None
        self.is_speaking = False
        self._speaking_lock = threading.Lock()  # Add a lock for thread safety
        self.available_voices = []
        self.current_voice = None
        self.initialize_tts()
    
    def initialize_tts(self):
        """Initialize the TTS engine using the adapter system"""
        try:
            # Try to import the TTS adapters
            logger.info("Initializing TTS using adapter system")
            
            # Get the configured TTS engine preference
            try:
                from speech_mcp.config import get_setting
                tts_engine_name = get_setting("tts", "engine", "openai")
                logger.info(f"Using configured TTS engine: {tts_engine_name}")
            except Exception as e:
                logger.warning(f"Failed to get TTS engine preference: {e}")
                tts_engine_name = "openai"  # Default to OpenAI
            
            # Import adapters
            try:
                # Try the direct adapter first (more reliable)
                try:
                    from speech_mcp.tts_adapters import DirectTTS
                    logger.info("Direct TTS adapter imported successfully")
                    direct_available = True
                except ImportError as e:
                    logger.error(f"Direct TTS adapter import error: {e}")
                    direct_available = False
                
                # If direct adapter not available, try OpenAI adapter
                if not direct_available:
                    try:
                        from speech_mcp.tts_adapters import OpenAITTS
                        logger.info("OpenAI TTS adapter imported successfully")
                        openai_available = True
                    except ImportError as e:
                        logger.error(f"OpenAI TTS adapter import error: {e}")
                        openai_available = False
                        raise
                else:
                    openai_available = False
            except ImportError as e:
                logger.warning(f"Failed to import TTS adapters: {e}")
                direct_available = False
                openai_available = False
                raise
            
            # Try to initialize Direct TTS first (more reliable)
            if direct_available:
                try:
                    logger.info("Initializing Direct TTS adapter")
                    # Try to initialize with environment variables
                    model = os.environ.get("SPEECH_MCP_TTS_MODEL")
                    voice = os.environ.get("SPEECH_MCP_TTS_VOICE")
                    logger.info(f"TTS Model from env: {model}")
                    logger.info(f"TTS Voice from env: {voice}")
                    
                    self.tts_engine = DirectTTS(voice=voice, model=model)
                    logger.info(f"Direct TTS initialized: {self.tts_engine.is_initialized}")
                    
                    if self.tts_engine.is_initialized:
                        logger.info("Direct TTS adapter initialized successfully")
                    else:
                        logger.error("Direct TTS adapter initialization failed")
                        # Try OpenAI as fallback
                        raise ImportError("Direct TTS initialization failed")
                except Exception as e:
                    logger.warning(f"Failed to initialize Direct TTS adapter: {e}")
                    # Fall back to OpenAI TTS if available
                    if openai_available:
                        try:
                            logger.info("Falling back to OpenAI TTS adapter")
                            self.tts_engine = OpenAITTS(voice=voice, model=model)
                            logger.info(f"OpenAI TTS initialized: {self.tts_engine.is_initialized}")
                            
                            if self.tts_engine.is_initialized:
                                logger.info("OpenAI TTS adapter initialized successfully")
                            else:
                                logger.error("OpenAI TTS adapter initialization failed")
                                raise ImportError("OpenAI initialization failed")
                        except Exception as e2:
                            logger.warning(f"Failed to initialize OpenAI TTS adapter: {e2}")
                            self.tts_engine = None
                    else:
                        self.tts_engine = None
            # Initialize OpenAI TTS if Direct TTS wasn't available
            elif openai_available:
                try:
                    logger.info("Initializing OpenAI TTS adapter")
                    # Try to initialize with environment variables
                    model = os.environ.get("SPEECH_MCP_TTS_MODEL")
                    voice = os.environ.get("SPEECH_MCP_TTS_VOICE")
                    logger.info(f"TTS Model from env: {model}")
                    logger.info(f"TTS Voice from env: {voice}")
                    
                    self.tts_engine = OpenAITTS(voice=voice, model=model)
                    logger.info(f"OpenAI TTS initialized: {self.tts_engine.is_initialized}")
                    
                    if self.tts_engine.is_initialized:
                        logger.info("OpenAI TTS adapter initialized successfully")
                    else:
                        logger.error("OpenAI TTS adapter initialization failed")
                        raise ImportError("OpenAI initialization failed")
                except Exception as e:
                    logger.warning(f"Failed to initialize OpenAI TTS adapter: {e}")
                    self.tts_engine = None
            
            # If we have a TTS engine, get the available voices
            if self.tts_engine:
                voices = self.tts_engine.get_available_voices()
                logger.info(f"Available voices from TTS engine: {voices}")
                current_voice = self.tts_engine.voice
                logger.info(f"Current voice from TTS engine: {current_voice}")
                
                # Set instance variables
                self.available_voices = voices
                self.current_voice = current_voice
                
                logger.info(f"TTS initialized with {len(self.available_voices)} voices, current voice: {self.current_voice}")
                return True
            else:
                logger.error("No TTS engine available")
                self.available_voices = []
                self.current_voice = None
                return False
                
        except ImportError as e:
            logger.warning(f"Failed to import TTS adapters: {e}")
            logger.error("No OpenAI TTS engine available. Please check your configuration.")
            return False
            
        except Exception as e:
            logger.error(f"Error initializing TTS: {e}")
            return False
    
    def speak(self, text):
        """Speak the given text"""
        if not text:
            logger.warning("Empty text provided to speak")
            return False
               
        # Use a lock to safely check and update speaking state
        with self._speaking_lock:
            if self.is_speaking:
                logger.warning("Already speaking, ignoring new request")
                return False
            
            # Set speaking state before starting thread
            self.is_speaking = True
        
        logger.info(f"TTSAdapter.speak called with text: {text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Emit speaking started signal on the main thread
        self.speaking_started.emit()
        
        # USE THE DIRECT TTS IMPLEMENTATION
        try:
            # Import our direct TTS implementation
            from speech_mcp.utils.direct_tts import speak_text
            
            # Define a function to handle speech in a separate thread
            def speak_thread():
                result = speak_text(text, voice=self.current_voice)
                logger.info(f"Direct TTS speech completed with result: {result}")
                # Use a lock to safely update the speaking state
                with self._speaking_lock:
                    self.is_speaking = False
                
                # Emit the signal after releasing the lock
                self.speaking_finished.emit()
            
            # Start speaking in a background thread
            logger.info("Using direct TTS implementation")
            thread = threading.Thread(target=speak_thread, daemon=True)
            thread.start()
            logger.info("Direct TTS thread started")
            return True
            
        except Exception as e:
            logger.warning(f"Direct TTS not available, falling back to standard method: {e}")
            
            # Fall back to standard method
            speak_thread = threading.Thread(target=self._speak_thread, args=(text,), daemon=True)
            speak_thread.start()
            logger.debug("Started fallback _speak_thread")
            return True
    
    def emit_audio_level(self):
        """Emit audio level signal for visualization"""
        # Use the lock to safely check the speaking state
        with self._speaking_lock:
            is_speaking = self.is_speaking
        
        if not is_speaking:
            if hasattr(self, 'audio_level_timer') and self.audio_level_timer.isActive():
                self.audio_level_timer.stop()
            self.audio_level.emit(0.0)  # Reset to zero when not speaking
            return
        
        # When speaking, we don't need to emit actual levels since we're using pre-recorded patterns
        # Just emit a dummy signal to trigger visualization updates
        self.audio_level.emit(0.5)
    
    def _speak_thread(self, text):
        """Thread function for speaking text"""
        try:
            logger.info(f"_speak_thread started for text: {text[:50]}{'...' if len(text) > 50 else ''}")
            
            # DIRECT TTS METHOD - Bypass adapter layers
            try:
                # Get required settings
                voice = self.current_voice
                if not voice:
                    voice = "bm_daniel"  # Default fallback
                
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
                model = os.environ.get("SPEECH_MCP_TTS_MODEL")
                
                # Import required modules
                import tempfile
                import requests
                import sys
                
                logger.info(f"Direct TTS: Using voice={voice}, model={model}")
                
                # Set up request
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                url = f"{base_url}/audio/speech"
                
                # Request data
                data = {
                    "model": model,
                    "voice": voice,
                    "input": text,
                    "response_format": "wav"
                }
                
                # Make request
                logger.info(f"Direct TTS: Sending request to {url}")
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                # Save to temp file
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(response.content)
                
                logger.info(f"Direct TTS: Audio saved to {temp_path}, size: {os.path.getsize(temp_path)} bytes")
                
                # Play audio using platform-specific method
                if sys.platform == "darwin":  # macOS
                    os.system(f'afplay "{temp_path}"')
                    # Backup: also try opening the file
                    os.system(f'open "{temp_path}"')
                elif sys.platform == "win32":  # Windows
                    os.system(f'powershell -c "(New-Object Media.SoundPlayer \'{temp_path}\').PlaySync()"')
                else:  # Linux
                    os.system(f'aplay "{temp_path}"')
                
                logger.info("Direct TTS: Audio playback complete")
                
                # Clean up - delayed deletion
                import threading
                def delayed_delete():
                    import time
                    time.sleep(10)  # Wait to ensure audio completes
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                            logger.info(f"Direct TTS: Deleted temp file {temp_path}")
                    except Exception as e:
                        logger.error(f"Direct TTS: Error deleting temp file: {e}")
                
                # Start cleanup in background
                threading.Thread(target=delayed_delete, daemon=True).start()
                
                return True
                
            except Exception as e:
                logger.error(f"Direct TTS error: {e}")
                
                # Fall back to adapter method if direct method fails
                if hasattr(self.tts_engine, 'speak'):
                    logger.info("Falling back to TTS adapter speak method")
                    try:
                        result = self.tts_engine.speak(text)
                        logger.info(f"TTS speak result: {result}")
                        if not result:
                            logger.error("TTS failed")
                            return False
                        return True
                    except Exception as e:
                        logger.error(f"Exception in TTS speak fallback: {e}", exc_info=True)
                        return False
                else:
                    logger.error("TTS engine does not have speak method")
                    return False
            
            logger.info("Speech completed")
        except Exception as e:
            logger.error(f"Error during text-to-speech: {e}", exc_info=True)
        finally:
            # Use the lock to safely update the speaking state
            with self._speaking_lock:
                self.is_speaking = False
            
            # Emit the signal after releasing the lock
            self.speaking_finished.emit()
            logger.info("Speaking finished signal emitted")
    
    def set_voice(self, voice_id):
        """Set the voice to use for TTS"""
        if not self.tts_engine:
            logger.warning("No TTS engine available")
            return False
        
        try:
            if hasattr(self.tts_engine, 'set_voice'):
                # Use the adapter's set_voice method
                result = self.tts_engine.set_voice(voice_id)
                if result:
                    self.current_voice = voice_id
                    logger.info(f"Voice set to: {voice_id}")
                    return True
                else:
                    logger.error(f"Failed to set voice to: {voice_id}")
                    return False
            
            logger.warning("TTS engine does not support voice selection")
            return False
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
    
    def get_available_voices(self):
        """Get a list of available voices"""
        # Log available voices to help debug
        logger.debug(f"TTS Adapter returning available voices: {self.available_voices}")
        return self.available_voices
    
    def get_current_voice(self):
        """Get the current voice"""
        logger.debug(f"TTS Adapter returning current voice: {self.current_voice}")
        return self.current_voice