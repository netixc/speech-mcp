"""
Speech recognition module for speech-mcp.

This module provides centralized speech recognition functionality including:
- Model loading and initialization
- Audio transcription (file-based and streaming)
- Fallback mechanisms
- Consistent error handling

It consolidates speech recognition code that was previously duplicated
across server.py and speech_ui.py.
"""

# Import the streaming transcriber
from speech_mcp.streaming_transcriber import StreamingTranscriber

import os
import time
import tempfile
from datetime import timedelta, datetime
from typing import Optional, Tuple, Dict, Any, List, Union, Callable

# Import constants
from speech_mcp.constants import ENV_OPENAI_API_KEY, ENV_OPENAI_TTS_API_BASE, ENV_OPENAI_STT_API_BASE

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Try to import OpenAI STT adapter
try:
    from speech_mcp.tts_adapters.openai_stt_adapter import OpenAISTT
except ImportError:
    OpenAISTT = None

# Get a logger for this module
logger = get_logger(__name__, component="stt")

class SpeechRecognizer:
    """
    Core speech recognition class that handles transcription of audio files.
    
    This class provides a unified interface for speech recognition.
    It uses OpenAI STT as the primary and only engine, with SpeechRecognition as a fallback
    only if OpenAI is not available.
    """
    
    def __init__(self, model_name: str = None, language: str = "en"):
        """
        Initialize the speech recognizer.
        
        Args:
            model_name: The name of the model to use
            language: The language code for transcription (e.g., "en" for English)
        """
        self.sr_recognizer = None
        self.openai_stt = None
        
        # Get values from environment variables if not provided
        self.engine = "openai"  # Always use OpenAI STT
        self.model_name = model_name or os.environ.get('SPEECH_MCP_STT_MODEL')
        self.language = language or os.environ.get('SPEECH_MCP_STT_LANGUAGE', 'en')
        
        self.is_initialized = False
        
        # Add streaming transcriber
        self.streaming_transcriber = None
        
        # Initialize the speech recognition models in the background
        self._initialize_speech_recognition()
    
    def _initialize_speech_recognition(self) -> bool:
        """
        Initialize speech recognition models.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        if self.is_initialized:
            logger.info("Speech recognition already initialized")
            return True
            
        # Try to initialize OpenAI STT first if that's the selected engine
        if self.engine.lower() == "openai":
            try:
                if OpenAISTT is None:
                    raise ImportError("OpenAI STT adapter not available")
                    
                logger.info(f"Initializing OpenAI STT with model '{self.model_name}'...")
                
                # Initialize with keyword arguments for better clarity and stability
                api_key = os.environ.get("OPENAI_API_KEY")
                api_base = os.environ.get("OPENAI_STT_API_BASE_URL")
                
                # Log configuration for debugging
                logger.info(f"STT Configuration:")
                logger.info(f"  API Key: {'[SET]' if api_key else '[NOT SET]'}")
                logger.info(f"  API Base URL: {api_base}")
                logger.info(f"  Model: {self.model_name}")
                
                self.openai_stt = OpenAISTT(
                    api_key=api_key,
                    api_base=api_base,
                    model=self.model_name
                )
                
                logger.info("OpenAI STT initialized successfully!")
                
                self.is_initialized = True
                return True
                
            except ImportError as e:
                logger.error(f"Failed to load OpenAI STT adapter: {e}")
                self.is_initialized = False
                return False
            except Exception as e:
                logger.error(f"Error initializing OpenAI STT: {e}")
                self.is_initialized = False
                return False
        else:
            # Only OpenAI STT is supported
            logger.error(f"Invalid engine: {self.engine}, only 'openai' is supported")
            self.is_initialized = False
            return False
    
    def _initialize_speech_recognition_fallback(self) -> bool:
        """
        Initialize fallback speech recognition using SpeechRecognition library.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            logger.info("Initializing SpeechRecognition fallback...")
            import speech_recognition as sr
            self.sr_recognizer = sr.Recognizer()
            
            logger.info("SpeechRecognition library loaded successfully as fallback!")
            
            self.is_initialized = True
            return True
            
        except ImportError as e:
            logger.error(f"Failed to load SpeechRecognition: {e}")
            logger.warning("Please install it with: pip install SpeechRecognition")
            
            self.is_initialized = False
            return False
        except Exception as e:
            logger.error(f"Error initializing SpeechRecognition: {e}")
            
            self.is_initialized = False
            return False
    
    def transcribe(self, audio_file_path: str, language: str = "en", 
                  include_timestamps: bool = False, detect_speakers: bool = False) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe an audio file using OpenAI STT or fall back to Google Speech Recognition.
        
        Args:
            audio_file_path: Path to the audio file to transcribe
            language: Language code for transcription (default: "en" for English)
            include_timestamps: Whether to include word-level timestamps (not supported with OpenAI STT)
            detect_speakers: Whether to attempt speaker detection (not supported with OpenAI STT)
            
        Returns:
            Tuple containing:
                - The transcribed text
                - A dictionary with metadata about the transcription and timing information
        """
        # Check if the file exists
        if not os.path.exists(audio_file_path):
            error_msg = f"Audio file not found: {audio_file_path}"
            logger.error(error_msg)
            return "", {"error": error_msg, "engine": "none"}
        
        # Ensure speech recognition is initialized
        if not self.is_initialized and not self._initialize_speech_recognition():
            error_msg = "Failed to initialize speech recognition"
            logger.error(error_msg)
            return "", {"error": error_msg, "engine": "none"}
        
        # Use OpenAI STT if available
        if self.openai_stt is not None:
            try:
                transcription_start = time.time()
                
                logger.info(f"Transcribing audio with OpenAI STT: {audio_file_path}")
                logger.info(f"Audio file exists: {os.path.exists(audio_file_path)}")
                logger.info(f"Audio file size: {os.path.getsize(audio_file_path)} bytes")
                
                try:
                    # Use OpenAI STT with the configured language
                    result = self.openai_stt.transcribe_file(
                        audio_file_path,
                        language=language
                    )
                    
                    # Check if there was an error in the result
                    if "error" in result:
                        error_msg = result.get("error", "Unknown error in STT API response")
                        logger.error(f"Error in OpenAI STT response: {error_msg}")
                        raise RuntimeError(error_msg)
                    
                    # Log the result (only in debug mode to avoid logging sensitive information)
                    if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
                        logger.debug(f"Transcription result: {result}")
                    else:
                        logger.info("Transcription result received (set LOG_LEVEL=DEBUG to see details)")
                    
                    # Get the transcribed text from the result
                    transcription = self.openai_stt.get_text_from_result(result)
                    logger.info(f"Extracted transcription: {transcription[:100]}{'...' if len(transcription) > 100 else ''}")
                    
                except Exception as e:
                    logger.error(f"Error in OpenAI STT transcription: {e}")
                    raise
                
                transcription_time = time.time() - transcription_start
                
                logger.info(f"OpenAI STT transcription completed in {transcription_time:.2f}s")
                
                # Create metadata for the result
                metadata = {
                    "engine": "openai",
                    "model": self.model_name,
                    "time_taken": transcription_time,
                    "has_timestamps": False,
                    "has_speakers": False,
                    "language": language
                }
                
                # Note about unsupported features
                if include_timestamps or detect_speakers:
                    logger.warning("Timestamps and speaker detection are not supported with OpenAI STT")
                
                return transcription, metadata
                
            except Exception as e:
                logger.error(f"Error transcribing with OpenAI STT: {e}")
                logger.info("Falling back to Google Speech Recognition...")
        
        # Fall back to SpeechRecognition if available
        if self.sr_recognizer is not None:
            try:
                import speech_recognition as sr
                
                logger.info(f"Transcribing audio with SpeechRecognition (fallback): {audio_file_path}")
                
                transcription_start = time.time()
                
                with sr.AudioFile(audio_file_path) as source:
                    audio_data = self.sr_recognizer.record(source)
                    transcription = self.sr_recognizer.recognize_google(audio_data, language=language)
                
                transcription_time = time.time() - transcription_start
                
                logger.info(f"Fallback transcription completed in {transcription_time:.2f}s: {transcription}")
                
                # Return the transcription and metadata
                return transcription, {
                    "engine": "speech_recognition",
                    "api": "google",
                    "time_taken": transcription_time
                }
                
            except Exception as e:
                logger.error(f"Error transcribing with SpeechRecognition: {e}")
        
        # If all methods fail, return an error
        error_msg = "All speech recognition methods failed"
        logger.error(error_msg)
        
        return "", {"error": error_msg, "engine": "none"}
    
    
    def start_streaming_transcription(self, 
                                    language: str = "en",
                                    on_partial_transcription: Optional[Callable[[str], None]] = None,
                                    on_final_transcription: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> bool:
        """
        Start streaming transcription using the configured model.
        
        Args:
            language: Language code for transcription (default: "en")
            on_partial_transcription: Callback for partial transcription updates
            on_final_transcription: Callback for final transcription with metadata
            
        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        # Ensure speech recognition is initialized
        if not self.is_initialized and not self._initialize_speech_recognition():
            logger.error("Failed to initialize speech recognition for streaming")
            return False
            
        try:
            # Create streaming transcriber if needed
            if self.streaming_transcriber is None:
                try:
                    self.streaming_transcriber = StreamingTranscriber(
                        model_name=self.model_name,
                        language=language,
                        on_partial_transcription=on_partial_transcription,
                        on_final_transcription=on_final_transcription
                    )
                except TypeError as e:
                    # Handle BaseModel.init() error by capturing it
                    logger.error(f"Error creating StreamingTranscriber: {e}")
                    logger.info("Retrying with positional arguments")
                    # Try again with a simpler approach that uses positional arguments
                    self.streaming_transcriber = StreamingTranscriber(
                        self.model_name,
                        language,
                        on_partial_transcription,
                        on_final_transcription
                    )
            
            # Start streaming
            success = self.streaming_transcriber.start_streaming()
            if success:
                logger.info(f"Streaming transcription started successfully using OpenAI engine")
            else:
                logger.error("Failed to start streaming transcription")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting streaming transcription: {e}")
            return False
    
    def add_streaming_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Add an audio chunk to the streaming transcription.
        
        Args:
            audio_chunk: Raw audio data to process
        """
        if self.streaming_transcriber is not None and self.streaming_transcriber.is_active():
            self.streaming_transcriber.add_audio_chunk(audio_chunk)
    
    def stop_streaming_transcription(self) -> Tuple[str, Dict[str, Any]]:
        """
        Stop streaming transcription and get final results.
        
        Returns:
            Tuple containing:
                - The final transcription text
                - A dictionary with metadata about the transcription
        """
        if self.streaming_transcriber is not None:
            try:
                return self.streaming_transcriber.stop_streaming()
            except Exception as e:
                logger.error(f"Error stopping streaming transcription: {e}")
        
        return "", {"error": "No active streaming transcription", "engine": "none"}
    
    def get_current_streaming_transcription(self) -> str:
        """
        Get the current partial transcription.
        
        Returns:
            str: The current partial transcription text
        """
        if self.streaming_transcriber is not None:
            return self.streaming_transcriber.get_current_transcription()
        return ""
    
    def is_streaming_active(self) -> bool:
        """
        Check if streaming transcription is active.
        
        Returns:
            bool: True if streaming is active, False otherwise
        """
        return (self.streaming_transcriber is not None and 
                self.streaming_transcriber.is_active())
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available speech recognition models.
        
        Returns:
            List of dictionaries containing model information
        """
        # Only OpenAI STT is supported
        return [
            {"name": "Systran/faster-whisper-medium", "engine": "openai", "description": "Default OpenAI STT model"}
        ]
    
    def get_current_model(self) -> Dict[str, Any]:
        """
        Get information about the currently active model.
        
        Returns:
            Dictionary containing information about the current model
        """
        if self.openai_stt is not None:
            return {
                "name": self.model_name,
                "engine": "openai"
            }
        else:
            return {
                "name": "none",
                "engine": "none",
                "error": "No speech recognition model initialized"
            }
    
    def set_model(self, model_name: str, engine: Optional[str] = None) -> bool:
        """
        Set the speech recognition model to use.
        
        Args:
            model_name: The name of the model to use
            engine: The speech recognition engine to use (optional, only 'openai' is supported)
            
        Returns:
            bool: True if the model was set successfully, False otherwise
        """
        # Validate engine if provided
        if engine is not None and engine.lower() != "openai":
            logger.error(f"Invalid engine: {engine}. Only 'openai' is supported.")
            return False
        
        # If the model name is the same and already initialized, no need to reinitialize
        if (model_name == self.model_name and 
            self.is_initialized and 
            self.openai_stt is not None):
            return True
        
        # Update the model name
        self.model_name = model_name
        
        # Reset initialization state
        self.is_initialized = False
        self.openai_stt = None
        
        # Reinitialize with the new model
        return self._initialize_speech_recognition()


# Create a singleton instance for easy import
default_recognizer = SpeechRecognizer()

def transcribe_audio(audio_file_path: str, language: str = "en", include_timestamps: bool = False, detect_speakers: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Transcribe an audio file using the default speech recognizer.
    
    This is a convenience function that uses the default recognizer instance.
    
    Args:
        audio_file_path: Path to the audio file to transcribe
        language: Language code for transcription (default: "en" for English)
        include_timestamps: Whether to include word-level timestamps (default: False)
        detect_speakers: Whether to attempt speaker detection (default: False)
        
    Returns:
        If include_timestamps=False or detect_speakers=False:
            The transcribed text as a string
        If include_timestamps=True or detect_speakers=True:
            Tuple containing:
                - The formatted text with timestamps/speakers
                - A dictionary with metadata and timing information
    """
    transcription, metadata = default_recognizer.transcribe(
        audio_file_path, 
        language=language,
        include_timestamps=include_timestamps,
        detect_speakers=detect_speakers
    )
    
    # Always return both transcription and metadata since server.py expects it
    return transcription, metadata

def initialize_speech_recognition(
    model_name: str = None,
    language: str = None
) -> bool:
    """
    Initialize the default speech recognizer with the specified parameters.
    
    Args:
        model_name: The name of the model to use (default: from env SPEECH_MCP_STT_MODEL)
        language: The language code for transcription (default: from env SPEECH_MCP_STT_LANGUAGE)
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    global default_recognizer
    
    # Get values from environment variables if not provided
    model_name = model_name or os.environ.get('SPEECH_MCP_STT_MODEL')
    language = language or os.environ.get('SPEECH_MCP_STT_LANGUAGE', 'en')
    
    default_recognizer = SpeechRecognizer(model_name=model_name, language=language)
    return default_recognizer.is_initialized

def get_available_models() -> List[Dict[str, Any]]:
    """
    Get a list of available speech recognition models.
    
    Returns:
        List of dictionaries containing model information
    """
    # Only OpenAI STT is supported
    model_name = os.environ.get('SPEECH_MCP_STT_MODEL')
    if not model_name:
        logger.warning("No STT model configured in environment variable SPEECH_MCP_STT_MODEL")
        return []
        
    return [
        {"name": model_name, "engine": "openai", "description": "Default OpenAI STT model"}
    ]

def get_current_model() -> Dict[str, Any]:
    """
    Get information about the currently active model.
    
    Returns:
        Dictionary containing information about the current model
    """
    return default_recognizer.get_current_model()

def start_streaming_transcription(
    language: str = "en",
    on_partial_transcription: Optional[Callable[[str], None]] = None,
    on_final_transcription: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    **kwargs  # Accept additional kwargs for compatibility
) -> bool:
    """
    Start streaming transcription using the default speech recognizer.
    
    Args:
        language: Language code for transcription (default: "en")
        on_partial_transcription: Callback for partial transcription updates
        on_final_transcription: Callback for final transcription with metadata
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        bool: True if streaming started successfully, False otherwise
    """
    return default_recognizer.start_streaming_transcription(
        language=language,
        on_partial_transcription=on_partial_transcription,
        on_final_transcription=on_final_transcription
    )

def add_streaming_audio_chunk(audio_chunk: bytes) -> None:
    """
    Add an audio chunk to the streaming transcription.
    
    Args:
        audio_chunk: Raw audio data to process
    """
    default_recognizer.add_streaming_audio_chunk(audio_chunk)

def stop_streaming_transcription() -> Tuple[str, Dict[str, Any]]:
    """
    Stop streaming transcription and get final results.
    
    Returns:
        Tuple containing:
            - The final transcription text
            - A dictionary with metadata about the transcription
    """
    return default_recognizer.stop_streaming_transcription()

def get_current_streaming_transcription() -> str:
    """
    Get the current partial transcription.
    
    Returns:
        str: The current partial transcription text
    """
    return default_recognizer.get_current_streaming_transcription()

def is_streaming_active() -> bool:
    """
    Check if streaming transcription is active.
    
    Returns:
        bool: True if streaming is active, False otherwise
    """
    return default_recognizer.is_streaming_active()