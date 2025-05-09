"""
OpenAI TTS adapter for speech-mcp

This adapter allows the speech-mcp extension to use OpenAI's TTS service.
It provides a high-quality text-to-speech option with multiple voices.

Usage:
    from speech_mcp.tts_adapters.openai_adapter import OpenAITTS
    
    # Initialize the TTS engine
    tts = OpenAITTS()
    
    # Speak text
    tts.speak("Hello, world!")
"""

import os
import sys
import tempfile
import io
import time
import threading
import importlib.util
from typing import Optional, Dict, Any, List

# Import base adapter class
from speech_mcp.tts_adapters import BaseTTSAdapter

# Import voices override if available
try:
    from speech_mcp.tts_adapters.voices_override import VOICES_OVERRIDE
except ImportError:
    VOICES_OVERRIDE = None

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="tts")

class OpenAITTS(BaseTTSAdapter):
    """
    Text-to-speech adapter for OpenAI's TTS service
    
    This class provides an interface to use OpenAI for TTS.
    """
    
    def __init__(self, voice: str = None, lang_code: str = "en", speed: float = 1.0, model: str = None):
        """
        Initialize the OpenAI TTS adapter
        
        Args:
            voice: The voice to use (default from config or "bm_daniel")
            lang_code: The language code (not used by OpenAI but kept for interface compatibility)
            speed: The speaking speed (default: 1.0)
            model: The TTS model to use (default: from environment or "tts-1")
        """
        # Call parent constructor to initialize common attributes
        super().__init__(voice, lang_code, speed)
        
        # Get all values from environment variables with no hardcoded defaults
        self.voice = voice or os.environ.get("SPEECH_MCP_TTS_VOICE")
        self.model = model or os.environ.get("SPEECH_MCP_TTS_MODEL")
        self.openai_client = None
        self.is_initialized = False
        self.api_key = os.environ.get("OPENAI_API_KEY") 
        self.base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
        
        # Add last_error attribute to track most recent error
        self.last_error = None
        
        # Add request tracking and rate limiting parameters
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 0.1  # Minimum seconds between requests
        self.retry_count = 2  # Default retry count for operations
        
        # Track temporary files for cleanup
        self.temp_files = []
        
        # Log the configuration for debugging
        logger.info(f"OpenAI TTS Adapter initialized with:")
        logger.info(f"  Voice: {self.voice}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  API Key: {'[SET]' if self.api_key else '[NOT SET]'}")
        logger.info(f"  Base URL: {self.base_url or '[DEFAULT OpenAI API]'}")
        
        # Try to initialize the OpenAI client
        self._initialize_openai()
    
    def _initialize_openai(self, retry_count: int = 2) -> bool:
        """
        Initialize the OpenAI client with retry logic.
        
        This method attempts to initialize the OpenAI client and test its connectivity.
        It includes retry logic for transient errors and better error categorization.
        
        Args:
            retry_count: Number of retries for initialization (default: 2)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Start with known error state
        self.last_error = None
        
        # Check if already initialized successfully
        if self.is_initialized and self.openai_client:
            logger.debug("OpenAI client already initialized")
            return True
        
        # Check for required OpenAI package
        if importlib.util.find_spec("openai") is None:
            self.last_error = "OpenAI package not installed"
            logger.error(f"{self.last_error}. Install with: pip install openai")
            return False
            
        # Import OpenAI
        try:
            from openai import OpenAI
            
            # Define error classes that might not exist in all OpenAI package versions
            # These are used for type checking but not directly instantiated
            try:
                from openai.types.error import APIError, AuthenticationError, RateLimitError
            except ImportError:
                # Define dummy classes if imports fail
                class APIError(Exception): pass
                class AuthenticationError(Exception): pass
                class RateLimitError(Exception): pass
            
            # The Response type is not critical for functionality
            class Response:
                pass
                
            self.is_initialized = True
        except ImportError as e:
            self.last_error = f"Failed to import OpenAI modules: {str(e)}"
            logger.error(self.last_error)
            return False
            
        # Validate configuration
        try:
            # Handle API key for local or remote endpoints
            if not self.api_key:
                if self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
                    logger.info("Using default API key for local endpoint")
                    self.api_key = "local"
                else:
                    self.last_error = "OpenAI API key not provided"
                    logger.error(f"{self.last_error}. Set OPENAI_API_KEY environment variable.")
                    return False
            
            # Model and voice validation
            if not self.model:
                self.last_error = "TTS model not configured"
                logger.error(f"{self.last_error}. Set SPEECH_MCP_TTS_MODEL environment variable.")
                return False
                
            if not self.voice:
                self.last_error = "TTS voice not configured"
                logger.error(f"{self.last_error}. Set SPEECH_MCP_TTS_VOICE environment variable.")
                return False
                
            logger.info(f"Initializing OpenAI client with base_url: {self.base_url or 'default OpenAI API'}")
        except Exception as e:
            self.last_error = f"Error validating configuration: {str(e)}"
            logger.error(self.last_error)
            return False
            
        # Initialize with retry logic
        for attempt in range(retry_count + 1):  # +1 for initial attempt
            try:
                # Create the client
                if self.base_url:
                    self.openai_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
                else:
                    self.openai_client = OpenAI(api_key=self.api_key)
                
                logger.info(f"OpenAI client created (attempt {attempt+1}/{retry_count+1}), testing connection...")
                
                # Test connection
                try:
                    # Use short timeout and minimal test input
                    logger.debug(f"Testing OpenAI TTS with model: {self.model}, voice: {self.voice}")
                    test_response = self.openai_client.audio.speech.create(
                        model=self.model,
                        voice=self.voice,
                        input="Test",
                    )
                    
                    # Verify response
                    if not test_response or not hasattr(test_response, 'content'):
                        logger.warning("OpenAI TTS test returned incomplete response")
                        # Continue with initialization for local APIs
                        if self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
                            logger.info("Using local API, continuing despite test issues")
                            self.is_initialized = True
                            return True
                        
                        if attempt < retry_count:
                            logger.info(f"Retrying initialization (attempt {attempt+1}/{retry_count})")
                            time.sleep(1)  # Brief delay before retry
                            continue
                        else:
                            self.last_error = "OpenAI TTS test failed - incomplete response"
                            return False
                    
                    # Success case
                    self.is_initialized = True
                    logger.info("OpenAI TTS initialized and tested successfully")
                    return True
                    
                except AuthenticationError as e:
                    # Authentication errors shouldn't be retried
                    self.last_error = f"OpenAI authentication error: {str(e)}"
                    logger.error(self.last_error)
                    return False
                    
                except RateLimitError as e:
                    # Rate limit errors - retry with longer delay
                    logger.warning(f"OpenAI rate limit error: {str(e)}")
                    if attempt < retry_count:
                        delay = (attempt + 1) * 2  # Progressive delay: 2s, 4s
                        logger.info(f"Rate limited, retrying in {delay}s (attempt {attempt+1}/{retry_count})")
                        time.sleep(delay)
                        continue
                    else:
                        self.last_error = f"OpenAI rate limit exceeded: {str(e)}"
                        return False
                        
                except APIError as e:
                    # General API errors - retry for transient issues
                    logger.warning(f"OpenAI API error: {str(e)}")
                    if attempt < retry_count:
                        logger.info(f"Retrying after API error (attempt {attempt+1}/{retry_count})")
                        time.sleep(1)
                        continue
                    else:
                        self.last_error = f"OpenAI API error: {str(e)}"
                        return False
                
                except Exception as e:
                    # Other errors during testing
                    logger.error(f"Error testing OpenAI TTS: {str(e)}")
                    
                    # For local API endpoints, tolerate failures
                    if self.base_url and ("localhost" in self.base_url or "127.0.0.1" in self.base_url):
                        logger.info("Using local API, continuing despite test failure")
                        self.is_initialized = True
                        return True
                        
                    # Retry transient errors
                    if attempt < retry_count:
                        logger.info(f"Retrying initialization (attempt {attempt+1}/{retry_count})")
                        time.sleep(1)
                        continue
                    else:
                        self.last_error = f"OpenAI TTS connection test failed: {str(e)}"
                        return False
            
            except Exception as e:
                # Errors creating the client
                logger.error(f"Error creating OpenAI client (attempt {attempt+1}/{retry_count}): {str(e)}")
                if attempt < retry_count:
                    logger.info(f"Retrying client creation (attempt {attempt+1}/{retry_count})")
                    time.sleep(1)
                    continue
                else:
                    self.last_error = f"Failed to create OpenAI client: {str(e)}"
                    return False
        
        # Should not reach here, but just in case
        logger.error("OpenAI initialization failed after all attempts")
        if not self.last_error:
            self.last_error = "OpenAI initialization failed for unknown reason"
        return False
    
    def speak(self, text: str, retry_count: int = None) -> bool:
        """
        Speak the given text using OpenAI TTS with robust error handling.
        
        This method creates a temporary audio file, uses OpenAI's API to generate
        speech, and then plays the audio. It includes retry logic for API errors
        and cleanup for temporary files.
        
        Args:
            text: The text to speak
            retry_count: Number of retries for API calls (default: use instance default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Reset last error
        self.last_error = None
        
        # Use instance retry count if not specified
        if retry_count is None:
            retry_count = self.retry_count
        
        # Input validation
        if not text or not text.strip():
            self.last_error = "Empty text provided to speak"
            logger.warning(self.last_error)
            return False
        
        # Limit text length for logging
        log_text = text[:50] + ('...' if len(text) > 50 else '')
        logger.info(f"Speaking text ({len(text)} chars): {log_text}")
        
        # Check initialization status
        if not self.is_initialized or not self.openai_client:
            logger.warning("OpenAI TTS not initialized, attempting to initialize")
            if not self._initialize_openai():
                self.last_error = "TTS engine not initialized"
                logger.error(f"{self.last_error}: {self.last_error}")
                return False
        
        # Create a temporary file to store the audio
        temp_audio_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
                
            # Track temp file for cleanup
            self.temp_files.append(temp_audio_path)
            logger.debug(f"Created temporary file at {temp_audio_path}")
            
            # Apply rate limiting if needed
            current_time = time.time()
            elapsed = current_time - self.last_request_time
            if elapsed < self.min_request_interval:
                sleep_time = self.min_request_interval - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
            
            # Generate the audio file with retry logic
            generation_success = False
            for attempt in range(retry_count + 1):  # +1 for initial attempt
                try:
                    # Update request tracking
                    self.request_count += 1
                    self.last_request_time = time.time()
                    
                    logger.debug(f"Generating audio (attempt {attempt+1}/{retry_count+1})")
                    if self.save_to_file(text, temp_audio_path):
                        generation_success = True
                        break
                    
                    # If save_to_file failed but didn't raise an exception
                    logger.warning(f"Failed to generate audio on attempt {attempt+1}/{retry_count+1}")
                    if attempt < retry_count:
                        delay = (attempt + 1) * 0.5  # Progressive delay: 0.5s, 1s, 1.5s
                        logger.info(f"Retrying audio generation in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        self.last_error = "Failed to generate audio file after all attempts"
                        
                except Exception as e:
                    logger.error(f"Error generating audio (attempt {attempt+1}/{retry_count+1}): {e}")
                    if attempt < retry_count:
                        delay = (attempt + 1) * 0.5
                        logger.info(f"Retrying after error in {delay:.1f}s")
                        time.sleep(delay)
                    else:
                        self.last_error = f"Error generating audio: {str(e)}"
            
            # Check if generation was successful
            if not generation_success:
                logger.error(f"Failed to generate audio after {retry_count+1} attempts: {self.last_error}")
                self._cleanup_temp_file(temp_audio_path)
                return False
            
            # Verify the audio file
            if not os.path.exists(temp_audio_path):
                self.last_error = "Generated audio file not found"
                logger.error(self.last_error)
                return False
                
            file_size = os.path.getsize(temp_audio_path)
            if file_size <= 100:  # Less than 100 bytes is likely an error or empty file
                self.last_error = f"Generated audio file too small: {file_size} bytes"
                logger.error(self.last_error)
                self._cleanup_temp_file(temp_audio_path)
                return False
                
            logger.info(f"Audio file generated successfully: {file_size} bytes")
            
            # Play the audio file
            play_start = time.time()
            play_result = self._play_audio_file(temp_audio_path)
            play_duration = time.time() - play_start
            
            if not play_result:
                self.last_error = "Failed to play audio file"
                logger.error(self.last_error)
                self._cleanup_temp_file(temp_audio_path)
                return False
                
            logger.info(f"Audio playback completed in {play_duration:.2f}s")
            
            # Clean up temp file
            self._cleanup_temp_file(temp_audio_path)
            return True
            
        except Exception as e:
            self.last_error = f"Error in speak method: {str(e)}"
            logger.error(self.last_error)
            
            # Clean up temp file if it exists
            if temp_audio_path:
                self._cleanup_temp_file(temp_audio_path)
                
            return False
            
    def _cleanup_temp_file(self, file_path: str) -> None:
        """Safely clean up a temporary file and remove it from tracking."""
        if not file_path:
            return
            
        try:
            # Remove from tracking list first
            if file_path in self.temp_files:
                self.temp_files.remove(file_path)
                
            # Delete if it exists
            if os.path.exists(file_path):
                os.unlink(file_path)
                logger.debug(f"Deleted temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp file {file_path}: {e}")
    
    def save_to_file(self, text: str, file_path: str, retry_count: int = None) -> bool:
        """
        Save speech as an audio file using OpenAI TTS with robust error handling.
        
        This method tries three different approaches in sequence:
        1. Direct API approach using requests
        2. Streaming response with OpenAI client
        3. Non-streaming response with OpenAI client
        
        Args:
            text: The text to convert to speech
            file_path: Path where to save the audio file
            retry_count: Number of retries for API calls (default: use instance default)
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Reset last error
        self.last_error = None
        
        # Use instance retry count if not specified
        if retry_count is None:
            retry_count = self.retry_count
        
        # Input validation
        if not text or not text.strip():
            self.last_error = "Empty text provided to save_to_file"
            logger.warning(self.last_error)
            return False
        
        # Check initialization status
        if not self.is_initialized or not self.openai_client:
            logger.warning("OpenAI TTS not initialized, attempting to initialize")
            if not self._initialize_openai():
                self.last_error = "TTS engine not initialized"
                logger.error(self.last_error)
                return False
        
        # Check if model and voice are configured
        if not self.model or not self.voice:
            self.last_error = "Missing model or voice configuration"
            logger.error(f"Cannot generate TTS audio: {self.last_error}")
            return False
            
        # File path validation
        if not file_path:
            self.last_error = "No file path provided"
            logger.error(self.last_error)
            return False
        
        # Apply rate limiting if needed
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_request_interval:
            sleep_time = self.min_request_interval - elapsed
            logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        # Update request tracking
        self.request_count += 1
        self.last_request_time = time.time()
        
        # Log for debugging
        log_text = text[:50] + ('...' if len(text) > 50 else '')
        logger.info(f"Generating TTS audio for text ({len(text)} chars): {log_text}")
        logger.debug(f"Using model: {self.model}, voice: {self.voice}, path: {file_path}")
        
        # ============== METHOD 1: DIRECT API APPROACH ==============
        try:
            import requests
            from requests.exceptions import RequestException, Timeout, ConnectionError
            
            # Set up headers and API URL
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            
            if not self.base_url:
                self.last_error = "No base URL provided for API request"
                logger.error(self.last_error)
                return False
                
            speech_url = f"{self.base_url}/audio/speech"
            
            # Prepare the request data
            data = {
                "model": self.model,
                "voice": self.voice,
                "speed": self.speed,
                "input": text,
                "response_format": "wav"
            }
            
            logger.debug(f"Sending direct TTS request to {speech_url}")
            
            # Try with retries for transient issues
            for attempt in range(retry_count + 1):
                try:
                    # Explicitly specify timeouts
                    response = requests.post(
                        speech_url,
                        headers=headers,
                        json=data,
                        timeout=(5, 60)  # 5s connect timeout, 60s read timeout
                    )
                    
                    # Check for errors
                    response.raise_for_status()
                    
                    # Create parent directory if it doesn't exist
                    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                    
                    # Save the audio to the file with explicit encoding
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    # Verify the file was written and has content
                    if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
                        logger.info(f"Audio saved successfully ({os.path.getsize(file_path)} bytes)")
                        return True
                    else:
                        logger.warning(f"Generated file exists but may be empty or corrupt: {os.path.getsize(file_path)} bytes")
                        if attempt < retry_count:
                            logger.info(f"Retrying due to suspicious file size (attempt {attempt+1}/{retry_count})")
                            time.sleep(0.5)
                            continue
                        
                except ConnectionError as e:
                    logger.warning(f"Connection error on attempt {attempt+1}: {e}")
                    if attempt < retry_count:
                        time.sleep(1)
                        continue
                    else:
                        self.last_error = f"Connection error: {str(e)}"
                        raise
                        
                except Timeout as e:
                    logger.warning(f"Timeout error on attempt {attempt+1}: {e}")
                    if attempt < retry_count:
                        time.sleep(1)
                        continue
                    else:
                        self.last_error = f"Request timeout: {str(e)}"
                        raise
                        
                except RequestException as e:
                    logger.warning(f"Request error on attempt {attempt+1}: {e}")
                    if attempt < retry_count:
                        time.sleep(1)
                        continue
                    else:
                        self.last_error = f"API request error: {str(e)}"
                        raise
                
                except Exception as e:
                    logger.warning(f"Unexpected error on attempt {attempt+1}: {e}")
                    if attempt < retry_count:
                        time.sleep(1)
                        continue
                    else:
                        self.last_error = f"Error during API request: {str(e)}"
                        raise
        
        except Exception as api_error:
            logger.error(f"Direct API approach failed: {api_error}")
            # Continue to next method - don't return yet
        
        # ============== METHOD 2: STREAMING RESPONSE APPROACH ==============
        try:
            logger.info("Trying streaming response approach...")
            
            with self.openai_client.audio.speech.with_streaming_response.create(
                model=self.model,
                voice=self.voice,
                speed=self.speed,
                input=text,
                response_format="wav"
            ) as response:
                # Create parent directory if it doesn't exist
                os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
                
                # Stream to file
                with open(file_path, 'wb') as f:
                    bytes_written = 0
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                        bytes_written += len(chunk)
                
                # Verify the file has content
                if bytes_written > 100:
                    logger.info(f"Audio saved successfully using streaming method ({bytes_written} bytes)")
                    return True
                else:
                    logger.warning(f"Generated file may be empty or corrupt: {bytes_written} bytes")
                    # Continue to next method
            
        except Exception as streaming_error:
            logger.error(f"Streaming response approach failed: {streaming_error}")
            # Continue to next method - don't return yet
        
        # ============== METHOD 3: NON-STREAMING RESPONSE APPROACH ==============
        try:
            logger.info("Trying non-streaming response approach...")
            
            response = self.openai_client.audio.speech.create(
                model=self.model,
                voice=self.voice,
                speed=self.speed,
                input=text,
                response_format="wav"
            )
            
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
            
            # Save the audio file
            with open(file_path, 'wb') as f:
                f.write(response.content)
            
            # Verify the file has content
            if os.path.exists(file_path) and os.path.getsize(file_path) > 100:
                logger.info(f"Audio saved successfully using non-streaming method ({os.path.getsize(file_path)} bytes)")
                return True
            else:
                logger.error(f"Generated file may be empty or corrupt: {os.path.getsize(file_path)} bytes")
                self.last_error = "Generated file is empty or corrupt"
                return False
                
        except Exception as non_streaming_error:
            logger.error(f"Non-streaming response approach failed: {non_streaming_error}")
            self.last_error = f"All TTS approaches failed: {str(non_streaming_error)}"
            return False
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the TTS adapter.
        
        This removes any temporary files created during TTS operations.
        """
        logger.info("Cleaning up OpenAI TTS adapter resources")
        
        # Clean up all temporary files
        temp_files = self.temp_files.copy()
        self.temp_files.clear()
        
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)
                    logger.debug(f"Cleaned up temporary file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
                
        logger.info(f"Cleaned up {len(temp_files)} temporary files")
    
    def get_available_voices(self) -> List[str]:
        """
        Get a list of available voices from the API or fallback to default list
        
        Returns:
            List[str]: List of available voice names
        """
        # Try to get voices from the API if base_url is set
        if self.base_url:
            try:
                # Create a voices endpoint URL
                voices_url = f"{self.base_url}/audio/voices"
                logger.info(f"Fetching available voices from: {voices_url}")
                
                # Make a request to get available voices
                import requests
                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'
                
                response = requests.get(
                    voices_url,
                    headers=headers if headers else None,
                    timeout=10  # 10 second timeout
                )
                
                # Check for errors
                response.raise_for_status()
                
                # Parse the response
                result = response.json()
                logger.info(f"Voices API response: {result}")
                
                # Direct extraction method (simplest approach that works)
                if 'voices' in result and isinstance(result['voices'], list):
                    voices = result['voices']  # Just use the list directly
                    if voices and all(isinstance(v, str) for v in voices):
                        logger.info(f"Found {len(voices)} voices from API")
                        return voices
                
                # Fallback - try more complex parsing if needed
                try:
                    if 'voices' in result:
                        voices_data = result['voices']
                        voices = []
                        
                        # Handle both string lists and object lists
                        if isinstance(voices_data, list):
                            for voice in voices_data:
                                if isinstance(voice, str):
                                    voices.append(voice)
                                elif isinstance(voice, dict) and 'name' in voice:
                                    voices.append(voice['name'])
                            
                            if voices:
                                logger.info(f"Found {len(voices)} voices from API using fallback parser")
                                return voices
                except Exception as e:
                    logger.warning(f"Error extracting voices from response: {e}", exc_info=False)
                
                logger.warning("Voices API response does not contain usable voices field")
            except Exception as e:
                logger.warning(f"Error fetching voices from API: {e}")
                logger.info("Falling back to default voices list")
        
        # Include default OpenAI TTS voices as fallback
        if VOICES_OVERRIDE is not None:
            return VOICES_OVERRIDE
            
        default_voices = [
            "bm_daniel",
            "alloy",
            "echo",
            "fable",
            "onyx",
            "nova",
            "shimmer"
        ]
        logger.info(f"Using default voices list: {default_voices}")
        return default_voices
    
    def set_voice(self, voice: str) -> bool:
        """
        Set the voice to use
        
        Args:
            voice: The voice to use
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Check if the voice is valid
            if voice not in self.get_available_voices():
                logger.warning(f"Invalid voice: {voice}")
                return False
            
            # Call parent method to update self.voice and save preference
            super().set_voice(voice)
            return True
            
        except Exception as e:
            logger.error(f"Error setting voice: {e}")
            return False
    
    def _play_audio_file(self, file_path: str) -> bool:
        """
        Play an audio file on the platform
        
        Args:
            file_path: Path to the audio file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                logger.error(f"Audio file does not exist: {file_path}")
                return False
                
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                logger.error(f"Audio file is empty: {file_path}")
                return False
                
            logger.info(f"Playing audio file: {file_path} (size: {file_size} bytes)")
            
            # Play audio based on platform with proper quotation
            result = None
            if sys.platform == "darwin":  # macOS
                result = os.system(f"afplay \"{file_path}\"")
                logger.info(f"afplay result: {result}")
            elif sys.platform == "win32":  # Windows
                escaped_path = file_path.replace("'", "''")
                result = os.system(f"start /min powershell -c \"(New-Object Media.SoundPlayer '{escaped_path}').PlaySync()\"")
                logger.info(f"powershell result: {result}")
            else:  # Linux and others
                result = os.system(f"aplay \"{file_path}\"")
                logger.info(f"aplay result: {result}")
            
            # Check if the command executed successfully (0 means success)
            if result != 0:
                logger.warning(f"Audio playback command returned non-zero exit code: {result}")
                # Try alternative player if available
                if sys.platform == "darwin":
                    logger.info("Trying alternative player (play)")
                    alt_result = os.system(f"play \"{file_path}\"")
                    logger.info(f"Alternative player result: {alt_result}")
                    return alt_result == 0
                elif sys.platform != "win32":  # Linux or other unix
                    logger.info("Trying alternative player (mpg123)")
                    alt_result = os.system(f"mpg123 \"{file_path}\"")
                    logger.info(f"Alternative player result: {alt_result}")
                    return alt_result == 0
                    
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error playing audio file: {e}")
            return False