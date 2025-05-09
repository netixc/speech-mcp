"""
OpenAI Speech-to-Text Adapter for speech_mcp

This adapter provides integration with OpenAI's Whisper API for high-quality
speech-to-text transcription.
"""

import os
import tempfile
import json
import requests
import sys
import io
import wave
import numpy as np
from typing import Optional, Dict, Any, Union, BinaryIO, Tuple

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="stt")

class STTConfig:
    """Configuration for Speech-to-Text"""
    RATE = 16000   # Audio sample rate expected by the STT API
    STT_LANGUAGE = "en"  # Default language for speech recognition
    STT_MODEL = os.environ.get('SPEECH_MCP_STT_MODEL')  # Get from environment
    BASE_URL_STT = os.environ.get('OPENAI_STT_API_BASE_URL')  # Get STT API endpoint from environment
    DEBUG = False  # Enable debug mode

class OpenAISTT:
    """OpenAI Whisper-based speech recognition adapter"""
    
    def __init__(self, api_key: Optional[str] = None, api_base: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize the OpenAI STT adapter
        
        Args:
            api_key: OpenAI API key. Defaults to environment variable OPENAI_API_KEY
            api_base: OpenAI API base URL. Defaults to environment variable OPENAI_API_BASE_URL as fallback
            model: Model name to use. Defaults to environment variable SPEECH_MCP_STT_MODEL
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        
        # Use STT-specific base URL
        self.api_base = api_base or os.environ.get("OPENAI_STT_API_BASE_URL")
        self.model = model or os.environ.get("SPEECH_MCP_STT_MODEL", STTConfig.STT_MODEL)
        
        # Log initial values
        logger.info(f"Initial API base URL: {self.api_base}")
        
        # Check if API base URL is provided
        if not self.api_base:
            logger.error("No API base URL provided in environment variables (OPENAI_STT_API_BASE_URL)")
            self.api_base = None
            
        # Ensure API base is properly formatted if it exists
        if self.api_base:
            # Remove trailing slashes
            self.api_base = self.api_base.rstrip('/')
        
        # Construct the full API URL if base URL exists
        if self.api_base:
            self.api_url = f"{self.api_base}/audio/transcriptions"
            logger.info(f"Using STT API URL: {self.api_url}")
        else:
            self.api_url = None
            logger.error("Cannot construct API URL: base URL is missing")
            
        logger.info(f"Using STT model: {self.model}")
        
        # Check if we should enable debug mode
        if os.environ.get("LOG_LEVEL", "").upper() == "DEBUG":
            STTConfig.DEBUG = True
            logger.info("Debug mode enabled for STT")
    
    def debug_print(self, message: str) -> None:
        """Print debug messages when debug mode is enabled"""
        if STTConfig.DEBUG:
            logger.debug(message)
    
    def transcribe_file(self, audio_file: Union[str, BinaryIO], 
                       language: Optional[str] = None,
                       prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe an audio file using OpenAI's Whisper API
        
        Args:
            audio_file: Path to audio file or file-like object
            language: Language code (e.g., 'en', 'fr', etc.)
            prompt: Optional prompt to guide the transcription
            
        Returns:
            Dict containing the transcription result
        """
        # Prepare request data
        data = {'model': self.model}
        if language:
            data['language'] = language
        if prompt:
            data['prompt'] = prompt
        
        file_to_close = None
        
        try:
            # Handle string file path
            if isinstance(audio_file, str):
                logger.info(f"Transcribing file: {audio_file}")
                # Check if file exists
                if not os.path.exists(audio_file):
                    error_msg = f"Audio file not found: {audio_file}"
                    logger.error(error_msg)
                    return {"error": error_msg}
                
                # Check file size
                file_size = os.path.getsize(audio_file)
                logger.info(f"Audio file size: {file_size} bytes")
                
                with open(audio_file, 'rb') as f:
                    files = {
                        'file': ('audio.wav', f, 'audio/wav')
                    }
                    
                    # Log request details
                    logger.info(f"Making STT API request to: {self.api_url}")
                    self.debug_print(f"With model: {self.model}")
                    self.debug_print(f"API key begins with: {self.api_key[:4] if self.api_key and len(self.api_key) >= 4 else 'None'}")
                    
                    # Make the API request with a timeout
                    try:
                        headers = {}
                        if self.api_key:
                            headers['Authorization'] = f'Bearer {self.api_key}'
                        
                        response = requests.post(
                            self.api_url,
                            headers=headers if headers else None,
                            data=data,
                            files=files,
                            timeout=30  # 30 second timeout
                        )
                        
                        # Check for errors
                        response.raise_for_status()
                        
                        # Log success and response status
                        logger.info(f"STT API request successful: {response.status_code}")
                        
                        # Parse and return the response
                        result = response.json()
                        self.debug_print(f"API response: {result}")
                        return result
                        
                    except requests.exceptions.Timeout:
                        logger.error(f"STT API request timed out after 30 seconds")
                        raise RuntimeError("OpenAI STT API request timed out after 30 seconds")
                    
            # Handle file-like object
            else:
                logger.info("Transcribing in-memory audio data")
                files = {
                    'file': ('audio.wav', audio_file, 'audio/wav')
                }
                
                # Log request details
                logger.info(f"Making STT API request to: {self.api_url}")
                self.debug_print(f"With model: {self.model}")
                self.debug_print(f"API key begins with: {self.api_key[:4] if self.api_key and len(self.api_key) >= 4 else 'None'}")
                
                # Make the API request with a timeout
                try:
                    headers = {}
                    if self.api_key:
                        headers['Authorization'] = f'Bearer {self.api_key}'
                    
                    response = requests.post(
                        self.api_url,
                        headers=headers if headers else None,
                        data=data,
                        files=files,
                        timeout=30  # 30 second timeout
                    )
                    
                    # Check for errors
                    response.raise_for_status()
                    
                    # Log success
                    logger.info(f"STT API request successful: {response.status_code}")
                    
                    # Parse and return the response
                    result = response.json()
                    self.debug_print(f"API response: {result}")
                    return result
                    
                except requests.exceptions.Timeout:
                    logger.error(f"STT API request timed out after 30 seconds")
                    raise RuntimeError("OpenAI STT API request timed out after 30 seconds")
            
        except requests.exceptions.RequestException as e:
            error_msg = f"OpenAI STT API request failed: {str(e)}"
            logger.error(error_msg)
            
            if hasattr(e, 'response') and e.response:
                try:
                    error_data = e.response.json()
                    if 'error' in error_data:
                        error_msg = f"OpenAI STT API error: {error_data['error'].get('message', str(e))}"
                    logger.error(f"Response JSON: {error_data}")
                except Exception as json_error:
                    error_msg = f"OpenAI STT API error: {e.response.text}"
                    logger.error(f"Response text: {e.response.text}")
                    logger.error(f"Failed to parse JSON: {json_error}")
                
                logger.error(f"Response status code: {e.response.status_code}")
            else:
                logger.error("No response object available")
            
            # Log URL and API parameters for debugging
            logger.error(f"Request URL: {self.api_url}")
            logger.error(f"Request model: {self.model}")
            
            return {"error": error_msg}
            
        finally:
            # Clean up file handle if we opened one
            if file_to_close:
                file_to_close.close()
    
    def transcribe_bytes(self, audio_bytes: bytes, 
                        language: Optional[str] = None,
                        prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Transcribe audio from bytes using OpenAI's Whisper API
        
        Args:
            audio_bytes: Audio data as bytes
            language: Language code (e.g., 'en', 'fr', etc.)
            prompt: Optional prompt to guide the transcription
            
        Returns:
            Dict containing the transcription result
        """
        logger.info(f"Transcribing audio from bytes, size: {len(audio_bytes)} bytes")
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_file.write(audio_bytes)
            temp_file.flush()
            
            temp_path = temp_file.name
            logger.info(f"Saved audio to temporary file: {temp_path}")
            
            try:
                return self.transcribe_file(temp_path, language, prompt)
            finally:
                # Clean up the temporary file
                try:
                    os.unlink(temp_path)
                    logger.info(f"Deleted temporary file: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file: {e}")
    
    def get_text_from_result(self, result: Dict[str, Any]) -> str:
        """
        Extract the transcribed text from the API result
        
        Args:
            result: API response from transcribe_file or transcribe_bytes
            
        Returns:
            Transcribed text as string
        """
        if not result:
            logger.warning("Empty result received from STT API")
            return ""
        
        # Check for error
        if "error" in result:
            logger.error(f"Error in STT result: {result['error']}")
            return ""
        
        text = result.get('text', '')
        if text:
            logger.info(f"Transcription result: {text[:50]}{'...' if len(text) > 50 else ''}")
        else:
            logger.warning("Empty transcription text received from STT API")
            
        return text
        
    def save_audio_to_file(self, audio_data: bytes, filename: str = "recorded_audio.wav") -> str:
        """
        Save audio data to a WAV file
        
        Args:
            audio_data: The audio data as bytes
            filename: The filename to save to
            
        Returns:
            The path to the saved file
        """
        try:
            with open(filename, "wb") as f:
                f.write(audio_data)
            logger.info(f"Audio saved to: {filename}")
            return filename
        except Exception as e:
            logger.error(f"Error saving audio file: {e}")
            return ""