"""
Direct TTS helper functions for speech-mcp.

This module provides direct, reliable TTS functionality that can be used
across the application, independent of the UI threading.
"""

import os
import sys
import tempfile
import requests
import subprocess
from typing import Optional

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE, ENV_OPENAI_API_KEY, ENV_OPENAI_TTS_API_BASE

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="tts_helper")

def play_tts_direct(text: str, voice: Optional[str] = None, temp_dir: Optional[str] = None) -> bool:
    """
    Play TTS directly using API requests, bypassing the UI thread.
    
    This is the most reliable way to play TTS across platforms.
    
    Args:
        text: The text to speak
        voice: Optional voice to use (falls back to env var if not specified)
        temp_dir: Optional temp directory to use
        
    Returns:
        bool: True if successful, False otherwise
    """
    if not text:
        logger.warning("Empty text provided for TTS")
        return False
    
    try:
        # Get environment variables
        api_key = os.environ.get(ENV_OPENAI_API_KEY)
        base_url = os.environ.get(ENV_OPENAI_TTS_API_BASE)
        model = os.environ.get("SPEECH_MCP_TTS_MODEL")
        
        # Use provided voice or fall back to environment variable
        if not voice:
            voice = os.environ.get(ENV_TTS_VOICE)
        
        logger.info(f"Direct TTS: voice={voice}, model={model}, text={text[:50]}{'...' if len(text) > 50 else ''}")
        
        # Set up headers and API URL
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        speech_url = f"{base_url}/audio/speech"
        
        # Request data
        data = {
            "model": model,
            "voice": voice,
            "input": text,
            "response_format": "wav"
        }
        
        # Make request
        response = requests.post(
            speech_url,
            headers=headers,
            json=data,
            timeout=30  # 30 second timeout
        )
        
        # Check for errors
        response.raise_for_status()
        
        # Save to temp file using the specified temp dir if provided
        if temp_dir:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir=temp_dir)
        else:
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            
        temp_path = temp_file.name
        temp_file.write(response.content)
        temp_file.close()
        
        logger.info(f"TTS audio saved to {temp_path}")
        
        # Play audio based on platform with proper quotation
        if sys.platform == "darwin":  # macOS
            result = os.system(f"afplay \"{temp_path}\"")
            logger.info(f"afplay result: {result}")
        elif sys.platform == "win32":  # Windows
            escaped_path = temp_path.replace("'", "''")
            result = os.system(f"start /min powershell -c \"(New-Object Media.SoundPlayer '{escaped_path}').PlaySync()\"")
            logger.info(f"powershell result: {result}")
        else:  # Linux and others
            result = os.system(f"aplay \"{temp_path}\"")
            logger.info(f"aplay result: {result}")
        
        # Clean up
        try:
            os.unlink(temp_path)
            logger.info("Temporary audio file deleted")
        except Exception as e:
            logger.warning(f"Failed to delete temporary file: {e}")
        
        return True
    except Exception as e:
        logger.error(f"Error in direct TTS: {e}")
        return False

def play_tts_subprocess(text: str, voice: Optional[str] = None) -> bool:
    """
    Play TTS using a completely separate subprocess.
    
    This is the most reliable way to play TTS when the UI thread is busy.
    
    Args:
        text: The text to speak
        voice: Optional voice to use (falls back to env var if not specified)
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Create environment with needed variables
        env = os.environ.copy()
        
        
        # Create command to run
        python_exec = sys.executable
        script_content = f'''#!/usr/bin/env python3
import os
import sys
import tempfile
import requests

def main():
    voice = "{voice}"
    text = sys.argv[1] if len(sys.argv) > 1 else "Voice preference saved. You can now start listening."
    
    # Get environment variables
    api_key = os.environ.get("{ENV_OPENAI_API_KEY}")
    base_url = os.environ.get("{ENV_OPENAI_TTS_API_BASE}")
    model = os.environ.get("SPEECH_MCP_TTS_MODEL")
    
    print(f"Playing voice {{voice}} with text: {{text}}")
    
    # Set up request
    headers = {{
        'Authorization': f'Bearer {{api_key}}',
        'Content-Type': 'application/json'
    }}
    url = f"{{base_url}}/audio/speech"
    
    # Request data
    data = {{
        "model": model,
        "voice": voice,
        "input": text,
        "response_format": "wav"
    }}
    
    # Make request
    response = requests.post(url, headers=headers, json=data, timeout=30)
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(response.content)
    
    # Play audio
    if sys.platform == "darwin":  # macOS
        os.system(f"afplay \\"{{temp_path}}\\"")
    elif sys.platform == "win32":  # Windows
        os.system(f"start /min powershell -c \\"(New-Object Media.SoundPlayer '{{temp_path}}').PlaySync()\\"")
    else:  # Linux
        os.system(f"aplay \\"{{temp_path}}\\"")
    
    # Clean up
    try:
        os.unlink(temp_path)
    except:
        pass

if __name__ == "__main__":
    main()
'''
        
        # Create a temporary script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
            script_path = script_file.name
            script_file.write(script_content)
        
        # Make it executable
        os.chmod(script_path, 0o755)
        
        logger.info(f"Launching TTS subprocess: {script_path} '{text}'")
        
        # Launch the subprocess
        subprocess.Popen(
            [python_exec, script_path, text],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        return True
    except Exception as e:
        logger.error(f"Error launching TTS subprocess: {e}")
        return False