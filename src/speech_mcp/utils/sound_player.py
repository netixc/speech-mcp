"""
Sound player module for speech-mcp.

Provides reliable sound playback independent of UI thread.
"""

import os
import sys
import tempfile
import subprocess
import threading
import time
import shutil
from pathlib import Path

class SoundPlayer:
    """Reliable sound player that works independently of UI thread."""
    
    @staticmethod
    def play_audio_file(audio_file_path, blocking=False):
        """
        Play an audio file using the appropriate audio player for the platform.
        
        Args:
            audio_file_path: Path to the audio file
            blocking: Whether to block until playback is complete
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"SoundPlayer: Playing audio file: {audio_file_path}")
        
        # Verify the file exists and has content
        if not os.path.exists(audio_file_path):
            print(f"SoundPlayer: Audio file does not exist: {audio_file_path}")
            return False
            
        file_size = os.path.getsize(audio_file_path)
        if file_size == 0:
            print(f"SoundPlayer: Audio file is empty: {audio_file_path}")
            return False
            
        print(f"SoundPlayer: Audio file size: {file_size} bytes")
        
        # Create a copy of the file to avoid issues with it being deleted
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"speech_mcp_audio_{int(time.time())}.wav")
        try:
            shutil.copy2(audio_file_path, temp_file)
            print(f"SoundPlayer: Created backup copy at {temp_file}")
        except Exception as e:
            print(f"SoundPlayer: Failed to create backup copy: {e}")
            temp_file = audio_file_path  # Use original file if copy fails
        
        # Play the audio based on platform
        if blocking:
            return SoundPlayer._play_blocking(temp_file)
        else:
            # Start playback in a separate thread
            thread = threading.Thread(
                target=SoundPlayer._play_blocking, 
                args=(temp_file,),
                daemon=True
            )
            thread.start()
            return True
    
    @staticmethod
    def _play_blocking(audio_file_path):
        """Play audio and block until complete. Internal method."""
        try:
            if sys.platform == "darwin":  # macOS
                print(f"SoundPlayer: Using afplay on macOS")
                
                # First, try subprocess.run as it's more robust
                try:
                    proc = subprocess.run(
                        ["afplay", audio_file_path],
                        check=False,
                        capture_output=True,
                        text=True
                    )
                    print(f"SoundPlayer: afplay return code: {proc.returncode}")
                    if proc.stderr:
                        print(f"SoundPlayer: afplay stderr: {proc.stderr}")
                    
                    # If successful, return True
                    if proc.returncode == 0:
                        print("SoundPlayer: afplay successful")
                        return True
                except Exception as e:
                    print(f"SoundPlayer: subprocess.run error: {e}")
                
                # If subprocess failed, try os.system as backup
                try:
                    cmd = f"afplay \"{audio_file_path}\""
                    print(f"SoundPlayer: Trying os.system: {cmd}")
                    result = os.system(cmd)
                    print(f"SoundPlayer: os.system result: {result}")
                    
                    if result == 0:
                        print("SoundPlayer: os.system successful")
                        return True
                except Exception as e:
                    print(f"SoundPlayer: os.system error: {e}")
                
                # If both failed, try open as a last resort
                try:
                    print(f"SoundPlayer: Trying open as last resort")
                    os.system(f"open \"{audio_file_path}\"")
                    # No good way to check if open was successful
                    print(f"SoundPlayer: open command executed")
                    time.sleep(3)  # Sleep to approximate audio playing
                    return True
                except Exception as e:
                    print(f"SoundPlayer: open error: {e}")
                    return False
                
            elif sys.platform == "win32":  # Windows
                try:
                    print(f"SoundPlayer: Using PowerShell Media.SoundPlayer on Windows")
                    cmd = f"powershell -c \"(New-Object Media.SoundPlayer '{audio_file_path}').PlaySync()\""
                    result = subprocess.run(cmd, shell=True, check=False)
                    print(f"SoundPlayer: PowerShell result: {result.returncode}")
                    return result.returncode == 0
                except Exception as e:
                    print(f"SoundPlayer: Windows playback error: {e}")
                    return False
            else:  # Linux and others
                try:
                    print(f"SoundPlayer: Using aplay on Linux")
                    # Try aplay first
                    cmd = ["aplay", audio_file_path]
                    result = subprocess.run(cmd, check=False, capture_output=True)
                    print(f"SoundPlayer: aplay result: {result.returncode}")
                    
                    if result.returncode == 0:
                        return True
                    
                    # Try paplay (PulseAudio) as fallback
                    print(f"SoundPlayer: Trying paplay as fallback")
                    cmd = ["paplay", audio_file_path]
                    result = subprocess.run(cmd, check=False, capture_output=True)
                    print(f"SoundPlayer: paplay result: {result.returncode}")
                    
                    return result.returncode == 0
                except Exception as e:
                    print(f"SoundPlayer: Linux playback error: {e}")
                    return False
        except Exception as e:
            print(f"SoundPlayer: Unexpected error in _play_blocking: {e}")
            return False
        finally:
            # Try to clean up the temp file
            try:
                if os.path.exists(audio_file_path) and "speech_mcp_audio_" in audio_file_path:
                    os.unlink(audio_file_path)
                    print(f"SoundPlayer: Deleted temp file: {audio_file_path}")
            except Exception as e:
                print(f"SoundPlayer: Error deleting temp file: {e}")
    
    @staticmethod
    def play_tts_in_new_process(text, voice, api_key=None, base_url=None, model=None):
        """
        Play TTS by launching a completely separate Python process.
        
        This is the most reliable way to play TTS, as it's completely independent
        of the main application and any threading issues.
        
        Args:
            text: The text to speak
            voice: The voice to use
            api_key: API key (falls back to environment variable)
            base_url: API base URL (falls back to environment variable)
            model: Model to use (falls back to environment variable)
            
        Returns:
            bool: True if the process was launched successfully
        """
        try:
            print(f"SoundPlayer: Playing TTS in new process: voice={voice}, text={text}")
            
            # Create a temporary script
            text_escaped = text.replace('"', '\\"').replace("'", "\\'")
            script_content = f'''#!/usr/bin/env python3
import os
import sys
import requests
import tempfile
import subprocess
import time

def main():
    voice = "{voice}"
    text = """{text_escaped}"""
    
    # Get environment variables with fallbacks
    api_key = "{api_key}" or os.environ.get("OPENAI_API_KEY")
    base_url = "{base_url}" or os.environ.get("OPENAI_TTS_API_BASE_URL")
    model = "{model}" or os.environ.get("SPEECH_MCP_TTS_MODEL")
    
    print(f"TTS Process: Playing voice {{voice}}")
    print(f"TTS Process: Text: {{text[:100]}}")
    print(f"TTS Process: API settings: URL={{base_url}}, model={{model}}")
    
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
    
    try:
        # Make request
        print("TTS Process: Sending request to API...")
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        print("TTS Process: Got response from API")
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(response.content)
        
        print(f"TTS Process: Audio saved to {{temp_path}}, size: {{os.path.getsize(temp_path)}} bytes")
        
        # Play audio
        if sys.platform == "darwin":  # macOS
            print("TTS Process: Using afplay on macOS")
            result = subprocess.run(["afplay", temp_path], check=False)
            print(f"TTS Process: afplay result: {{result.returncode}}")
            
            if result.returncode != 0:
                print("TTS Process: afplay failed, trying os.system")
                result = os.system(f'afplay "{{temp_path}}"')
                print(f"TTS Process: os.system result: {{result}}")
                
                if result != 0:
                    print("TTS Process: Trying open as last resort")
                    os.system(f'open "{{temp_path}}"')
                    time.sleep(5)  # Sleep longer to ensure audio plays
                    
        elif sys.platform == "win32":  # Windows
            print("TTS Process: Using PowerShell on Windows")
            os.system(f'powershell -c "(New-Object Media.SoundPlayer \\'{{temp_path}}\\').PlaySync()"')
        else:  # Linux
            print("TTS Process: Using aplay on Linux")
            os.system(f'aplay "{{temp_path}}"')
        
        # Clean up after a delay to ensure audio has time to play
        print("TTS Process: Sleeping before cleanup")
        time.sleep(10)  # Longer sleep to ensure audio completes
        
        try:
            os.unlink(temp_path)
            print("TTS Process: Temp file deleted")
        except Exception as e:
            print(f"TTS Process: Error deleting temp file: {{e}}")
            
        print("TTS Process: Playback complete")
        
    except Exception as e:
        print(f"TTS Process: Error: {{e}}")
        sys.exit(1)

if __name__ == "__main__":
    main()
'''
            
            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_path = script_file.name
                script_file.write(script_content)
            
            print(f"SoundPlayer: Created temp script: {script_path}")
            
            # Make the script executable
            os.chmod(script_path, 0o755)
            
            # Create a clean environment with only the necessary variables
            env = os.environ.copy()
            
            # Add our variables with defaults if not already set
            if api_key and "OPENAI_API_KEY" not in env:
                env["OPENAI_API_KEY"] = api_key
            if base_url and "OPENAI_TTS_API_BASE_URL" not in env:
                env["OPENAI_TTS_API_BASE_URL"] = base_url
            if model and "SPEECH_MCP_TTS_MODEL" not in env:
                env["SPEECH_MCP_TTS_MODEL"] = model
            
            # Launch the process with no stdout/stderr capture to keep it fully independent
            process = subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                start_new_session=True,  # Fully detach from parent process
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            print(f"SoundPlayer: Launched TTS process (PID: {process.pid})")
            
            # Return immediately - we've successfully launched the process
            return True
            
        except Exception as e:
            print(f"SoundPlayer: Error launching TTS process: {e}")
            return False