"""
Direct TTS adapter for speech-mcp

This adapter provides direct TTS functionality using direct API calls and
audio playback without threading, which is more reliable.
"""

import os
import sys
import tempfile
import subprocess
import requests
import time
import threading
from typing import Optional, List

# Import base adapter class
from speech_mcp.tts_adapters import BaseTTSAdapter

# Import centralized constants
from speech_mcp.constants import ENV_TTS_VOICE, ENV_OPENAI_API_KEY, ENV_OPENAI_TTS_API_BASE

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="direct_tts")

class DirectTTS(BaseTTSAdapter):
    """
    Direct TTS adapter using subprocess for reliable audio playback

    This adapter uses direct API calls and subprocess for playing audio,
    which is more reliable than the threaded approach.
    """

    def __init__(self, voice: str = None, lang_code: str = "en", speed: float = 1.0, model: str = None):
        """
        Initialize the Direct TTS adapter

        Args:
            voice: The voice to use
            lang_code: The language code
            speed: The speaking speed
            model: The TTS model to use
        """
        # Call parent constructor to initialize common attributes
        super().__init__(voice, lang_code, speed)

        # Get settings from environment variables with fallbacks
        self.voice = voice or os.environ.get(ENV_TTS_VOICE,)
        self.model = model or os.environ.get("SPEECH_MCP_TTS_MODEL")
        self.api_key = os.environ.get(ENV_OPENAI_API_KEY)
        self.base_url = os.environ.get(ENV_OPENAI_TTS_API_BASE)

        logger.info(f"Direct TTS Adapter initialized with:")
        logger.info(f"  Voice: {self.voice}")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  API Key: {'[SET]' if self.api_key else '[NOT SET]'}")
        logger.info(f"  Base URL: {self.base_url or '[DEFAULT OpenAI API]'}")

        # Flag indicating if the adapter is ready
        self.is_initialized = True

    def speak(self, text: str) -> bool:
        """
        Speak the given text using a direct approach

        Args:
            text: The text to speak

        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            logger.warning("Empty text provided to speak")
            return False

        logger.info(f"DirectTTS.speak called with text: {text[:50]}{'...' if len(text) > 50 else ''}")

        return self._speak_subprocess(text)

    def _speak_subprocess(self, text: str) -> bool:
        """
        Speak using a subprocess for more reliable playback

        This implementation uses the centralized ProcessManager to ensure proper
        process creation, monitoring, and cleanup.

        Args:
            text: The text to speak

        Returns:
            bool: True if successful, False otherwise
        """
        # Initialize variables that will be used in the function
        script_path = None
        temp_dir = None
        temp_path = None
        use_managers = False

        try:
            # Import the process manager and temp file manager
            try:
                # Use the function-based API to avoid missing dependencies
                from speech_mcp.utils.process_manager import create_process
                from speech_mcp.utils.temp_file_manager import create_temp_file, cleanup_temp_file
                use_managers = True
            except ImportError:
                logger.warning("Process or temp file manager not available, using legacy method")
                use_managers = False

            logger.info(f"Creating script for text: {text[:30]}...")

            # Create a temporary script
            text_escaped = text.replace('"', '\\"').replace("'", "\\'")
            script_content = f'''#!/usr/bin/env python3
import os
import sys
import requests
import tempfile
import subprocess
import time
import signal
import threading

def main():
    # Setup signal handler for graceful termination
    def signal_handler(sig, frame):
        print(f"DirectTTS Process: Received signal {{sig}}, shutting down...")
        sys.exit(0)

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Process audio with proper exception handling and cleanup
    try:
        voice = "{self.voice}"
        text = """{text_escaped}"""

        # Get environment variables with fallbacks
        api_key = "{self.api_key}" or os.environ.get("OPENAI_API_KEY")
        base_url = "{self.base_url}" or os.environ.get("OPENAI_TTS_API_BASE_URL")
        model = "{self.model}" or os.environ.get("SPEECH_MCP_TTS_MODEL")

        print(f"DirectTTS Process: Playing voice {{voice}}")
        print(f"DirectTTS Process: Text length: {{len(text)}} chars")
        print(f"DirectTTS Process: API settings: URL={{base_url}}, model={{model}}")

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

        # Track created temp files for cleanup
        temp_files = []

        # Make request with retry logic
        def make_request(max_retries=2):
            for attempt in range(max_retries + 1):
                try:
                    print(f"DirectTTS Process: Sending request to API (attempt {{attempt+1}}/{{max_retries+1}})...")
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    print(f"DirectTTS Process: Got response from API on attempt {{attempt+1}}")
                    return response
                except Exception as e:
                    print(f"DirectTTS Process: Error on attempt {{attempt+1}}: {{e}}")
                    if attempt < max_retries:
                        retry_delay = (attempt + 1) * 1.0  # Progressive delay
                        print(f"DirectTTS Process: Retrying in {{retry_delay}} seconds...")
                        time.sleep(retry_delay)
                    else:
                        raise

        # Attempt the request with retries
        response = make_request()

        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(response.content)
            temp_files.append(temp_path)

        file_size = os.path.getsize(temp_path)
        print(f"DirectTTS Process: Audio saved to {{temp_path}}, size: {{file_size}} bytes")

        # Validate audio file
        if file_size < 100:
            print("DirectTTS Process: Warning - audio file is suspiciously small")
            if file_size < 10:  # Likely an error
                raise Exception(f"Generated audio file too small: {{file_size}} bytes")

        # Play audio with different methods depending on platform
        if sys.platform == "darwin":  # macOS
            # Use a more reliable approach for macOS
            try:
                # First try using subprocess with error handling
                print("DirectTTS Process: Using subprocess with afplay...")
                result = subprocess.run(["afplay", temp_path], check=False,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Check result
                if result.returncode == 0:
                    print("DirectTTS Process: Audio played successfully with afplay")
                else:
                    error_out = result.stderr.decode() if result.stderr else "Unknown error"
                    print(f"DirectTTS Process: afplay failed ({{result.returncode}}): {{error_out}}")

                    # Try alternative methods
                    print("DirectTTS Process: Trying open command...")
                    open_result = subprocess.run(["open", temp_path], check=False)
                    if open_result.returncode == 0:
                        print("DirectTTS Process: Audio opened with default player")
                        # Wait for playback to complete (approximate)
                        play_duration = file_size / 32000  # Rough estimate: 32KB per second
                        print(f"DirectTTS Process: Waiting {{play_duration:.1f}} seconds for playback...")
                        time.sleep(max(play_duration, 3.0))  # At least 3 seconds
                    else:
                        print(f"DirectTTS Process: open command failed ({{open_result.returncode}})")
                        return 1
            except Exception as play_error:
                print(f"DirectTTS Process: Error playing audio: {{play_error}}")
                return 1

        elif sys.platform == "win32":  # Windows
            try:
                print("DirectTTS Process: Using PowerShell on Windows...")
                # Use subprocess for better error handling
                result = subprocess.run([
                    "powershell",
                    "-c",
                    f"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()"
                ], check=False)

                if result.returncode != 0:
                    print(f"DirectTTS Process: PowerShell playback failed ({{result.returncode}})")
                    return 1
            except Exception as play_error:
                print(f"DirectTTS Process: Error playing audio on Windows: {{play_error}}")
                return 1

        else:  # Linux
            try:
                print("DirectTTS Process: Using aplay on Linux...")
                result = subprocess.run(["aplay", temp_path], check=False)

                if result.returncode != 0:
                    print(f"DirectTTS Process: aplay failed ({{result.returncode}}), trying paplay...")
                    result = subprocess.run(["paplay", temp_path], check=False)

                    if result.returncode != 0:
                        print(f"DirectTTS Process: paplay also failed ({{result.returncode}})")
                        return 1
            except Exception as play_error:
                print(f"DirectTTS Process: Error playing audio on Linux: {{play_error}}")
                return 1

        # Clean up all temp files
        print("DirectTTS Process: Playback complete, cleaning up...")
        cleanup_errors = []
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                cleanup_errors.append(str(e))

        if cleanup_errors:
            print(f"DirectTTS Process: Errors during cleanup: {{cleanup_errors}}")
        else:
            print("DirectTTS Process: All temporary files cleaned up successfully")

        return 0

    except Exception as e:
        print(f"DirectTTS Process: Fatal error: {{e}}")
        return 1
    finally:
        # Ensure we exit cleanly
        print("DirectTTS Process: Exiting")

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
'''
            # Write script to temp file
            if use_managers:
                script_path = create_temp_file(
                    suffix='.py',
                    prefix='tts_script_',
                    component='direct_tts',
                    auto_cleanup_seconds=300  # Auto-cleanup after 5 minutes
                )

                # Write content to the file
                with open(script_path, 'w') as script_file:
                    script_file.write(script_content)
            else:
                # Fallback to traditional method
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                    script_path = script_file.name
                    script_file.write(script_content)

            # Make it executable
            os.chmod(script_path, 0o755)
            logger.info(f"Created temporary script at {script_path}")

            # Create a clean environment with only the necessary variables
            env = os.environ.copy()

            # Launch the process using the process manager
            if use_managers:
                # Define callback for process completion
                def on_process_exit(exit_code, reason):
                    """Handle TTS process exit."""
                    if exit_code == 0:
                        logger.info(f"TTS process completed successfully ({reason})")
                    else:
                        logger.warning(f"TTS process failed with code {exit_code if exit_code is not None else 'unknown'} ({reason})")

                    # Clean up the script file if it exists
                    if os.path.exists(script_path):
                        cleanup_temp_file(script_path)

                # Create the process with the manager
                try:
                    # Use the new ProcessManager class-based API
                    try:
                        from speech_mcp.utils.process_manager import get_process_manager
                        process_manager = get_process_manager()
                        process, error = process_manager.create_process(
                            cmd=[sys.executable, script_path],
                            component='direct_tts',
                            env=env,
                            detached=True,
                            capture_output=True,
                            timeout_seconds=120,  # 2 minute timeout
                            exit_callback=on_process_exit
                        )
                    except (ImportError, AttributeError):
                        # Fall back to function-based API for backward compatibility
                        process, error = create_process(
                            cmd=[sys.executable, script_path],
                            component='direct_tts',
                            env=env,
                            detached=True,
                            capture_output=True,
                            timeout_seconds=120,  # 2 minute timeout
                            exit_callback=on_process_exit
                        )

                    if error:
                        logger.error(f"Error creating TTS process: {error}")
                        if os.path.exists(script_path):
                            cleanup_temp_file(script_path)
                        return False

                    logger.info(f"Launched TTS process with PID {process.pid} via process manager")
                except Exception as create_error:
                    logger.error(f"Error using process manager: {create_error}")
                    # Fall back to legacy method if process manager fails
                    use_managers = False

            # Legacy method if process manager isn't available or fails
            if not use_managers:
                try:
                    process = subprocess.Popen(
                        [sys.executable, script_path],
                        env=env,
                        start_new_session=True,  # Fully detach
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE
                    )

                    logger.info(f"Launched TTS process with PID {process.pid} (legacy method)")

                    # Try to get initial output (non-blocking)
                    try:
                        stdout, stderr = process.communicate(timeout=1)
                        if stdout:
                            logger.info(f"Initial TTS process output: {stdout.decode(errors='replace')[:100]}...")
                        if stderr:
                            logger.warning(f"Initial TTS process error: {stderr.decode(errors='replace')[:100]}...")
                    except subprocess.TimeoutExpired:
                        # This is expected and fine, we don't want to block
                        pass

                    # Legacy cleanup method
                    def delayed_cleanup():
                        try:
                            time.sleep(20)  # Wait 20 seconds before cleanup
                            if os.path.exists(script_path):
                                os.unlink(script_path)
                                logger.info(f"Deleted temporary script: {script_path}")
                        except Exception as e:
                            logger.warning(f"Error in delayed cleanup: {e}")

                    # Start cleanup in a daemon thread
                    cleanup_thread = threading.Thread(target=delayed_cleanup, daemon=True)
                    cleanup_thread.start()

                except Exception as popen_error:
                    logger.error(f"Error launching process (legacy method): {popen_error}")
                    # Clean up the script file
                    if script_path and os.path.exists(script_path):
                        try:
                            os.unlink(script_path)
                        except Exception:
                            pass
                    return False

            return True

        except Exception as e:
            logger.error(f"Error in _speak_subprocess: {e}")

            # Clean up the script file if it exists
            if script_path and os.path.exists(script_path):
                try:
                    if 'cleanup_temp_file' in locals():
                        cleanup_temp_file(script_path)
                    else:
                        os.unlink(script_path)
                except Exception:
                    pass

            return False

    def save_to_file(self, text: str, file_path: str) -> bool:
        """
        Save speech to an audio file

        Args:
            text: The text to convert to speech
            file_path: Path where to save the audio file

        Returns:
            bool: True if successful, False otherwise
        """
        if not text:
            logger.warning("Empty text provided to save_to_file")
            return False

        try:
            logger.info(f"Generating TTS audio for: {text[:50]}...")

            # Set up request
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }
            url = f"{self.base_url}/audio/speech"

            # Request data
            data = {
                "model": self.model,
                "voice": self.voice,
                "input": text,
                "response_format": "wav"
            }

            # Make request
            logger.info(f"Sending request to {url}")
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()

            # Save to file
            with open(file_path, 'wb') as f:
                f.write(response.content)

            logger.info(f"Audio saved to {file_path}, size: {os.path.getsize(file_path)} bytes")
            return True

        except Exception as e:
            logger.error(f"Error in save_to_file: {e}")
            return False

    def get_available_voices(self) -> List[str]:
        """
        Get a list of available voices

        Returns:
            List[str]: List of available voice names
        """
        # Default list of voices
        default_voices = [
            "af_alloy", "af_aoede", "af_bella", "af_heart", "af_jadzia", "af_jessica",
            "af_kore", "af_nicole", "af_nova", "af_river", "af_sarah", "af_sky",
            "af_v0", "af_v0bella", "af_v0irulan", "af_v0nicole", "af_v0sarah", "af_v0sky",
            "am_adam", "am_echo", "am_eric", "am_fenrir", "am_liam", "am_michael",
            "am_onyx", "am_puck", "am_santa", "am_v0adam", "am_v0gurney", "am_v0michael",
            "bf_alice", "bf_emma", "bf_lily", "bf_v0emma", "bf_v0isabella",
            "bm_daniel", "bm_fable", "bm_george", "bm_lewis", "bm_v0george", "bm_v0lewis",
            "ef_dora", "em_alex", "em_santa", "ff_siwis", "hf_alpha", "hf_beta",
            "hm_omega", "hm_psi", "if_sara", "im_nicola", "jf_alpha", "jf_gongitsune",
            "jf_nezumi", "jf_tebukuro", "jm_kumo", "pf_dora", "pm_alex", "pm_santa",
            "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
            "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
        ]

        # Try to get voices from the API
        try:
            if self.base_url:
                voices_url = f"{self.base_url}/audio/voices"
                logger.info(f"Fetching available voices from: {voices_url}")

                headers = {}
                if self.api_key:
                    headers['Authorization'] = f'Bearer {self.api_key}'

                response = requests.get(
                    voices_url,
                    headers=headers if headers else None,
                    timeout=10
                )

                response.raise_for_status()
                result = response.json()

                if 'voices' in result and isinstance(result['voices'], list):
                    voices = result['voices']
                    if voices and all(isinstance(v, str) for v in voices):
                        logger.info(f"Found {len(voices)} voices from API")
                        return voices
        except Exception as e:
            logger.warning(f"Error fetching voices from API: {e}")

        logger.info(f"Using default voices list: {len(default_voices)} voices")
        return default_voices