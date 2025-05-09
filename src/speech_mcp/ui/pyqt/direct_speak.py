"""
Direct speaking functionality for PyQt UI.
This exactly matches the test_voice_direct approach which we know works.

This module provides direct TTS functionality that uses a completely separate
Python process to perform text-to-speech operations. This approach isolates
the TTS process from any threading or application issues that might prevent
audio playback in the main application.

IMPORTANT: This module is designed to be loaded and applied at runtime by
patching the PyQtSpeechUI.check_for_responses method. The apply_direct_speak_patch()
function should be called as early as possible in the application startup process.
"""

import os
import sys
import tempfile
import requests
import subprocess
import threading
from PyQt5.QtCore import QTimer

def speak_direct(text, voice, transcription_label=None):
    """
    Speak text using the exact same approach as test_voice_direct.
    This is the method that is confirmed to work.
    
    Args:
        text: Text to speak
        voice: Voice to use
        transcription_label: Optional UI label to update
    """
    # Add TTSConfig for debugging control - needed for environment variable debugging
    class TTSConfig:
        DEBUG = True  # Enable debug output for environment variable tracing
    # Create a completely separate function for thread
    def speak_thread_func():
        try:
            # Get API settings directly
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
            model = os.environ.get("SPEECH_MCP_TTS_MODEL")
            
            print(f"Direct Speak: Using voice {voice} with model {model}")
            print(f"Direct Speak: Text: {text[:50]}..." if len(text) > 50 else text)
            
            # Create a temporary script to handle the TTS completely independently
            script_content = f'''#!/usr/bin/env python3
import os
import sys
import requests
import tempfile

def main():
    voice = "{voice}"
    text = """{text.replace('"', '\\"').replace("'", "\\'")}"""
    
    # Get environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
    model = os.environ.get("SPEECH_MCP_TTS_MODEL")
    
    # Print environment vars for debug
    print("Script Environment Variables:")
    for var in ["OPENAI_API_KEY", "OPENAI_TTS_API_BASE_URL", "SPEECH_MCP_TTS_MODEL", "SPEECH_MCP_TTS_VOICE"]:
        value = os.environ.get(var, "[NOT SET]")
        if var == "OPENAI_API_KEY" and value != "[NOT SET]":
            print(f"  - {{var}}: [SET]")
        else:
            print(f"  - {{var}}: {{value}}")
    
    print(f"TTS Script: Playing voice {{voice}} with text: {{text[:50]}}...")
    
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
    print("TTS Script: Sending request...")
    response = requests.post(url, headers=headers, json=data, timeout=30)
    response.raise_for_status()
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_path = temp_file.name
        temp_file.write(response.content)
    
    print(f"TTS Script: Audio saved to {{temp_path}}, size: {{os.path.getsize(temp_path)}} bytes")
    
    # Play audio using multiple methods for maximum reliability
    if sys.platform == "darwin":  # macOS
        print("TTS Script: Using afplay on macOS...")
        # First try to play with afplay
        os.system(f'afplay "{{temp_path}}"')
        
        # Also try with open as a backup
        print("TTS Script: Also trying open command...")
        os.system(f'open "{{temp_path}}"')
    elif sys.platform == "win32":  # Windows
        print("TTS Script: Using PowerShell on Windows...")
        os.system(f'powershell -c "(New-Object Media.SoundPlayer \\'{{temp_path}}\\').PlaySync()"')
    else:  # Linux
        print("TTS Script: Using aplay on Linux...")
        os.system(f'aplay "{{temp_path}}"')
    
    # Clean up after a delay
    import time
    time.sleep(5)
    try:
        os.unlink(temp_path)
        print("TTS Script: Temp file deleted")
    except:
        pass
    
    print("TTS Script: Completed successfully")

if __name__ == "__main__":
    main()
'''
            
            # Write script to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                script_path = script_file.name
                script_file.write(script_content)
            
            # Make executable
            os.chmod(script_path, 0o755)
            
            print(f"Direct Speak: Created script at {script_path}")
            
            # Launch as a completely independent background process
            env = os.environ.copy()
            
            # Debug: Print key environment variables to verify they're passed correctly
            if TTSConfig.DEBUG:
                print("Direct Speak: Environment variables for subprocess:")
                for key in ["OPENAI_API_KEY", "OPENAI_TTS_API_BASE_URL", "SPEECH_MCP_TTS_MODEL", "SPEECH_MCP_TTS_VOICE"]:
                    value = env.get(key, "[NOT SET]")
                    if key == "OPENAI_API_KEY" and value != "[NOT SET]":
                        print(f"  - {key}: [SET]")
                    else:
                        print(f"  - {key}: {value}")
            
            # Launch subprocess with environment variables
            subprocess.Popen(
                [sys.executable, script_path],
                env=env,
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            print("Direct Speak: Launched external TTS process")
            
            # Set a timer to clean up the script file
            threading.Timer(30.0, lambda: os.unlink(script_path) if os.path.exists(script_path) else None).start()
            
            print("Direct Speak: Thread complete")
            
        except Exception as e:
            print(f"Direct Speak Error: {e}")

    # Start the thread
    thread = threading.Thread(target=speak_thread_func, daemon=True)
    thread.start()
    print("Direct Speak: Started thread")
    
    # Update UI if label is provided
    if transcription_label:
        transcription_label.setText(f"Speaking with voice: {voice}")

# Patch the PyQtSpeechUI class to use this direct speaking method
def apply_direct_speak_patch():
    """Patch the check_for_responses method to use direct_speak at runtime"""
    try:
        from speech_mcp.ui.pyqt.pyqt_ui import PyQtSpeechUI
        
        # Keep a reference to the original method
        original_check_for_responses = PyQtSpeechUI.check_for_responses
        
        # Define patched method
        def patched_check_for_responses(self):
            """Patched check_for_responses that uses direct_speak"""
            from speech_mcp.constants import RESPONSE_FILE
            if os.path.exists(RESPONSE_FILE):
                try:
                    # Read the response
                    with open(RESPONSE_FILE, 'r') as f:
                        response = f.read().strip()
                    
                    # Delete the file immediately to prevent duplicate processing
                    try:
                        os.remove(RESPONSE_FILE)
                    except Exception:
                        pass
                    
                    # If TTS is not ready yet, show a message and return
                    if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
                        self.transcription_label.setText("Response received but TTS not ready yet")
                        return
                    
                    # Display the response text in the transcription label
                    self.transcription_label.setText(f"Agent: {response}")
                    
                    # Create a dedicated animation timer if it doesn't exist
                    if not hasattr(self, 'agent_animation_timer'):
                        self.agent_animation_timer = QTimer(self)
                        self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
                        
                    # Restart the timer to ensure consistent animation
                    if self.agent_animation_timer.isActive():
                        self.agent_animation_timer.stop()
                    self.agent_animation_timer.start(50)  # Update every 50ms
                    
                    # Activate agent visualizer
                    self.set_agent_visualizer_active(True)
                    self.set_user_visualizer_active(False)
                    
                    # IMPORTANT: Use the direct_speak function that matches test_voice_direct
                    if response:
                        # Get the current voice
                        voice = self.tts_adapter.get_current_voice()
                        if not voice:
                            voice = "bm_daniel"  # Default fallback
                        
                        # Use our direct_speak function
                        from speech_mcp.ui.pyqt.direct_speak import speak_direct
                        print(f"Using direct_speak for response with voice: {voice}")
                        speak_direct(response, voice, self.transcription_label)
                    
                except Exception as e:
                    self.transcription_label.setText(f"Error processing response: {str(e)}")
                    QTimer.singleShot(3000, lambda: self.transcription_label.setText("Ready for voice interaction"))
                    print(f"Error in patched check_for_responses: {e}")
                    
                    # Deactivate agent visualizer on error
                    self.set_agent_visualizer_active(False)
                    if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
                        self.agent_animation_timer.stop()
            
            # Also call the original method as a fallback
            original_check_for_responses(self)
        
        # Apply the patch
        PyQtSpeechUI.check_for_responses = patched_check_for_responses
        print("Direct speak patch applied successfully")
        return True
    except Exception as e:
        print(f"Error applying direct speak patch: {e}")
        return False
        
if __name__ == "__main__":
    # Test direct speaking
    speak_direct("This is a test of direct speaking. If you can hear this, direct speaking works correctly.", "bm_daniel")
    print("Test direct speak initiated")