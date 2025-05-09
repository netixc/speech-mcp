"""
Main PyQt UI implementation for the Speech UI.

This module provides the main PyQt window for the speech interface.
"""

import os
import sys
import time
import threading
import random
import math
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
    QWidget, QLabel, QComboBox
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal

# Import state manager
from speech_mcp.state_manager import StateManager

# Import centralized constants
from speech_mcp.constants import (
    TRANSCRIPTION_FILE, RESPONSE_FILE, COMMAND_FILE,
    CMD_LISTEN, CMD_SPEAK, CMD_IDLE, CMD_UI_READY, CMD_UI_CLOSED,
    ENV_TTS_VOICE
)

# Import UI components
from speech_mcp.ui.components import (
    AudioVisualizer, 
    AnimatedButton, 
    TTSAdapter, 
    AudioProcessorUI
)

# Import configuration module for voice preferences
from speech_mcp.config import get_env_setting, get_setting, set_setting, set_env_setting

class PyQtSpeechUI(QMainWindow):
    """
    Main speech UI window implemented with PyQt.
    """
    # Signal for when components are fully loaded
    components_ready = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Goose Speech Interface")
        self.resize(500, 300)
        
        # Initialize state manager
        self.state_manager = StateManager.get_instance()
        
        # Set initial loading state
        self.tts_ready = False
        self.stt_ready = False
        self.audio_ready = False
        
        # Add a watchdog timer to ensure UI responsiveness
        self.watchdog_timer = QTimer(self)
        self.watchdog_timer.timeout.connect(self.check_ui_responsiveness)
        self.watchdog_timer.start(5000)  # Check every 5 seconds
        
        # Create UI first (will be in loading state)
        self.setup_ui()
        
        # Create a command file to indicate UI is visible (but not fully ready)
        try:
            with open(COMMAND_FILE, 'w') as f:
                f.write("UI_READY")
        except Exception:
            pass
        
        # Start checking for server commands
        self.command_check_timer = QTimer(self)
        self.command_check_timer.timeout.connect(self.check_for_commands)
        self.command_check_timer.start(100)  # Check every 100ms
        
        # Start checking for response files
        self.response_check_timer = QTimer(self)
        self.response_check_timer.timeout.connect(self.check_for_responses)
        self.response_check_timer.start(100)  # Check every 100ms
        
        # Connect the components_ready signal to update UI
        self.components_ready.connect(self.on_components_ready)
        
        # Initialize components in background threads
        QTimer.singleShot(100, self.initialize_components)
        
    def setup_ui(self):
        """Set up the UI components."""
        # Main widget and layout
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)
        
        # Create a layout for the visualizer labels
        label_layout = QHBoxLayout()
        
        # User label
        user_label = QLabel("User")
        user_label.setAlignment(Qt.AlignCenter)
        user_label.setStyleSheet("""
            font-size: 14px;
            color: #00c8ff;
            font-weight: bold;
        """)
        label_layout.addWidget(user_label, 1)
        
        # Agent label
        agent_label = QLabel("Agent")
        agent_label.setAlignment(Qt.AlignCenter)
        agent_label.setStyleSheet("""
            font-size: 14px;
            color: #00ff64;
            font-weight: bold;
        """)
        label_layout.addWidget(agent_label, 1)
        
        # Add the label layout to the main layout
        main_layout.addLayout(label_layout)
        
        # Create a layout for the visualizers
        visualizer_layout = QHBoxLayout()
        
        # User audio visualizer (blue)
        self.user_visualizer = AudioVisualizer(mode="user", width_factor=1.0)
        visualizer_layout.addWidget(self.user_visualizer, 1)  # Equal ratio
        
        # Agent audio visualizer (green)
        self.agent_visualizer = AudioVisualizer(mode="agent", width_factor=1.0)
        visualizer_layout.addWidget(self.agent_visualizer, 1)  # Equal ratio
        
        # Add the visualizer layout to the main layout
        main_layout.addLayout(visualizer_layout)
        
        # Transcription display
        self.transcription_label = QLabel("Ready for voice interaction")
        self.transcription_label.setAlignment(Qt.AlignCenter)
        self.transcription_label.setWordWrap(True)
        self.transcription_label.setStyleSheet("""
            font-size: 14px;
            color: #ffffff;
            background-color: #2a2a2a;
            border-radius: 5px;
            padding: 10px;
        """)
        main_layout.addWidget(self.transcription_label)
        
        # Voice selection
        voice_layout = QHBoxLayout()
        voice_label = QLabel("Voice:")
        voice_label.setStyleSheet("color: #ffffff;")
        self.voice_combo = QComboBox()
        self.voice_combo.setStyleSheet("""
            background-color: #2a2a2a;
            color: #ffffff;
            border: 1px solid #3a3a3a;
            border-radius: 3px;
            padding: 5px;
        """)
        
        # Add loading placeholder
        self.voice_combo.addItem("Loading voices...")
        self.voice_combo.setEnabled(False)
        self.voice_combo.currentIndexChanged.connect(self.on_voice_changed)
        
        voice_layout.addWidget(voice_label)
        voice_layout.addWidget(self.voice_combo, 1)  # 1 = stretch factor
        main_layout.addLayout(voice_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        # Add Select Voice button
        self.select_voice_button = AnimatedButton("Save Voice")
        self.select_voice_button.clicked.connect(self.save_selected_voice)
        self.select_voice_button.setEnabled(True)
        self.select_voice_button.setMinimumWidth(120)
        self.select_voice_button.set_style("""
            background-color: #9b59b6;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Use AnimatedButton for Test Voice button
        self.speak_button = AnimatedButton("Test Voice")
        self.speak_button.clicked.connect(self.test_voice)
        self.speak_button.setEnabled(True)
        self.speak_button.setMinimumWidth(120)
        self.speak_button.set_style("""
            background-color: #27ae60;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Use AnimatedButton for Close button
        self.close_button = AnimatedButton("Close")
        self.close_button.clicked.connect(self.close)
        self.close_button.setMinimumWidth(120)
        self.close_button.set_style("""
            background-color: #e74c3c;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """)
        
        # Add buttons to layout with equal spacing
        button_layout.addStretch(1)
        button_layout.addWidget(self.select_voice_button)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.speak_button)
        button_layout.addSpacing(10)
        button_layout.addWidget(self.close_button)
        button_layout.addStretch(1)
        
        main_layout.addLayout(button_layout)
        
        # Set the main widget
        self.setCentralWidget(main_widget)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #121212;
                color: #ffffff;
            }
            QLabel {
                color: #ffffff;
            }
        """)
        
        # Initialize visualizers to inactive state
        self.set_user_visualizer_active(False)
        self.set_agent_visualizer_active(False)
    
    def set_user_visualizer_active(self, active):
        """Set the user visualizer as active or inactive."""
        self.user_visualizer.set_active(active)
    
    def set_agent_visualizer_active(self, active):
        """Set the agent visualizer as active or inactive."""
        self.agent_visualizer.set_active(active)
    
    def update_voice_list(self):
        """Update the voice selection combo box"""
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            return
        
        # Save current selection text if any
        current_text = self.voice_combo.currentText() if self.voice_combo.currentIndex() >= 0 else None
        
        # Clear the combo box
        self.voice_combo.clear()
        
        # STRICTLY HARDCODED VOICES - don't try to fetch from server
        # These are directly provided and guaranteed to work
        voices = [
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
        print(f"Using strictly hardcoded voices: {len(voices)} voices")
        
        # Set a default voice if none is set
        current_voice = "bm_daniel"  # Default fallback
        
        # Try to get current voice from adapter if possible
        try:
            adapter_voice = self.tts_adapter.get_current_voice()
            if adapter_voice:
                current_voice = adapter_voice
                print(f"Current voice from adapter: {current_voice}")
        except Exception as e:
            print(f"Error getting current voice, using default: {e}")
            
        # Add all available voices to the combo box
        selected_index = 0
        found_current = False
        
        for i, voice in enumerate(voices):
            # Add the voice name directly
            self.voice_combo.addItem(voice, voice)
            
            # Select the current voice if it matches
            if voice == current_voice:
                selected_index = i
                found_current = True
        
        # Enable the combo box now that it has real data
        self.voice_combo.setEnabled(True)
        
        # Set the current selection
        if found_current:
            self.voice_combo.setCurrentIndex(selected_index)
        elif self.voice_combo.count() > 0:
            # Default to first voice
            self.voice_combo.setCurrentIndex(0)
        
        # IMPORTANT: Force the available voices in the TTS adapter to use our hardcoded list
        # This ensures that any attempt to use an API URL for voices is bypassed
        if hasattr(self.tts_adapter, 'available_voices'):
            self.tts_adapter.available_voices = voices
        
        # Set the current voice in the adapter
        if hasattr(self.tts_adapter, 'current_voice'):
            self.tts_adapter.current_voice = voices[selected_index]
        
        # Set status message
        try:
            self.transcription_label.setText(f"Loaded {len(voices)} voices. Select a voice and click 'Test Voice'.")
        except Exception as e:
            print(f"Error updating transcription label: {e}")
    
    def initialize_components(self):
        """Initialize components in background threads"""
        # Start background threads for initialization
        threading.Thread(target=self.initialize_audio_processor, daemon=True).start()
        threading.Thread(target=self.initialize_tts_adapter, daemon=True).start()
    
    def initialize_audio_processor(self):
        """Initialize audio processor in background thread"""
        try:
            self.audio_processor = AudioProcessorUI()
            self.audio_processor.audio_level_updated.connect(self.update_audio_level)
            self.audio_processor.transcription_ready.connect(self.handle_transcription)
            self.audio_ready = True
            self.check_all_components_ready()
        except Exception:
            pass
    
    def initialize_tts_adapter(self):
        """Initialize TTS adapter in background thread"""
        try:
            self.tts_adapter = TTSAdapter()
            self.tts_adapter.speaking_started.connect(self.on_speaking_started)
            self.tts_adapter.speaking_finished.connect(self.on_speaking_finished)
            
            # Connect audio level signal to agent visualizer
            self.tts_adapter.audio_level.connect(self.update_agent_audio_level)
            
            # Create audio level timer if it doesn't exist yet
            if not hasattr(self.tts_adapter, 'audio_level_timer'):
                self.tts_adapter.audio_level_timer = QTimer()
                self.tts_adapter.audio_level_timer.timeout.connect(self.tts_adapter.emit_audio_level)
            
            self.tts_ready = True
            
            # Update voice list when TTS is ready - use QTimer to call from main thread
            QTimer.singleShot(0, self.update_voice_list)
            
            self.check_all_components_ready()
        except Exception:
            pass
    
    def check_all_components_ready(self):
        """Check if all components are ready and emit signal if they are"""
        if self.audio_ready and self.tts_ready:
            # Use QTimer to safely emit signal from background thread
            QTimer.singleShot(0, lambda: self.components_ready.emit())
    
    def on_components_ready(self):
        """Called when all components are ready"""
        # Clear initialization message from transcription label
        self.transcription_label.setText("Ready for voice interaction")
        
        # Check for any pending commands
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                    if command == "LISTEN" and self.has_saved_voice_preference():
                        # Start listening since we have a saved voice preference
                        self.start_listening()
            except Exception:
                pass
        
        # If no voice preference is saved, show guidance message
        if not self.has_saved_voice_preference():
            self.transcription_label.setText("Please select a voice from the dropdown and click 'Save Voice' to continue")
            # Wait a moment before speaking to ensure UI is fully ready
            QTimer.singleShot(500, self.play_guidance_message)
    
    def has_saved_voice_preference(self):
        """Check if a voice preference has been saved"""
        try:
            # First check environment variable
            env_voice = get_env_setting(ENV_TTS_VOICE)
            if env_voice:
                return True
                
            # Then check config file
            config_voice = get_setting("tts", "voice", None)
            if config_voice:
                return True
                
            return False
        except ImportError:
            return False
        except Exception:
            return False
    
    def save_voice_preference(self, voice):
        """Save the selected voice preference to config"""
        try:
            # Save to config file
            result = set_setting("tts", "voice", voice)
            
            # Also set environment variable for current session
            set_env_setting(ENV_TTS_VOICE, voice)
            
            return result
        except ImportError:
            return False
        except Exception:
            return False
    
    def save_selected_voice(self):
        """Save the selected voice and switch to listen mode"""
        # Get the currently selected voice
        index = self.voice_combo.currentIndex()
        if index < 0:
            self.transcription_label.setText("Please select a voice from the dropdown")
            return
        
        voice = self.voice_combo.itemData(index)
        if not voice:
            self.transcription_label.setText("Please select a valid voice from the dropdown")
            return
        
        print(f"Saving voice preference: {voice}")
        
        # Save the voice preference
        if self.save_voice_preference(voice):
            self.transcription_label.setText(f"Voice '{voice}' saved as your preference")
            
            # Create a UI_READY command file to signal back to the server
            try:
                with open(COMMAND_FILE, 'w') as f:
                    f.write(CMD_UI_READY)
            except Exception as e:
                print(f"Error writing command file: {e}")
            
            print(f"Voice saved, now playing confirmation message with voice: {voice}")
            
            # Set up animation for visual feedback
            if not hasattr(self, 'agent_animation_timer'):
                self.agent_animation_timer = QTimer(self)
                self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
                
            if self.agent_animation_timer.isActive():
                self.agent_animation_timer.stop()
            self.agent_animation_timer.start(50)
            
            self.set_agent_visualizer_active(True)
            self.set_user_visualizer_active(False)
            
            # Use our standalone sound player - completely isolated from the UI
            try:
                from speech_mcp.utils.sound_player import SoundPlayer
                
                # Message to speak
                message = "Voice preference saved. You can now start listening."
                
                # Get API settings from environment
                api_key = os.environ.get("OPENAI_API_KEY")
                base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
                model = os.environ.get("SPEECH_MCP_TTS_MODEL")
                
                # Launch TTS in completely separate process
                print(f"Using SoundPlayer to play confirmation with voice {voice}")
                success = SoundPlayer.play_tts_in_new_process(
                    text=message,
                    voice=voice,
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
                
                if success:
                    print("Sound player process launched successfully")
                else:
                    print("Failed to launch sound player process")
                
                # Just to make sure the UI shows success regardless of whether we hear audio
                self.transcription_label.setText(f"Voice '{voice}' saved. Start listening.")
                
                # Stop the animation after a delay (allow time for the sound to play)
                QTimer.singleShot(5000, lambda: self.set_agent_visualizer_active(False))
                QTimer.singleShot(5000, lambda: self.agent_animation_timer.stop() if self.agent_animation_timer.isActive() else None)
            
            except ImportError:
                print("SoundPlayer not available, falling back to direct method")
                
                # DIRECT METHOD - as fallback
                try:
                    import tempfile
                    import requests
                    
                    # Get API settings directly
                    api_key = os.environ.get("OPENAI_API_KEY")
                    base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
                    model = os.environ.get("SPEECH_MCP_TTS_MODEL")
                    
                    # Confirmation message
                    message = "Voice preference saved. You can now start listening."
                    print(f"Playing confirmation message with voice: {voice}")
                    
                    # Set up request to directly call the API
                    headers = {
                        'Authorization': f'Bearer {api_key}',
                        'Content-Type': 'application/json'
                    }
                    url = f"{base_url}/audio/speech"
                    
                    # Request data
                    data = {
                        "model": model,
                        "voice": voice,
                        "input": message,
                        "response_format": "wav"
                    }
                    
                    # Log what we're doing
                    print(f"Sending TTS request to {url} with voice={voice}")
                    self.transcription_label.setText(f"Generating speech with voice: {voice}...")
                    
                    # Make the API request
                    response = requests.post(url, headers=headers, json=data, timeout=30)
                    response.raise_for_status()
                    
                    # Save the audio to a temp file for playing
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                        temp_path = temp_file.name
                        temp_file.write(response.content)
                    
                    print(f"Audio file saved to {temp_path}, file size: {os.path.getsize(temp_path)}")
                    self.transcription_label.setText(f"Playing voice: {voice}")
                    
                    # Play using whatever method is available
                    import subprocess
                    
                    # Special case for macOS
                    import sys
                    if sys.platform == "darwin":
                        # Try to avoid UI freezing by using os.system directly
                        os.system(f"afplay \"{temp_path}\" &")
                        print("Launched afplay in background")
                        
                        # Also try with open command as backup
                        os.system(f"open \"{temp_path}\"")
                        print("Also tried 'open' command")
                    else:
                        # For other platforms, use the subprocess module
                        if sys.platform == "win32":
                            cmd = f"start /min powershell -c \"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()\""
                            subprocess.Popen(cmd, shell=True)
                        else:
                            subprocess.Popen(["aplay", temp_path])
                    
                    # Set a delayed delete
                    QTimer.singleShot(10000, lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)
                    
                except Exception as e:
                    print(f"ERROR in direct TTS method: {e}")
            
            # Regardless of sound issues, show success and stop animation
            self.transcription_label.setText(f"Voice '{voice}' saved. Start listening.")
            QTimer.singleShot(3000, lambda: self.set_agent_visualizer_active(False))
            QTimer.singleShot(3000, lambda: self.agent_animation_timer.stop() if self.agent_animation_timer.isActive() else None)
        else:
            self.transcription_label.setText("Failed to save voice preference. Please try again.")
    
    def test_save_voice_directly(self, voice):
        """Directly test saving voice without any delay or threading"""
        try:
            import os
            import requests
            import tempfile
            import sys
            
            # Get environment variables
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")
            base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
            model = os.environ.get("SPEECH_MCP_TTS_MODEL")
            
            message = "Voice preference saved. You can now start listening."
            print(f"Playing confirmation with voice {voice}: {message}")
            
            # Set up animation
            if not hasattr(self, 'agent_animation_timer'):
                self.agent_animation_timer = QTimer(self)
                self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
                
            if self.agent_animation_timer.isActive():
                self.agent_animation_timer.stop()
            self.agent_animation_timer.start(50)
            
            self.set_agent_visualizer_active(True)
            self.set_user_visualizer_active(False)
            
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
                "input": message,
                "response_format": "wav"
            }
            
            print(f"Sending TTS request to {url} with voice={voice}")
            self.transcription_label.setText(f"Generating speech with voice: {voice}...")
            
            # Make request
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(response.content)
            
            print(f"TTS response saved to {temp_path}, now playing")
            self.transcription_label.setText(f"Playing voice: {voice}")
            
            # Play audio
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay \"{temp_path}\"")
            elif sys.platform == "win32":  # Windows
                os.system(f"start /min powershell -c \"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()\"")
            else:  # Linux
                os.system(f"aplay \"{temp_path}\"")
            
            # Clean up
            try:
                os.unlink(temp_path)
            except Exception as e:
                print(f"Error deleting temp file: {e}")
            
            # Update UI
            self.transcription_label.setText(f"Voice '{voice}' saved as your preference")
            
            # Stop animation
            self.set_agent_visualizer_active(False)
            if self.agent_animation_timer.isActive():
                self.agent_animation_timer.stop()
                
        except Exception as e:
            print(f"SAVE VOICE PLAY ERROR: {e}")
            self.transcription_label.setText(f"Error: {str(e)}")
            
            # Stop animation on error
            self.set_agent_visualizer_active(False)
            if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
                self.agent_animation_timer.stop()
    
    def play_guidance_message(self):
        """Play a guidance message for first-time users"""
        print("Playing guidance message to instruct user to select a voice")
        
        # Add a highlight effect to the Select Voice button
        original_style = self.select_voice_button.styleSheet()
        highlight_style = """
            background-color: #e74c3c;
            color: white;
            border: 2px solid #f39c12;
            border-radius: 5px;
            padding: 8px 16px;
            font-weight: bold;
        """
        self.select_voice_button.setStyleSheet(highlight_style)
        
        # Use a default voice for the guidance message
        default_voice = "bm_daniel"
        
        # Use direct approach for more reliable playback
        if self.voice_combo.count() > 0:
            # Use first voice in the list if available
            default_voice = self.voice_combo.itemData(0)
            print(f"Using first available voice: {default_voice}")
        else:
            print(f"No voices in combo box, using default: {default_voice}")
        
        # Use our standalone sound player - completely isolated from the UI
        try:
            from speech_mcp.utils.sound_player import SoundPlayer
            
            # Message to speak
            message = "Please select a voice from the dropdown menu and click Save Voice to continue."
            
            # Get API settings from environment
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")
            base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
            model = os.environ.get("SPEECH_MCP_TTS_MODEL")
            
            # Launch TTS in completely separate process
            print(f"Using SoundPlayer to play guidance with voice {default_voice}")
            success = SoundPlayer.play_tts_in_new_process(
                text=message,
                voice=default_voice,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            
            if success:
                print("Guidance sound player process launched successfully")
            else:
                print("Failed to launch guidance sound player process")
                # Try fallback method
                raise ImportError("Sound player launch failed")
                
        except ImportError as e:
            print(f"SoundPlayer not available for guidance: {e}, using fallback")
            
            # FALLBACK METHOD - MacOS specific for the guidance message
            try:
                import tempfile
                import requests
                import sys
                import subprocess
                
                # Guidance message
                message = "Please select a voice from the dropdown menu and click Save Voice to continue."
                
                # Create a temporary script to handle the TTS
                script_content = f'''#!/usr/bin/env python3
import os
import sys
import requests
import tempfile

def main():
    voice = "{default_voice}"
    text = "Please select a voice from the dropdown menu and click Save Voice to continue."
    
    # Get environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
    model = os.environ.get("SPEECH_MCP_TTS_MODEL")
    
    print(f"Guidance script: Playing voice {{voice}} with text: {{text}}")
    
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
        os.system(f'afplay "{{temp_path}}"')
        os.system(f'open "{{temp_path}}"')
    elif sys.platform == "win32":  # Windows
        os.system(f'powershell -c "(New-Object Media.SoundPlayer \\\'{{temp_path}}\\\').PlaySync()"')
    else:  # Linux
        os.system(f'aplay "{{temp_path}}"')
    
    # Clean up after a delay
    import time
    time.sleep(5)
    try:
        os.unlink(temp_path)
    except:
        pass

if __name__ == "__main__":
    main()
'''
                
                # Write script to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                    script_path = script_file.name
                    script_file.write(script_content)
                
                # Make executable
                os.chmod(script_path, 0o755)
                
                # Launch as a background process
                env = os.environ.copy()
                subprocess.Popen(
                    [sys.executable, script_path],
                    env=env,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                print(f"Launched guidance background process")
                
            except Exception as e:
                print(f"ERROR in fallback guidance method: {e}")
        
        # Restore the original style after a delay
        QTimer.singleShot(3000, lambda: self.select_voice_button.setStyleSheet(original_style))
    
    
    def on_voice_changed(self, index):
        """Handle voice selection change"""
        # Skip if TTS adapter is not ready yet
        if not hasattr(self, 'tts_adapter') or not self.tts_adapter:
            return
            
        if index < 0:
            return
        
        voice = self.voice_combo.itemData(index)
        if not voice:
            return
        
        self.tts_adapter.set_voice(voice)
    
    def test_voice_direct(self):
        """Test voice directly using requests library (more reliable)"""
        # Get the selected voice
        index = self.voice_combo.currentIndex()
        if index < 0:
            self.transcription_label.setText("Please select a voice first")
            return
            
        voice = self.voice_combo.itemData(index)
        if not voice:
            self.transcription_label.setText("Invalid voice selected")
            return
            
        self.transcription_label.setText(f"Testing voice {voice}...")
        
        # Set up animation
        if not hasattr(self, 'agent_animation_timer'):
            self.agent_animation_timer = QTimer(self)
            self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
            
        if self.agent_animation_timer.isActive():
            self.agent_animation_timer.stop()
        self.agent_animation_timer.start(50)
        
        self.set_agent_visualizer_active(True)
        self.set_user_visualizer_active(False)
        
        # Use our standalone sound player - completely isolated from the UI
        try:
            from speech_mcp.utils.sound_player import SoundPlayer
            
            # Message to speak
            message = "This is a test of the selected voice. Hello, I am Goose!"
            
            # Get API settings from environment
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
            model = os.environ.get("SPEECH_MCP_TTS_MODEL")
            
            # Launch TTS in completely separate process
            print(f"Using SoundPlayer to test voice {voice}")
            success = SoundPlayer.play_tts_in_new_process(
                text=message,
                voice=voice,
                api_key=api_key,
                base_url=base_url,
                model=model
            )
            
            if success:
                print("Test voice sound player process launched successfully")
                # Update UI to show success
                self.transcription_label.setText(f"Playing test of voice: {voice}")
                # Set a timer to update the UI after a delay
                QTimer.singleShot(5000, lambda: self.transcription_label.setText(f"Voice tested successfully: {voice}"))
            else:
                print("Failed to launch test voice sound player process")
                self.transcription_label.setText(f"Failed to test voice {voice}")
            
            # Stop the animation after a delay (allow time for the sound to play)
            QTimer.singleShot(5000, lambda: self.set_agent_visualizer_active(False))
            QTimer.singleShot(5000, lambda: self.agent_animation_timer.stop() if self.agent_animation_timer.isActive() else None)
            
            # Return early - we're done
            return
            
        except ImportError as e:
            print(f"SoundPlayer not available: {e}, falling back to direct method")
        
        # FALLBACK DIRECT METHOD - if SoundPlayer is not available
        try:
            import tempfile
            import requests
            import sys
            import subprocess
            
            # Get API settings directly
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
            model = os.environ.get("SPEECH_MCP_TTS_MODEL")
            
            # Test message
            message = "This is a test of the selected voice. Hello, I am Goose!"
            print(f"Testing voice using fallback method: {voice}")
            
            # Log what we're doing
            self.transcription_label.setText(f"Generating speech with voice: {voice}...")
            
            # Special case for macOS - use an asynchronous approach
            if sys.platform == "darwin":
                # Create a temporary script to handle the TTS
                script_content = f'''#!/usr/bin/env python3
import os
import sys
import requests
import tempfile

def main():
    voice = "{voice}"
    text = "This is a test of the selected voice. Hello, I am Goose!"
    
    # Get environment variables
    api_key = os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
    model = os.environ.get("SPEECH_MCP_TTS_MODEL")
    
    print(f"Test script: Playing voice {{voice}} with text: {{text}}")
    
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
    os.system(f'afplay "{{temp_path}}"')
    
    # Also try with open as a backup
    os.system(f'open "{{temp_path}}"')
    
    # Clean up after a delay
    import time
    time.sleep(5)
    try:
        os.unlink(temp_path)
    except:
        pass

if __name__ == "__main__":
    main()
'''
                
                # Write script to temp file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
                    script_path = script_file.name
                    script_file.write(script_content)
                
                # Make executable
                os.chmod(script_path, 0o755)
                
                # Launch as a background process
                env = os.environ.copy()
                subprocess.Popen(
                    [sys.executable, script_path],
                    env=env,
                    start_new_session=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                print(f"Launched test voice background process")
                self.transcription_label.setText(f"Playing test of voice: {voice}")
                
            else:
                # Other platforms - use the standard approach
                # Set up request to directly call the API
                headers = {
                    'Authorization': f'Bearer {api_key}',
                    'Content-Type': 'application/json'
                }
                url = f"{base_url}/audio/speech"
                
                # Request data
                data = {
                    "model": model,
                    "voice": voice,
                    "input": message,
                    "response_format": "wav"
                }
                
                # Make the API request
                response = requests.post(url, headers=headers, json=data, timeout=30)
                response.raise_for_status()
                
                # Save the audio to a temp file for playing
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    temp_file.write(response.content)
                
                # Play using platform-specific method
                if sys.platform == "win32":  # Windows
                    cmd = f"start /min powershell -c \"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()\""
                    subprocess.Popen(cmd, shell=True)
                else:  # Linux
                    subprocess.Popen(["aplay", temp_path])
                
                # Set a delayed delete
                QTimer.singleShot(10000, lambda: os.unlink(temp_path) if os.path.exists(temp_path) else None)
            
            # Update UI after a delay
            QTimer.singleShot(5000, lambda: self.transcription_label.setText(f"Voice tested successfully: {voice}"))
            
        except Exception as e:
            print(f"ERROR in fallback TTS method: {e}")
            self.transcription_label.setText(f"Error testing voice: {str(e)}")
        
        # Stop the animation after a delay regardless of success/failure
        QTimer.singleShot(5000, lambda: self.set_agent_visualizer_active(False))
        QTimer.singleShot(5000, lambda: self.agent_animation_timer.stop() if self.agent_animation_timer.isActive() else None)
    
    def _test_voice_direct_thread(self, voice, custom_text=None):
        """Run direct voice test in background thread"""
        try:
            import os
            import requests
            import tempfile
            
            # Get environment variables
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
            model = os.environ.get("SPEECH_MCP_TTS_MODEL")
            
            # Default text if not provided
            if custom_text is None:
                custom_text = "This is a test of the selected voice. Hello, I am Goose!"
            
            # Update UI
            QTimer.singleShot(0, lambda: self.transcription_label.setText(f"Generating speech with voice: {voice}..."))
            
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
                "input": custom_text,
                "response_format": "wav"
            }
            
            # Make request
            response = requests.post(url, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(response.content)
            
            # Update UI
            QTimer.singleShot(0, lambda: self.transcription_label.setText(f"Playing voice: {voice}"))
            
            # Play audio
            import sys
            if sys.platform == "darwin":  # macOS
                os.system(f"afplay \"{temp_path}\"")
            elif sys.platform == "win32":  # Windows
                os.system(f"start /min powershell -c \"(New-Object Media.SoundPlayer '{temp_path}').PlaySync()\"")
            else:  # Linux
                os.system(f"aplay \"{temp_path}\"")
            
            # Clean up
            try:
                os.unlink(temp_path)
            except:
                pass
            
            # Update UI - use different messages for test vs confirmation
            if "Voice preference saved" in custom_text:
                QTimer.singleShot(0, lambda: self.transcription_label.setText(f"Voice '{voice}' saved as your preference"))
                QTimer.singleShot(2000, lambda: self.transcription_label.setText("Ready for voice interaction"))
            else:
                QTimer.singleShot(0, lambda: self.transcription_label.setText(f"Voice tested successfully: {voice}"))
                QTimer.singleShot(2000, lambda: self.transcription_label.setText("Select a voice and click 'Test Voice' to hear it"))
            
            # Stop animation
            QTimer.singleShot(0, lambda: self.set_agent_visualizer_active(False))
            QTimer.singleShot(0, lambda: self.agent_animation_timer.stop() if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive() else None)
            
        except Exception as e:
            print(f"Direct voice test error: {e}")
            QTimer.singleShot(0, lambda: self.transcription_label.setText(f"Error: {str(e)}"))
            QTimer.singleShot(3000, lambda: self.transcription_label.setText("Select a voice and click 'Test Voice' to hear it"))
            
            # Stop animation on error
            QTimer.singleShot(0, lambda: self.set_agent_visualizer_active(False))
            QTimer.singleShot(0, lambda: self.agent_animation_timer.stop() if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive() else None)
    
    def test_voice(self):
        """Test the selected voice"""
        # Use direct method which is more reliable
        self.test_voice_direct()
    
    def update_audio_level(self, level):
        """Update the user audio level visualization."""
        self.user_visualizer.update_level(level)
    
    def update_agent_audio_level(self, level):
        """Update the agent audio level visualization."""
        self.agent_visualizer.update_level(level)
    
    def handle_transcription(self, text):
        """Handle new transcription text."""
        self.transcription_label.setText(f"You: {text}")
    
    def start_listening(self):
        """Start listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            self.transcription_label.setText("Speech recognition not ready yet")
            return
            
        self.audio_processor.start_listening()
        
        # Update state
        self.state_manager.update_state({"listening": True})
        
        # Activate user visualizer, deactivate agent visualizer
        self.set_user_visualizer_active(True)
        self.set_agent_visualizer_active(False)
    
    def stop_listening(self):
        """Stop listening mode."""
        # Skip if audio processor is not ready yet
        if not hasattr(self, 'audio_processor') or not self.audio_processor:
            return
            
        self.audio_processor.stop_listening()
        
        # Update state
        self.state_manager.update_state({"listening": False})
        
        # Deactivate user visualizer
        self.set_user_visualizer_active(False)
    
    def on_speaking_started(self):
        """Called when speaking starts."""
        self.speak_button.setEnabled(False)
        
        # Update state
        self.state_manager.update_state({"speaking": True})
        
        # Record when speaking started for the watchdog timer
        self._speaking_start_time = time.time()
        
        # Activate agent visualizer, deactivate user visualizer
        self.set_agent_visualizer_active(True)
        self.set_user_visualizer_active(False)
        
        # Create a dedicated animation timer if it doesn't exist
        if not hasattr(self, 'agent_animation_timer'):
            self.agent_animation_timer = QTimer(self)
            self.agent_animation_timer.timeout.connect(self.animate_agent_visualizer)
            
        # Restart the timer to ensure consistent animation
        if self.agent_animation_timer.isActive():
            self.agent_animation_timer.stop()
        self.agent_animation_timer.start(50)  # Update every 50ms
        
    def on_speaking_finished(self):
        """Called when speaking finishes."""
        self.speak_button.setEnabled(True)
        
        # Update state
        self.state_manager.update_state({"speaking": False})
        
        # Clear the speaking start time
        if hasattr(self, '_speaking_start_time'):
            del self._speaking_start_time
        
        # Deactivate agent visualizer
        self.set_agent_visualizer_active(False)
        
        # Stop the animation timer to ensure the agent visualizer stops completely
        if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
            self.agent_animation_timer.stop()
            
        # Reset the agent visualizer to ensure it's completely inactive
        QTimer.singleShot(100, lambda: self.agent_visualizer.update_level(0.0))
            
    def animate_agent_visualizer(self):
        """Animate the agent visualizer with pre-recorded patterns"""
        # Just trigger an update - the visualizer will use its internal patterns
        self.agent_visualizer.update_level(0.0)  # The level is ignored for agent mode
    
    def check_for_commands(self):
        """Check for commands from the server."""
        if os.path.exists(COMMAND_FILE):
            try:
                with open(COMMAND_FILE, 'r') as f:
                    command = f.read().strip()
                
                # Process the command
                if command == CMD_LISTEN:
                    # If components are not ready, store the command to process later
                    if not hasattr(self, 'audio_processor') or not self.audio_processor:
                        # Command will be processed in on_components_ready
                        return
                    
                    # Only start listening if we have a saved voice preference
                    if self.has_saved_voice_preference():
                        self.start_listening()
                    else:
                        # Show guidance message instead
                        self.transcription_label.setText("Please select a voice from the dropdown and click 'Select Voice' to continue")
                        # Wait a moment before speaking to ensure UI is fully ready
                        QTimer.singleShot(500, self.play_guidance_message)
                        
                elif command == CMD_IDLE and hasattr(self, 'audio_processor') and self.audio_processor and self.audio_processor.is_listening:
                    self.stop_listening()
                elif command == CMD_SPEAK:
                    # We'll handle speaking in check_for_responses
                    if hasattr(self, 'tts_adapter') and self.tts_adapter:
                        # Activate agent visualizer
                        self.set_agent_visualizer_active(True)
                        self.set_user_visualizer_active(False)
            except Exception:
                pass
    
    def check_for_responses(self):
        """Check for response files to speak."""
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
                
                # DIRECT TTS - use the direct module instead of the adapter
                if response:
                    try:
                        # Get the current voice
                        voice = self.tts_adapter.get_current_voice()
                        if not voice:
                            voice = "bm_daniel"  # Default fallback
                        
                        # Import and use the direct TTS module
                        from speech_mcp.utils.sound_player import SoundPlayer
                        print(f"Speaking response with SoundPlayer using voice: {voice}")
                        
                        # Get API settings from environment
                        api_key = os.environ.get("OPENAI_API_KEY")
                        base_url = os.environ.get("OPENAI_TTS_API_BASE_URL")
                        model = os.environ.get("SPEECH_MCP_TTS_MODEL")
                        
                        # Use the sound player for more reliable playback
                        success = SoundPlayer.play_tts_in_new_process(
                            text=response,
                            voice=voice,
                            api_key=api_key,
                            base_url=base_url,
                            model=model
                        )
                        
                        if not success:
                            # Fall back to direct TTS if sound player fails
                            from speech_mcp.utils.direct_tts import speak_text
                            print(f"SoundPlayer failed, using direct TTS with voice: {voice}")
                            success = speak_text(response, voice)
                            
                            if not success:
                                raise Exception("Both SoundPlayer and direct TTS failed")
                    except Exception as e:
                        print(f"Error with direct methods, falling back to adapter: {e}")
                        self.tts_adapter.speak(response)
                
            except Exception as e:
                self.transcription_label.setText(f"Error processing response: {str(e)}")
                QTimer.singleShot(3000, lambda: self.transcription_label.setText("Ready for voice interaction"))
                print(f"Error in check_for_responses: {e}")
                
                # Deactivate agent visualizer on error
                self.set_agent_visualizer_active(False)
                if hasattr(self, 'agent_animation_timer') and self.agent_animation_timer.isActive():
                    self.agent_animation_timer.stop()
    def check_ui_responsiveness(self):
        """Check if UI is responsive and reset state if needed."""
        # Check if TTS adapter is in a stuck state
        if hasattr(self, 'tts_adapter') and self.tts_adapter:
            # Use the lock to safely check the speaking state
            with self.tts_adapter._speaking_lock:
                is_speaking = self.tts_adapter.is_speaking
            
            # If speaking state has been active for too long, reset it
            if is_speaking and hasattr(self, '_speaking_start_time'):
                duration = time.time() - self._speaking_start_time
                if duration > 30:  # 30 seconds max for speaking
                    with self.tts_adapter._speaking_lock:
                        self.tts_adapter.is_speaking = False
                    self.on_speaking_finished()
            elif is_speaking:
                # Record when speaking started
                self._speaking_start_time = time.time()
            else:
                # Clear the timestamp when not speaking
                if hasattr(self, '_speaking_start_time'):
                    del self._speaking_start_time


def run_ui():
    """Run the PyQt speech UI."""
    app = QApplication(sys.argv)
    window = PyQtSpeechUI()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    # Run the UI
    sys.exit(run_ui())