"""
Audio processor UI wrapper for the Speech UI.

This module provides a PyQt wrapper around the AudioProcessor for speech recognition.
"""

import os
import threading
from typing import Optional
from PyQt5.QtCore import QObject, pyqtSignal

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="stt")

# Import centralized constants
from speech_mcp.constants import TRANSCRIPTION_FILE

# Import shared audio processor and speech recognition
from speech_mcp.audio_processor import AudioProcessor
from speech_mcp.speech_recognition import transcribe_audio

class AudioProcessorUI(QObject):
    """
    UI wrapper for AudioProcessor that handles speech recognition.
    
    This is a thin wrapper around the AudioProcessor class that provides PyQt signals
    for audio level updates and transcription results.
    """
    audio_level_updated = pyqtSignal(float)
    transcription_ready = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_listening = False
        
        # Create the shared AudioProcessor with a callback for audio levels
        self.audio_processor = AudioProcessor(on_audio_level=self._on_audio_level)
        
        # Initialize speech recognition in a background thread to avoid UI blocking
        threading.Thread(target=self._initialize_speech_recognition, daemon=True).start()
    
    def _on_audio_level(self, level):
        """Callback for audio level updates from the AudioProcessor"""
        self.audio_level_updated.emit(level)
    
    def _initialize_speech_recognition(self):
        """
        Initialize speech recognition in a background thread.
        
        This is a lightweight initialization that just imports the necessary modules,
        but doesn't create any objects since we'll use the centralized transcribe_audio function.
        """
        try:
            # Import here to trigger model loading in the background
            from speech_mcp.speech_recognition import initialize_speech_recognition
            initialize_speech_recognition()
            logger.info("Speech recognition initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing speech recognition: {e}")
    
    def start_listening(self):
        """Start listening for audio input."""
        if self.is_listening:
            logger.debug("Already listening, ignoring start_listening call")
            return
            
        self.is_listening = True
        
        # Start the shared audio processor with a callback for when recording is complete
        if not self.audio_processor.start_listening(on_recording_complete=self._on_recording_complete):
            logger.error("Failed to start audio processor")
            self.is_listening = False
            return
        
        logger.info("Started listening for audio input")
    
    def _on_recording_complete(self, audio_path: Optional[str]):
        """
        Callback for when recording is complete.
        
        Args:
            audio_path: Path to the recorded audio file, or None if recording failed
        """
        try:
            if not audio_path:
                logger.warning("No audio data to process")
                self.is_listening = False
                return
            
            logger.info(f"Recording complete, processing audio file: {audio_path}")
            
            # Use the centralized transcribe_audio function to process the audio
            transcription, metadata = transcribe_audio(audio_path)
            
            # Log the transcription details
            logger.info(f"Transcription completed: {transcription}")
            logger.debug(f"Transcription metadata: {metadata}")
            
            # Clean up the temporary file
            try:
                logger.debug(f"Removing temporary WAV file: {audio_path}")
                os.unlink(audio_path)
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")
            
            # Write the transcription to a file for the server to read
            try:
                logger.debug(f"Writing transcription to file: {TRANSCRIPTION_FILE}")
                with open(TRANSCRIPTION_FILE, 'w') as f:
                    f.write(transcription)
                logger.debug("Transcription file written successfully")
            except Exception as e:
                logger.error(f"Error writing transcription to file: {e}")
            
            # Emit the transcription signal
            logger.info("Emitting transcription_ready signal")
            self.transcription_ready.emit(transcription)
            
        except Exception as e:
            logger.error(f"Error processing recording: {e}")
            self.transcription_ready.emit(f"Error processing speech: {str(e)}")
        finally:
            self.is_listening = False
    
    def stop_listening(self):
        """Stop listening for audio input."""
        try:
            logger.info("Stopping audio recording")
            self.audio_processor.stop_listening()
            self.is_listening = False
            logger.info("Audio recording stopped")
            
        except Exception as e:
            logger.error(f"Error stopping audio recording: {e}")
            self.is_listening = False
