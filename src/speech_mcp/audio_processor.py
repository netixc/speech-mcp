"""
Audio processing module for speech-mcp.

This module provides centralized audio processing functionality including:
- Audio device selection
- Audio recording (with both traditional and streaming modes)
- Audio playback
- Audio level visualization
- Silence detection (in traditional mode)

The module supports two recording modes:
1. Traditional mode: Uses silence detection to automatically stop recording
2. Streaming mode: Continuously streams audio chunks to a callback function
"""

import os
import time
import tempfile
import threading
import wave
import numpy as np
import pyaudio
from typing import Optional, List, Tuple, Callable, Any, Dict

# Import the centralized logger
from speech_mcp.utils.logger import get_logger

# Get a logger for this module
logger = get_logger(__name__, component="stt")

# Import centralized constants
from speech_mcp.constants import (
    CHUNK, FORMAT, CHANNELS, RATE,
    SILENCE_THRESHOLD, MAX_SILENCE_DURATION, SILENCE_CHECK_INTERVAL,
    START_LISTENING_SOUND, STOP_LISTENING_SOUND
)

class AudioProcessor:
    """
    Core audio processing class that handles device selection, recording, and playback.
    
    This class provides the shared audio functionality used by both the server and UI components.
    """
    
    def __init__(self, on_audio_level: Optional[Callable[[float], None]] = None):
        """
        Initialize the audio processor.
        
        Args:
            on_audio_level: Optional callback function that receives audio level updates (0.0 to 1.0)
        """
        self.pyaudio = None
        self.stream = None
        self.selected_device_index = None
        self.is_listening = False
        self.audio_frames = []
        self.on_audio_level = on_audio_level
        self._on_recording_complete = None
        self._on_audio_chunk = None  # Callback for streaming mode
        self._streaming_mode = False  # Flag for streaming mode
        self._lock = threading.Lock()  # Add lock for thread safety
        self._stopping = False  # Flag to indicate stopping in progress
        self._temp_files = []  # Track temp files for cleanup
        self._setup_audio()
        
        # Register cleanup on program exit
        import atexit
        atexit.register(self.cleanup)
    
    def _setup_audio(self) -> None:
        """Set up audio capture and processing."""
        try:
            logger.info("Setting up audio processing")
            self.pyaudio = pyaudio.PyAudio()
            
            # Log audio device information
            logger.info(f"PyAudio version: {pyaudio.get_portaudio_version()}")
            
            # Get all available audio devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            logger.info(f"Found {numdevices} audio devices:")
            
            # Find the best input device
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    device_name = device_info.get('name')
                    max_input_channels = device_info.get('maxInputChannels')
                    
                    logger.info(f"Device {i}: {device_name}")
                    logger.info(f"  Max Input Channels: {max_input_channels}")
                    logger.info(f"  Default Sample Rate: {device_info.get('defaultSampleRate')}")
                    
                    # Only consider input devices
                    if max_input_channels > 0:
                        logger.info(f"Found input device: {device_name}")
                        
                        # Prefer non-default devices as they're often external mics
                        if self.selected_device_index is None or 'default' not in device_name.lower():
                            self.selected_device_index = i
                            logger.info(f"Selected input device: {device_name} (index {i})")
                except Exception as e:
                    logger.warning(f"Error checking device {i}: {e}")
            
            if self.selected_device_index is None:
                logger.warning("No suitable input device found, using default")
            
        except Exception as e:
            logger.error(f"Error setting up audio: {e}")
    
    def start_listening(self, 
                   callback: Optional[Callable] = None, 
                   on_recording_complete: Optional[Callable[[str], None]] = None,
                   streaming_mode: bool = True,  # Force streaming mode to True
                   on_audio_chunk: Optional[Callable[[bytes], None]] = None) -> bool:
        """
        Start listening for audio input.
        
        Args:
            callback: Optional callback function to call when audio data is received
            on_recording_complete: Optional callback function to call when recording is complete,
                                  receives the path to the recorded audio file as an argument
            streaming_mode: Whether to use streaming mode (no silence detection) - Currently forced to True
            on_audio_chunk: Optional callback function to receive audio chunks in streaming mode
            
        Returns:
            bool: True if listening started successfully, False otherwise
        """
        with self._lock:
            if self.is_listening:
                logger.debug("Already listening, ignoring start_listening call")
                return True
                
            # Reset stopping flag
            self._stopping = False
            
            # Clear old audio frames
            self.audio_frames = []
            self._on_recording_complete = on_recording_complete
            self._streaming_mode = streaming_mode
            self._on_audio_chunk = on_audio_chunk
        
        # Play start listening notification sound
        threading.Thread(target=self.play_audio_file, args=(START_LISTENING_SOUND,), daemon=True).start()
        
        try:
            logger.info("Starting audio recording")
            
            # Ensure PyAudio instance exists
            if self.pyaudio is None:
                logger.info("Reinitializing PyAudio instance")
                self._setup_audio()
                if self.pyaudio is None:
                    logger.error("Failed to initialize PyAudio")
                    return False
            
            def audio_callback(in_data, frame_count, time_info, status):
                try:
                    # Check if we're stopping
                    if self._stopping:
                        logger.debug("Audio callback: stopping flag detected")
                        return (in_data, pyaudio.paComplete)
                    
                    # Check for audio status flags
                    if status:
                        status_flags = []
                        if status & pyaudio.paInputUnderflow:
                            status_flags.append("Input Underflow")
                        if status & pyaudio.paInputOverflow:
                            status_flags.append("Input Overflow")
                        if status & pyaudio.paOutputUnderflow:
                            status_flags.append("Output Underflow")
                        if status & pyaudio.paOutputOverflow:
                            status_flags.append("Output Overflow")
                        if status & pyaudio.paPrimingOutput:
                            status_flags.append("Priming Output")
                        
                        if status_flags:
                            logger.warning(f"Audio callback status flags: {', '.join(status_flags)}")
                    
                    # Thread-safe audio frame handling
                    with self._lock:
                        # Store audio data for processing
                        self.audio_frames.append(in_data)
                    
                    # Process audio for visualization
                    self._process_audio_for_visualization(in_data)
                    
                    # Call streaming callback if in streaming mode
                    if self._streaming_mode and self._on_audio_chunk:
                        self._on_audio_chunk(in_data)
                    
                    # Call user-provided callback if available
                    if callback:
                        callback(in_data)
                    
                    return (in_data, pyaudio.paContinue)
                    
                except Exception as e:
                    logger.error(f"Error in audio callback: {e}")
                    return (in_data, pyaudio.paContinue)  # Try to continue despite errors
            
            # Start the audio stream with the selected device
            logger.debug(f"Opening audio stream with FORMAT={FORMAT}, CHANNELS={CHANNELS}, RATE={RATE}, CHUNK={CHUNK}, DEVICE={self.selected_device_index}")
            
            # Close any existing stream first
            if self.stream is not None:
                try:
                    if self.stream.is_active():
                        self.stream.stop_stream()
                    self.stream.close()
                except Exception as e:
                    logger.warning(f"Error closing existing stream: {e}")
                self.stream = None
            
            # Open new stream
            self.stream = self.pyaudio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=self.selected_device_index,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback
            )
            
            # Verify stream is active and receiving audio
            if not self.stream.is_active():
                logger.error("Stream created but not active")
                return False
            
            logger.info("Audio stream initialized and receiving data")
            
            # Set is_listening flag after stream is confirmed active
            with self._lock:
                self.is_listening = True
            
            # Start silence detection thread only if not in streaming mode
            if not self._streaming_mode:
                def silence_detection_thread():
                    self._detect_silence()
                    # If recording completed and callback is provided, get the audio path and call the callback
                    with self._lock:
                        if self._on_recording_complete and not self.is_listening:
                            audio_path = self.get_recorded_audio_path()
                            if audio_path:
                                self._on_recording_complete(audio_path)
                
                thread = threading.Thread(target=silence_detection_thread, daemon=True)
                thread.name = "SilenceDetectionThread"
                thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting audio stream: {e}")
            with self._lock:
                self.is_listening = False
            return False
    
    def _process_audio_for_visualization(self, audio_data: bytes) -> None:
        """
        Process audio data for visualization.
        
        Args:
            audio_data: Raw audio data from PyAudio
        """
        try:
            # Convert to numpy array
            data = np.frombuffer(audio_data, dtype=np.int16)
            
            # Normalize the data to range [-1, 1]
            normalized = data.astype(float) / 32768.0
            
            # Take absolute value to get amplitude
            amplitude = np.abs(normalized).mean()
            
            # Apply amplification factor to make the visualization more prominent
            # Increase the factor from 1.0 to 5.0 to make the visualization more visible
            amplification_factor = 5.0
            amplified_amplitude = min(amplitude * amplification_factor, 1.0)  # Clamp to 1.0 max
            
            # Call the audio level callback if provided
            if self.on_audio_level:
                self.on_audio_level(amplified_amplitude)
            
        except Exception:
            pass
    
    def _detect_silence(self, max_total_duration: float = 60.0) -> None:
        """
        Detect when the user stops speaking and end recording.
        
        This method runs in a separate thread and monitors audio levels to detect
        when the user has stopped speaking.
        
        Args:
            max_total_duration: Maximum total recording duration in seconds regardless 
                               of silence detection (default: 60.0)
        """
        try:
            # Wait for initial audio to accumulate
            logger.info("Starting silence detection with initial wait")
            time.sleep(0.5)
            
            # Initialize silence detection parameters
            silence_duration = 0
            start_time = time.time()
            
            logger.info(f"Silence detection parameters: threshold={SILENCE_THRESHOLD}, " +
                       f"max_silence={MAX_SILENCE_DURATION}s, max_total={max_total_duration}s")
            
            while True:
                # Check if we should continue processing
                with self._lock:
                    if not self.is_listening or self._stopping:
                        logger.info("Silence detection stopped due to listening state change")
                        break
                
                # Check if recording has exceeded maximum duration
                elapsed_time = time.time() - start_time
                if elapsed_time > max_total_duration:
                    logger.info(f"Maximum recording duration ({max_total_duration}s) reached, stopping")
                    self.stop_listening()
                    break
                
                # Safely access audio frames
                latest_frame = None
                with self._lock:
                    if self.audio_frames and len(self.audio_frames) >= 2:
                        latest_frame = self.audio_frames[-1]
                
                if latest_frame is None:
                    # No frames available yet
                    time.sleep(SILENCE_CHECK_INTERVAL)
                    continue
                
                # Process the latest audio frame
                audio_data = np.frombuffer(latest_frame, dtype=np.int16)
                normalized = audio_data.astype(float) / 32768.0
                current_amplitude = np.abs(normalized).mean()
                
                # Log audio levels periodically for debugging
                if int(elapsed_time) % 5 == 0 and int(elapsed_time) > 0:
                    logger.debug(f"Current audio level: {current_amplitude:.6f} (threshold: {SILENCE_THRESHOLD})")
                
                # Update silence duration
                if current_amplitude < SILENCE_THRESHOLD:
                    silence_duration += SILENCE_CHECK_INTERVAL
                    if silence_duration >= 1.0 and int(silence_duration) % 1 == 0:
                        logger.debug(f"Silence detected for {silence_duration:.1f}s (max: {MAX_SILENCE_DURATION}s)")
                else:
                    if silence_duration > 0:
                        logger.debug(f"Speech resumed after {silence_duration:.1f}s of silence")
                    silence_duration = 0
                
                # Check if silence exceeds maximum duration
                if silence_duration >= MAX_SILENCE_DURATION:
                    logger.info(f"Stopping due to silence detection: {silence_duration:.1f}s > {MAX_SILENCE_DURATION}s threshold")
                    self.stop_listening()
                    break
                
                # Short sleep to prevent CPU spinning
                time.sleep(SILENCE_CHECK_INTERVAL)
            
        except Exception as e:
            logger.error(f"Error in silence detection: {e}")
            # Try to stop listening in case of error
            if self.is_listening:
                self.stop_listening()
    
    def stop_listening(self) -> None:
        """
        Stop listening for audio input.
        
        Returns:
            None
        """
        # Set stopping flag first to prevent race conditions in callbacks
        self._stopping = True
        logger.info("Stopping audio recording")
        
        try:
            # Thread-safe state update
            with self._lock:
                # Store current stream in a local variable to avoid race conditions
                current_stream = self.stream
                self.stream = None
                self.is_listening = False
            
            # Play stop listening notification sound
            threading.Thread(target=self.play_audio_file, args=(STOP_LISTENING_SOUND,), daemon=True).start()
            
            # Close and clean up stream outside the lock to prevent deadlocks
            if current_stream:
                try:
                    if current_stream.is_active():
                        logger.debug("Stopping active stream")
                        current_stream.stop_stream()
                    logger.debug("Closing stream")
                    current_stream.close()
                    logger.debug("Stream closed successfully")
                except Exception as e:
                    logger.error(f"Error closing audio stream: {e}")
                    
        except Exception as e:
            logger.error(f"Error stopping audio recording: {e}")
            with self._lock:
                self.is_listening = False
    
    def get_recorded_audio_path(self) -> Optional[str]:
        """
        Save the recorded audio to a temporary WAV file and return the path.
        
        This method now uses the centralized TempFileManager to ensure
        proper cleanup of temporary files.
        
        Returns:
            str: Path to the temporary WAV file, or None if an error occurred
        """
        # Thread-safe access to audio frames
        with self._lock:
            if not self.audio_frames:
                logger.warning("No audio frames available to save")
                return None
            
            # Make a copy of the audio frames to avoid race conditions
            frames_copy = self.audio_frames.copy()
        
        try:
            # Import the temp file manager
            try:
                from speech_mcp.utils.temp_file_manager import create_temp_file, cleanup_temp_file
                use_temp_manager = True
            except ImportError:
                logger.warning("Temp file manager not available, using legacy method")
                use_temp_manager = False
            
            # Calculate total audio time for logging
            total_audio_time = len(frames_copy) * (CHUNK / RATE)
            logger.info(f"Saving {len(frames_copy)} audio frames ({total_audio_time:.2f} seconds) to temporary file")
            
            # Save the recorded audio to a temporary WAV file
            temp_audio_path = None
            try:
                # Create the temp file
                if use_temp_manager:
                    # Use the centralized manager
                    temp_audio_path = create_temp_file(
                        suffix='.wav', 
                        prefix='audio_recording_', 
                        component='audio_processor',
                        auto_cleanup_seconds=600  # Auto-cleanup after 10 minutes
                    )
                else:
                    # Fallback to traditional method
                    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                        temp_audio_path = temp_audio.name
                        
                        # Track this file for later cleanup using legacy method
                        with self._lock:
                            self._temp_files.append(temp_audio_path)
                
                if not temp_audio_path:
                    logger.error("Failed to create temporary audio file")
                    return None
                
                # Create a WAV file from the recorded frames
                with wave.open(temp_audio_path, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    
                    # Handle potential race condition where PyAudio is being cleaned up
                    sample_size = 2  # Default to 16-bit (2 bytes)
                    if self.pyaudio:
                        try:
                            sample_size = self.pyaudio.get_sample_size(FORMAT)
                        except Exception as e:
                            logger.warning(f"Could not get sample size from PyAudio: {e}")
                    
                    wf.setsampwidth(sample_size)
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames_copy))
                
                # Verify file was created successfully
                if os.path.exists(temp_audio_path) and os.path.getsize(temp_audio_path) > 0:
                    logger.info(f"Audio saved to temporary file: {temp_audio_path} ({os.path.getsize(temp_audio_path)} bytes)")
                    return temp_audio_path
                else:
                    logger.error(f"Created file is empty or does not exist: {temp_audio_path}")
                    return None
                
            except Exception as e:
                logger.error(f"Error saving audio to file: {e}")
                # Clean up the temp file if it was created but failed to write
                if temp_audio_path:
                    if use_temp_manager:
                        cleanup_temp_file(temp_audio_path)
                    else:
                        # Fallback to direct deletion
                        try:
                            if os.path.exists(temp_audio_path):
                                os.unlink(temp_audio_path)
                                logger.info(f"Cleaned up failed temporary file: {temp_audio_path}")
                                with self._lock:
                                    if temp_audio_path in self._temp_files:
                                        self._temp_files.remove(temp_audio_path)
                        except Exception as cleanup_error:
                            logger.warning(f"Failed to clean up temporary file: {cleanup_error}")
                return None
                
        except Exception as e:
            logger.error(f"Error preparing audio data: {e}")
            return None
    
    def record_audio(self, streaming_mode: bool = True, on_audio_chunk: Optional[Callable[[bytes], None]] = None, 
                  max_duration: float = 30.0) -> Optional[str]:
        """
        Record audio from the microphone and return the path to the audio file.
        
        This is a blocking method that handles the entire recording process including
        starting recording and streaming transcription. Silence detection is disabled.
        
        Args:
            streaming_mode: Whether to use streaming mode (forced to True)
            on_audio_chunk: Optional callback function to receive audio chunks in streaming mode
            max_duration: Maximum recording duration in seconds (default: 30.0)
            
        Returns:
            str: Path to the recorded audio file, or None if an error occurred
        """
        logger.info(f"Starting audio recording (max duration: {max_duration}s)")
        
        # Start recording
        if not self.start_listening(streaming_mode=streaming_mode, on_audio_chunk=on_audio_chunk):
            logger.error("Failed to start audio recording")
            return None
        
        # Calculate end time
        start_time = time.time()
        end_time = start_time + max_duration
        
        try:
            # Wait for recording to complete (silence detection will stop it in non-streaming mode)
            logger.info("Waiting for recording to complete")
            while self.is_listening:
                # Check for timeout
                if time.time() >= end_time:
                    logger.warning(f"Recording reached maximum duration of {max_duration}s, stopping")
                    self.stop_listening()
                    break
                    
                # Short sleep to prevent CPU spinning
                time.sleep(0.1)
            
            # Get the recorded audio file path
            logger.info("Recording complete, retrieving audio file")
            audio_path = self.get_recorded_audio_path()
            
            if audio_path:
                file_size = os.path.getsize(audio_path)
                logger.info(f"Audio recording saved to {audio_path} ({file_size} bytes)")
            else:
                logger.warning("No audio file was generated from recording")
                
            return audio_path
            
        except Exception as e:
            logger.error(f"Error in record_audio: {e}")
            # Ensure we stop listening in case of error
            if self.is_listening:
                self.stop_listening()
            return None
    
    def play_audio_file(self, file_path: str) -> bool:
        """
        Play an audio file using PyAudio.
        
        Args:
            file_path: Path to the audio file to play
            
        Returns:
            bool: True if the file was played successfully, False otherwise
        """
        p = None
        stream = None
        
        try:
            if not os.path.exists(file_path):
                logger.warning(f"Audio file not found: {file_path}")
                return False
            
            logger.debug(f"Playing audio file: {file_path}")
            
            # Open the wave file
            with wave.open(file_path, 'rb') as wf:
                # Get file properties for logging
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / framerate
                
                logger.debug(f"Audio file properties: {channels} channels, {sample_width} bytes per sample, {framerate} Hz, {duration:.2f} seconds")
                
                # Create PyAudio instance
                p = pyaudio.PyAudio()
                
                # Open stream
                stream = p.open(format=p.get_format_from_width(sample_width),
                                channels=channels,
                                rate=framerate,
                                output=True)
                
                # Read data in chunks and play
                chunk_size = 1024
                data = wf.readframes(chunk_size)
                
                while data:
                    stream.write(data)
                    data = wf.readframes(chunk_size)
                
                logger.debug(f"Finished playing audio file: {file_path}")
                return True
                
        except Exception as e:
            logger.error(f"Error playing audio file {file_path}: {e}")
            return False
            
        finally:
            # Clean up resources
            try:
                if stream:
                    stream.stop_stream()
                    stream.close()
                    logger.debug("Audio stream closed")
            except Exception as e:
                logger.warning(f"Error closing audio stream: {e}")
                
            try:
                if p:
                    p.terminate()
                    logger.debug("PyAudio instance terminated")
            except Exception as e:
                logger.warning(f"Error terminating PyAudio: {e}")
                
            # Ensure we don't have any references to these objects
            stream = None
            p = None
    
    def get_available_devices(self) -> List[Dict[str, Any]]:
        """
        Get a list of available audio input devices.
        
        Returns:
            List of dictionaries containing device information
        """
        devices = []
        
        try:
            if not self.pyaudio:
                self._setup_audio()
                
            if not self.pyaudio:
                return devices
                
            # Get all available audio devices
            info = self.pyaudio.get_host_api_info_by_index(0)
            numdevices = info.get('deviceCount')
            
            for i in range(numdevices):
                try:
                    device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, i)
                    max_input_channels = device_info.get('maxInputChannels')
                    
                    # Only include input devices
                    if max_input_channels > 0:
                        devices.append({
                            'index': i,
                            'name': device_info.get('name'),
                            'channels': max_input_channels,
                            'sample_rate': device_info.get('defaultSampleRate')
                        })
                except Exception:
                    pass
                    
            return devices
            
        except Exception:
            return devices
    
    def set_device_index(self, device_index: int) -> bool:
        """
        Set the audio input device by index.
        
        Args:
            device_index: Index of the audio device to use
            
        Returns:
            bool: True if the device was set successfully, False otherwise
        """
        try:
            # Check if the device exists
            if not self.pyaudio:
                self._setup_audio()
                
            if not self.pyaudio:
                return False
                
            try:
                device_info = self.pyaudio.get_device_info_by_host_api_device_index(0, device_index)
                if device_info.get('maxInputChannels') > 0:
                    self.selected_device_index = device_index
                    return True
                else:
                    return False
            except Exception:
                return False
                
        except Exception:
            return False
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the audio processor.
        
        This should be called when the audio processor is no longer needed.
        It will properly release all PyAudio resources and clean up temporary files.
        """
        logger.info("Cleaning up audio processor resources")
        
        # Set stopping flag to prevent any callbacks from accessing resources
        self._stopping = True
        
        # First stop any active listening
        if self.is_listening:
            logger.info("Stopping active listening during cleanup")
            self.stop_listening()
        
        # Clean up temporary files using the temp file manager
        try:
            from speech_mcp.utils.temp_file_manager import get_temp_manager
            manager = get_temp_manager()
            cleaned_count = manager.cleanup_component('audio_processor')
            logger.info(f"Cleaned up {cleaned_count} temporary files via manager")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files via manager: {e}")
            
            # Fall back to legacy cleanup if manager fails
            temp_files_to_clean = []
            with self._lock:
                temp_files_to_clean = getattr(self, '_temp_files', []).copy()
                if hasattr(self, '_temp_files'):
                    self._temp_files.clear()
            
            for temp_file in temp_files_to_clean:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logger.debug(f"Cleaned up temp file directly: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temp file {temp_file}: {e}")
        
        # Clean up PyAudio resources
        stream_to_close = None
        pyaudio_to_terminate = None
        
        with self._lock:
            stream_to_close = self.stream
            self.stream = None
            pyaudio_to_terminate = self.pyaudio
            self.pyaudio = None
            self.is_listening = False
        
        # Clean up stream outside the lock
        if stream_to_close:
            try:
                if stream_to_close.is_active():
                    stream_to_close.stop_stream()
                stream_to_close.close()
                logger.debug("Audio stream closed during cleanup")
            except Exception as e:
                logger.warning(f"Error closing audio stream during cleanup: {e}")
        
        # Clean up PyAudio outside the lock
        if pyaudio_to_terminate:
            try:
                pyaudio_to_terminate.terminate()
                logger.debug("PyAudio terminated during cleanup")
            except Exception as e:
                logger.warning(f"Error terminating PyAudio during cleanup: {e}")
        
        logger.info("Audio processor cleanup completed")
