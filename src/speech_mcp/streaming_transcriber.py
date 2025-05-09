"""
StreamingTranscriber class for real-time speech transcription.

This module provides a streaming interface to speech recognition systems,
enabling real-time transcription with word-level timestamps and natural
end-of-speech detection based on word timing patterns.

It supports OpenAI STT for speech recognition.
"""

from typing import Optional, Callable, Dict, Any, Tuple, List, Union
import threading
import queue
import numpy as np
import tempfile
import time
import wave
import os
import io

# Import the centralized logger
from speech_mcp.utils.logger import get_logger
from speech_mcp.constants import (
    STREAMING_END_SILENCE_DURATION,
    STREAMING_INITIAL_WAIT,
    RATE, FORMAT, CHANNELS
)

# Get a logger for this module
logger = get_logger(__name__, component="stt")

class StreamingTranscriber:
    """
    Handles real-time streaming transcription using OpenAI STT.
    
    This class manages a continuous audio stream, processing chunks of audio data
    as they arrive and providing both partial and final transcriptions. It uses
    word-level timing information to detect natural speech boundaries rather than
    relying on simple silence detection.
    """
    
    def __init__(self, 
                 model_name: str = None,
                 language: str = "en",
                 on_partial_transcription: Optional[Callable[[str], None]] = None,
                 on_final_transcription: Optional[Callable[[str, Dict[str, Any]], None]] = None,
                 **kwargs):  # Accept kwargs to maintain compatibility
        """
        Initialize the StreamingTranscriber.
        
        Args:
            model_name: The name of the model to use
            language: Language code for transcription (e.g., "en" for English)
            on_partial_transcription: Callback for partial transcription updates
            on_final_transcription: Callback for final transcription with metadata
            **kwargs: Additional arguments (for compatibility)
        """
        # Initialize attributes
        self.model_name = model_name or os.environ.get('SPEECH_MCP_STT_MODEL')
        self.language = language
        self.engine = "openai"
        self.on_partial_transcription = on_partial_transcription
        self.on_final_transcription = on_final_transcription
        
        # Audio processing attributes
        self._audio_queue = queue.Queue()
        self._audio_buffer = []
        self._current_transcription = ""
        self._accumulated_transcription = ""  # Store all transcribed segments
        self._last_word_time = 0.0
        self._last_word_detected = time.time()
        self._stream_start_time = 0.0  # Track when streaming started
        
        # Synchronization primitives
        self._is_active = False
        self._stopping = False  # Flag to indicate clean shutdown in progress
        self._processing_thread = None
        self._thread_exception = None  # Store exceptions from background thread
        self._lock = threading.RLock()  # Main lock for shared state (reentrant to prevent deadlocks)
        self._buffer_lock = threading.RLock()  # Separate lock for audio buffer to reduce contention
        
        # Use the temp file manager for resource tracking
        try:
            from speech_mcp.utils.temp_file_manager import get_temp_manager
            self.temp_manager = get_temp_manager()
            # Keep legacy tracking for backward compatibility
            self._temp_files = []
        except Exception as e:
            logger.warning(f"Failed to initialize temp file manager: {e}")
            # Fallback to legacy tracking
            self._temp_files = []
        
        # API client
        self.openai_stt = None
        
        # Initialize the STT engine
        self._initialize_engine()
        
        # Register cleanup on program exit
        import atexit
        atexit.register(self.cleanup)
        
    def _initialize_engine(self) -> bool:
        """
        Initialize the OpenAI STT engine.
        
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            from speech_mcp.tts_adapters.openai_stt_adapter import OpenAISTT
            logger.info(f"Initializing OpenAI STT with model '{self.model_name}'...")
            
            # Get API key and base URL from environment
            api_key = os.environ.get("OPENAI_API_KEY", "ollama")
            api_base = os.environ.get("OPENAI_STT_API_BASE_URL")
            
            # Initialize with keyword arguments for better clarity
            self.openai_stt = OpenAISTT(
                api_key=api_key, 
                api_base=api_base, 
                model=self.model_name
            )
            logger.info("OpenAI STT initialized successfully!")
            return True
            
        except ImportError as e:
            logger.error(f"Failed to load OpenAI STT adapter: {e}")
            return False
        except Exception as e:
            logger.error(f"Error initializing OpenAI STT: {e}")
            return False
    
    
    def cleanup(self) -> None:
        """
        Clean up resources used by the transcriber.
        
        This method ensures all temporary files are deleted and resources are released.
        It now uses the centralized TempFileManager for temp file cleanup.
        """
        logger.info("Cleaning up streaming transcriber resources")
        
        # Stop streaming if active
        if self._is_active:
            logger.info("Stopping active stream during cleanup")
            self.stop_streaming()
        
        # Clean up temporary files using the temp file manager
        try:
            from speech_mcp.utils.temp_file_manager import get_temp_manager
            manager = get_temp_manager()
            cleaned_count = manager.cleanup_component('streaming_transcriber')
            logger.info(f"Cleaned up {cleaned_count} temporary files via manager")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files via manager: {e}")
            
            # Fall back to legacy cleanup if manager fails
            with self._lock:
                temp_files = getattr(self, '_temp_files', []).copy()
                if hasattr(self, '_temp_files'):
                    self._temp_files.clear()
                
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                        logger.debug(f"Cleaned up temporary file directly: {temp_file}")
                except Exception as e:
                    logger.warning(f"Failed to delete temporary file {temp_file}: {e}")
                
        # Reset all state
        with self._lock:
            self._audio_queue = queue.Queue()
            self._current_transcription = ""
            self._accumulated_transcription = ""
            self._thread_exception = None
            self._is_active = False
            self._stopping = False
        
        with self._buffer_lock:
            self._audio_buffer = []
            
        logger.info("Streaming transcriber cleanup completed")
    
    def start_streaming(self) -> bool:
        """
        Start processing the audio stream.
        
        Returns:
            bool: True if streaming started successfully, False otherwise
        """
        with self._lock:
            if self._is_active:
                logger.warning("Streaming is already active")
                return False
                
            # Reset stopping flag
            self._stopping = False
            self._thread_exception = None
            
            # Clear existing data
            self._audio_queue = queue.Queue()
            self._current_transcription = ""
            self._accumulated_transcription = ""  # Reset accumulated transcription
            self._last_word_time = 0.0
            self._last_word_detected = time.time()
            self._stream_start_time = time.time()  # Set stream start time
            
        # Clear audio buffer with separate lock to reduce contention
        with self._buffer_lock:
            self._audio_buffer = []
            
        try:
            # Re-initialize the engine if needed
            if self.openai_stt is None and not self._initialize_engine():
                logger.error("Failed to initialize speech recognition engine")
                return False
            
            logger.info(f"Starting streaming transcription with {self.engine} engine and model '{self.model_name}'")
                
            # Start the processing thread
            with self._lock:
                self._is_active = True
                
            # Create and start the thread with a meaningful name
            thread = threading.Thread(
                target=self._process_audio_stream,
                name="TranscriptionThread",
                daemon=True
            )
            self._processing_thread = thread
            thread.start()
            
            logger.info(f"Streaming started successfully with {self.engine} engine")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start streaming: {str(e)}")
            with self._lock:
                self._is_active = False
            return False
    
    def add_audio_chunk(self, audio_chunk: bytes) -> None:
        """
        Add an audio chunk to the processing queue.
        
        Args:
            audio_chunk: Raw audio data as bytes (assumed to be 16-bit PCM)
        """
        # Check if active in a thread-safe way
        is_active = False
        is_stopping = False
        
        with self._lock:
            is_active = self._is_active
            is_stopping = self._stopping
            
        # Don't process new audio during shutdown
        if is_stopping:
            return
            
        # Add to queue if active
        if is_active and audio_chunk:
            try:
                # Don't block indefinitely if queue is full
                self._audio_queue.put(audio_chunk, block=True, timeout=0.5)
            except queue.Full:
                logger.warning("Audio processing queue is full, dropping audio chunk")
            
            # Check for exceptions from the processing thread
            with self._lock:
                if self._thread_exception:
                    logger.error(f"Audio processing thread encountered an error: {self._thread_exception}")
                    # Clear the exception so we only log it once
                    self._thread_exception = None
    
    def _process_audio_stream(self) -> None:
        """
        Background thread function to process the audio stream.
        
        This continuously processes audio chunks from the queue and updates
        the transcription when enough data is available.
        """
        # Add noise handling at the start with much more sensitive threshold
        noise_calibration_samples = []
        noise_calibration_done = False
        noise_threshold = 30  # Much lower threshold for better sensitivity (from your example)
        consecutive_silence_frames = 0  # Counter for consecutive silent frames
        silence_limit = 1.5  # Time of silence before stopping (from your example)
        self.RATE = 16000  # Audio sample rate (from your example)
        
        # Wait a brief moment to collect ambient noise
        noise_collect_end = time.time() + 0.5  # 0.5 seconds of noise collection (decreased from 0.75)
        
        logger.info("Starting audio stream processing with initial noise calibration")
        
        while self._is_active:
            try:
                # Get audio chunk with timeout to allow checking _is_active
                chunk = self._audio_queue.get(timeout=0.1)
                
                # Convert bytes to numpy array
                audio_data = np.frombuffer(chunk, dtype=np.int16)
                
                # During initial period, calibrate for noise level
                if not noise_calibration_done and time.time() < noise_collect_end:
                    # Collect samples for noise calibration
                    noise_calibration_samples.append(np.abs(audio_data).mean())
                    continue
                elif not noise_calibration_done:
                    # Calculate noise threshold - much more sensitive like the example code
                    if noise_calibration_samples:
                        avg_noise = np.mean(noise_calibration_samples)
                        # Use a much lower threshold similar to your example code
                        noise_threshold = max(30, avg_noise * 1.1)  # Very sensitive threshold
                        logger.info(f"Noise calibration complete. Ambient level: {avg_noise:.2f}, Threshold: {noise_threshold:.2f}")
                    noise_calibration_done = True
                
                # Check if this chunk has meaningful audio or just background noise
                current_level = np.abs(audio_data).mean()
                
                # Only add to buffer if above noise threshold
                if current_level > noise_threshold:
                    # This chunk has speech, append to buffer
                    self._audio_buffer.append(audio_data)
                    logger.debug(f"Audio level: {current_level:.2f} (above threshold {noise_threshold:.2f})")
                else:
                    # This is likely noise, but we'll still track some quiet audio to maintain context
                    if len(self._audio_buffer) > 0:  # Only if we've already captured some speech
                        self._audio_buffer.append(audio_data)
                    logger.debug(f"Audio level: {current_level:.2f} (below threshold {noise_threshold:.2f})")
                
                # Process buffer when it reaches sufficient size
                # Using 20 chunks = ~1 second of audio at 16kHz
                if len(self._audio_buffer) >= 20:
                    logger.debug(f"Processing buffer with {len(self._audio_buffer)} chunks")
                    self._transcribe_buffer()
                    
                # Check if we're still in initial wait period
                time_since_start = time.time() - self._stream_start_time
                if time_since_start < STREAMING_INITIAL_WAIT:
                    continue
                    
                # Similar to your AudioRecorder example's silence detection
                current_time = time.time()
                current_level = np.abs(audio_data).mean()
                
                # Track speech vs silence more like your example
                if current_level > noise_threshold:
                    # Speech detected - reset silence counter
                    if consecutive_silence_frames > 0:
                        logger.info(f"Speech resumed after {consecutive_silence_frames} silent frames")
                    consecutive_silence_frames = 0
                    self._last_word_detected = current_time  # Update last activity time
                else:
                    # Silence detected - increment counter
                    consecutive_silence_frames += 1
                    
                    # Calculate silence duration in seconds
                    silence_duration = consecutive_silence_frames * (len(audio_data) / self.RATE)
                    
                    # Log silence duration
                    if consecutive_silence_frames % 5 == 0:
                        logger.info(f"Silence duration: {silence_duration:.2f}s (threshold: {silence_limit}s)")
                    
                    # Stop if silence exceeds limit and we have some transcription
                    if silence_duration > silence_limit and self._current_transcription.strip():
                        logger.info(f"STOPPING: Silence detected for {silence_duration:.2f}s (threshold: {silence_limit}s)")
                        # Process any remaining audio and notify through callbacks
                        self.stop_streaming()
                        break
                
                # Also keep the word-based detection as a backup
                time_since_last_word = current_time - self._last_word_detected
                if time_since_last_word > STREAMING_END_SILENCE_DURATION and self._current_transcription.strip():
                    logger.info(f"STOPPING: No activity for {time_since_last_word:.1f}s (threshold: {STREAMING_END_SILENCE_DURATION}s)")
                    self.stop_streaming()
                    break
                
                # Removed the frame-level silence detection to prevent cutting off speech
                # This improves the experience by relying solely on the word-level silence detection,
                # which is more reliable for detecting the actual end of speech
                
                # Only use the consecutive silence frames counter for debugging purposes
                if current_level <= noise_threshold * 1.3:
                    consecutive_silence_frames += 1
                else:
                    # Reset counter on louder sounds
                    if current_level > noise_threshold * 1.5:
                        consecutive_silence_frames = 0
                
                # Log the consecutive silence frames for debugging
                if consecutive_silence_frames > 0 and consecutive_silence_frames % 20 == 0:
                    logger.debug(f"Consecutive silence frames: {consecutive_silence_frames} (not stopping)")
                    
                # We no longer use this counter to stop streaming - rely on word-level detection instead
                
                # If we've been processing for more than 15 seconds without any text, also stop 
                # This only applies if NO speech has been detected yet
                time_since_start = current_time - self._stream_start_time
                if time_since_start > 15.0 and not self._current_transcription.strip():
                    logger.info(f"No speech detected after {time_since_start:.1f} seconds, stopping")
                    self.stop_streaming()
                    break
                    
                # Add a much longer maximum time limit for the entire recording session (after transcription has started)
                # This prevents the microphone from staying on indefinitely but allows for long pauses during speech
                if self._current_transcription.strip() and time_since_start > 60.0:
                    logger.info(f"Maximum recording time reached ({time_since_start:.1f}s), stopping")
                    self.stop_streaming()
                    break
                    
                # We're removing the additional check that was cutting off speech too early
                # This should prevent interruptions during normal speech pauses
                    
            except queue.Empty:
                # No new audio data, but process buffer if we have enough
                if len(self._audio_buffer) >= 10:
                    logger.debug(f"Processing buffer during quiet period: {len(self._audio_buffer)} chunks")
                    self._transcribe_buffer()
                continue
            except Exception as e:
                logger.error(f"Error processing audio stream: {str(e)}")
                # Continue processing despite errors
                continue
    
    def _save_buffer_to_wav(self) -> str:
        """
        Save the current audio buffer to a temporary WAV file.
        
        This method creates a WAV file from the audio buffer and tracks it
        for later cleanup using the centralized TempFileManager.
        
        Returns:
            str: Path to the temporary WAV file, or empty string on error
        """
        buffer_copy = None
        temp_path = ""
        
        try:
            # Import the temp file manager
            from speech_mcp.utils.temp_file_manager import create_temp_file, cleanup_temp_file
            
            # Safely get a copy of the buffer
            with self._buffer_lock:
                if not self._audio_buffer:
                    logger.warning("Audio buffer is empty, nothing to save")
                    return ""
                buffer_copy = self._audio_buffer.copy()
            
            # Get buffer size for logging
            buffer_frames = len(buffer_copy)
            logger.debug(f"Saving audio buffer with {buffer_frames} frames to temporary WAV file")
            
            # Combine audio chunks
            try:
                audio_data = np.concatenate(buffer_copy)
                audio_size = len(audio_data) * 2  # 16-bit = 2 bytes per sample
                logger.debug(f"Combined audio data size: {audio_size} bytes ({audio_size/1024:.1f} KB)")
            except Exception as e:
                logger.error(f"Error combining audio chunks: {e}")
                return ""
            
            # Create a temp file using the centralized manager
            try:
                temp_path = create_temp_file(
                    suffix='.wav', 
                    prefix='stt_buffer_', 
                    component='streaming_transcriber',
                    auto_cleanup_seconds=300  # Auto-cleanup after 5 minutes
                )
                
                if not temp_path:
                    logger.error("Failed to create temporary WAV file")
                    return ""
                    
                logger.debug(f"Created temporary file: {temp_path}")
            except Exception as e:
                logger.error(f"Error creating temporary file: {e}")
                return ""
            
            # Write wave file
            try:
                with wave.open(temp_path, 'wb') as wav_file:
                    wav_file.setnchannels(CHANNELS)
                    wav_file.setsampwidth(2)  # 16-bit
                    wav_file.setframerate(RATE)
                    wav_file.writeframes(audio_data.tobytes())
                
                # Log successful write
                file_size = os.path.getsize(temp_path)
                logger.debug(f"Successfully wrote WAV file: {temp_path} ({file_size} bytes)")
                return temp_path
            except Exception as e:
                logger.error(f"Error writing WAV file: {e}")
                # Clean up failed temp file
                cleanup_temp_file(temp_path)
                return ""
            
        except Exception as e:
            logger.error(f"Error saving audio buffer to WAV: {e}")
            # Clean up failed temp file
            if temp_path:
                try:
                    from speech_mcp.utils.temp_file_manager import cleanup_temp_file
                    cleanup_temp_file(temp_path)
                except Exception:
                    # Fallback to direct deletion if manager import fails
                    try:
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                    except Exception:
                        pass
            return ""
    
    def _cleanup_temp_file(self, file_path: str) -> None:
        """
        Helper method to clean up a temporary file.
        
        This is a legacy method that now delegates to the temp file manager.
        """
        if not file_path:
            return
            
        try:
            # Use centralized temp file manager
            from speech_mcp.utils.temp_file_manager import cleanup_temp_file
            cleanup_temp_file(file_path)
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {file_path} via manager: {e}")
            
            # Fallback to direct deletion
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    logger.debug(f"Cleaned up temporary file directly: {file_path}")
            except Exception as direct_error:
                logger.warning(f"Failed direct cleanup of file {file_path}: {direct_error}")
                
        # Remove from the legacy tracking list if it exists
        try:
            with self._lock:
                if hasattr(self, '_temp_files') and file_path in self._temp_files:
                    self._temp_files.remove(file_path)
        except Exception:
            pass
    
    def _transcribe_buffer(self) -> None:
        """
        Transcribe the current audio buffer and update transcription.
        """
        try:
            if self.openai_stt is not None:
                self._transcribe_buffer_openai()
            else:
                logger.error("OpenAI STT not initialized")
                
            # Clear the buffer after processing
            self._audio_buffer = []
            
        except Exception as e:
            logger.error(f"Error transcribing buffer: {str(e)}")
            if hasattr(e, 'args') and len(e.args) > 0:
                logger.error(f"Error details: {e.args[0]}")
    
    def _transcribe_buffer_openai(self) -> None:
        """
        Transcribe the current audio buffer using OpenAI STT.
        """
        # Save buffer to temporary file
        temp_path = self._save_buffer_to_wav()
        if not temp_path:
            return
            
        try:
            # Transcribe with OpenAI
            logger.debug("Transcribing buffer with OpenAI STT")
            
            # Check if we have reasonable amount of audio to process
            file_size = os.path.getsize(temp_path)
            if file_size < 1000:  # Less than 1KB is likely just background noise
                logger.debug(f"Buffer too small ({file_size} bytes), skipping transcription")
                return
                
            logger.debug(f"Processing audio buffer of size: {file_size} bytes")
            
            # Transcribe the audio file
            result = self.openai_stt.transcribe_file(
                temp_path,
                language=self.language
            )
            
            # Check for errors in the result
            if isinstance(result, dict) and "error" in result:
                logger.error(f"Error in STT response: {result.get('error')}")
                return
            
            # Get the transcribed text from the result
            new_text = self.openai_stt.get_text_from_result(result)
            
            # Skip if text is empty or just punctuation/whitespace
            if not new_text or not new_text.strip() or new_text.strip() in ".,?!;:-":
                logger.debug("Empty or insignificant text result, skipping")
                return
                
            # Update transcription if we got meaningful text
            if new_text:
                with self._lock:
                    # For first chunk, validate the text but be less strict
                    is_first_chunk = not self._accumulated_transcription
                    if is_first_chunk:
                        # For initial transcription, require at least 1 word
                        words = new_text.strip().split()
                        if len(words) < 1:
                            logger.debug(f"Initial transcription empty, waiting for more speech")
                            return
                    
                    # Append to accumulated transcription
                    if self._accumulated_transcription:
                        self._accumulated_transcription += " " + new_text.strip()
                    else:
                        self._accumulated_transcription = new_text.strip()
                        
                    self._current_transcription = self._accumulated_transcription
                    
                    logger.info(f"Updated transcription: {self._current_transcription}")
                    
                    # Call partial transcription callback if provided
                    if self.on_partial_transcription:
                        self.on_partial_transcription(self._current_transcription)
                
                # Update last word detected time with clear logging
                old_time = self._last_word_detected
                self._last_word_detected = time.time()
                logger.info(f"Speech detected! Updating last_word_detected from {old_time:.2f} to {self._last_word_detected:.2f} (delta: {self._last_word_detected - old_time:.2f}s)")
                
        except Exception as e:
            logger.error(f"Error transcribing with OpenAI STT: {str(e)}")
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error deleting temp file: {str(e)}")
    
    
    def stop_streaming(self) -> Tuple[str, Dict[str, Any]]:
        """
        Stop streaming and return the final transcription.
        
        This method safely stops the streaming transcription process, processes
        any remaining audio, and returns the final result.
        
        Returns:
            Tuple containing:
            - Final transcription text (str)
            - Metadata dictionary with timing information
        """
        logger.info("Stopping streaming transcription")
        
        # First check if we're actually active using thread-safe access
        with self._lock:
            if not self._is_active:
                logger.debug("Streaming is not active, nothing to stop")
                return "", {}
            
            # Set stopping flag first to prevent race conditions
            self._stopping = True
            
        try:
            # Ensure queue is drained to avoid blocking
            try:
                # Empty the queue to avoid blocking the processing thread
                while not self._audio_queue.empty():
                    try:
                        self._audio_queue.get_nowait()
                    except queue.Empty:
                        break
            except Exception as e:
                logger.warning(f"Error draining audio queue: {e}")
            
            # Signal thread to stop - needs to be done atomically
            with self._lock:
                self._is_active = False
                current_thread = self._processing_thread
            
            # Wait for processing thread to finish if it exists and is not the current thread
            if current_thread and current_thread is not threading.current_thread():
                logger.debug(f"Waiting for transcription thread to complete (timeout: 5.0s)")
                thread_name = current_thread.name
                
                # Set a timeout to prevent deadlocks
                join_timeout = 5.0
                join_start = time.time()
                current_thread.join(timeout=join_timeout)
                join_duration = time.time() - join_start
                
                # Check if thread terminated within timeout
                if current_thread.is_alive():
                    logger.warning(f"Transcription thread '{thread_name}' did not terminate within {join_timeout}s")
                else:
                    logger.debug(f"Transcription thread '{thread_name}' terminated in {join_duration:.2f}s")
            
            # Process any remaining audio if buffer is not empty
            remaining_audio = False
            with self._buffer_lock:
                remaining_audio = len(self._audio_buffer) > 0
                
            if remaining_audio:
                logger.info("Processing remaining audio before stopping")
                self._transcribe_buffer()
            
            # Get final transcription and metadata with thread-safe access
            with self._lock:
                final_text = self._current_transcription
                metadata = {
                    "last_word_time": self._last_word_time,
                    "engine": self.engine,
                    "model": self.model_name,
                    "language": self.language,
                    "time_since_last_word": time.time() - self._last_word_detected,
                    "total_duration": time.time() - self._stream_start_time
                }
            
            # Call final transcription callback if provided
            if self.on_final_transcription:
                try:
                    self.on_final_transcription(final_text, metadata)
                except Exception as e:
                    logger.error(f"Error in final transcription callback: {e}")
            
            logger.info(f"Streaming stopped successfully, final text length: {len(final_text)} chars")
            return final_text, metadata
            
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return "", {}
            
        finally:
            # Reset all state in a thread-safe way
            with self._lock:
                self._is_active = False
                self._stopping = False
                self._processing_thread = None
                self._audio_queue = queue.Queue()
                self._current_transcription = ""
                self._last_word_time = 0.0
                self._last_word_detected = time.time()
                
            with self._buffer_lock:
                self._audio_buffer = []
    
    def get_current_transcription(self) -> str:
        """
        Get the current partial transcription.
        
        Returns:
            str: Current transcription text
        """
        with self._lock:
            return self._current_transcription
    
    def is_active(self) -> bool:
        """
        Check if streaming is currently active.
        
        This method provides thread-safe access to the active state.
        
        Returns:
            bool: True if streaming is active, False otherwise
        """
        with self._lock:
            return self._is_active and not self._stopping
            
    def has_error(self) -> Optional[str]:
        """
        Check if the transcription thread has encountered an error.
        
        Returns:
            Optional[str]: Error message if an error occurred, None otherwise
        """
        with self._lock:
            return str(self._thread_exception) if self._thread_exception else None