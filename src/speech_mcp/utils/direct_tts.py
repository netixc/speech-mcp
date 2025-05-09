"""
Direct TTS implementation with PyAudio for streaming.
"""

import os
import sys
import pyaudio
from openai import OpenAI
import threading

class TTSConfig:
    DEBUG = True  # Enable debug output - improved for debugging
    AUDIO_RATE = 24000  # Audio sample rate
    BASE_URL_TTS = os.environ.get("OPENAI_TTS_API_BASE_URL")
    TTS_MODEL = os.environ.get("SPEECH_MCP_TTS_MODEL")
    VOICE_ID = os.environ.get("SPEECH_MCP_TTS_VOICE")  # Default voice
    API_KEY = os.environ.get("OPENAI_API_KEY")
    
def debug_print(message):
    """Print debug messages to stderr with proper formatting"""
    if TTSConfig.DEBUG:
        print(f"[TTS DEBUG] {message}", file=sys.stderr, flush=True)

class TextToSpeech:
    def __init__(self):
        debug_print("Initializing Direct Text-to-Speech Client")
        self.setup_client()
        
    def setup_client(self):
        """Set up the OpenAI client for TTS"""
        debug_print(f"Setting up TTS client with URL: {TTSConfig.BASE_URL_TTS}")
        try:
            self.client = OpenAI(api_key=TTSConfig.API_KEY, base_url=TTSConfig.BASE_URL_TTS)
            debug_print("Successfully connected to TTS server")
        except Exception as e:
            print(f"Error setting up TTS client: {e}", file=sys.stderr)
            raise

    @staticmethod
    def setup_stream():
        """Set up PyAudio stream for audio playback"""
        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=TTSConfig.AUDIO_RATE,
            output=True
        )
        return stream, p

    def add_silence(self, stream, duration):
        """Add silence to the audio stream"""
        silence = b'\x00' * int(TTSConfig.AUDIO_RATE * duration)
        stream.write(silence)

    def speak(self, text, voice=None):
        """Convert text to speech and play it"""
        if not text.strip():
            debug_print("Empty text provided, nothing to speak")
            return
            
        if voice is None:
            voice = TTSConfig.VOICE_ID
            
        debug_print(f"Speaking text with voice '{voice}': {text}")
        
        stream, p = self.setup_stream()
        
        try:
            # Add natural pause at start
            self.add_silence(stream, 0.2)
            
            # Print more detailed info for debugging
            print(f"Direct TTS: Speaking with voice={voice}, model={TTSConfig.TTS_MODEL}, base_url={TTSConfig.BASE_URL_TTS}", file=sys.stderr)

            with self.client.audio.speech.with_streaming_response.create(
                model=TTSConfig.TTS_MODEL,
                voice=voice,
                response_format="pcm",
                input=text
            ) as response:
                # Debug message before starting playback
                print(f"Direct TTS: Starting audio playback stream", file=sys.stderr)
                
                chunk_count = 0
                for chunk in response.iter_bytes(chunk_size=1024):
                    if chunk:
                        stream.write(chunk)
                        chunk_count += 1
                        
                # Debug message after playback
                print(f"Direct TTS: Processed {chunk_count} audio chunks", file=sys.stderr)

            # Add natural pause at end
            self.add_silence(stream, 0.1)
            debug_print("Speech playback completed")
            
            # Print additional confirmation that audio should be audible
            print(f"Direct TTS: Audio playback completed successfully", file=sys.stderr)
            return True

        except Exception as e:
            print(f"TTS Error: {e}", file=sys.stderr)
            debug_print(f"TTS error details: {str(e)}")
            
            # Try fallback method using sound_player if available
            try:
                from speech_mcp.utils.sound_player import SoundPlayer
                print(f"Direct TTS: Attempting fallback with SoundPlayer", file=sys.stderr)
                
                # Get API settings from environment
                api_key = TTSConfig.API_KEY
                base_url = TTSConfig.BASE_URL_TTS
                model = TTSConfig.TTS_MODEL
                
                # Launch TTS in completely separate process
                success = SoundPlayer.play_tts_in_new_process(
                    text=text,
                    voice=voice,
                    api_key=api_key,
                    base_url=base_url,
                    model=model
                )
                
                if success:
                    print("Direct TTS: Fallback sound player process launched successfully", file=sys.stderr)
                    return True
                else:
                    print("Direct TTS: Failed to launch fallback sound player process", file=sys.stderr)
            except ImportError:
                print("Direct TTS: SoundPlayer not available for fallback", file=sys.stderr)
            except Exception as e2:
                print(f"Direct TTS: Fallback error: {e2}", file=sys.stderr)
                
            return False
        
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()

# Thread-safe singleton instance
_instance = None
_instance_lock = threading.Lock()

def get_tts_instance():
    """Get the singleton instance of TextToSpeech"""
    global _instance
    
    # If instance already exists, return it
    if _instance is not None:
        return _instance
        
    # Create instance if it doesn't exist
    with _instance_lock:
        if _instance is None:
            try:
                _instance = TextToSpeech()
            except Exception as e:
                print(f"Error creating TTS instance: {e}", file=sys.stderr)
                return None
    
    return _instance

def speak_text(text, voice=None):
    """Speak text using the singleton TTS instance"""
    tts = get_tts_instance()
    if tts:
        return tts.speak(text, voice)
    return False

# For testing
if __name__ == "__main__":
    TTSConfig.DEBUG = True
    text = "This is a test of the direct TTS module. If you can hear this, the module is working correctly."
    voice = "bm_daniel"
    
    print(f"Testing with voice {voice}: {text}")
    result = speak_text(text, voice)
    print(f"Result: {result}")