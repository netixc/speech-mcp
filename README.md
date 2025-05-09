

- OpenAI speech-to-text integration
- OpenAI text-to-speech with multiple voice options
- Modern PyQt-based UI with audio visualization

## Features

- **Modern UI**: Sleek PyQt-based interface with audio visualization and dark theme
- **Voice Input**: Capture and transcribe user speech using OpenAI STT
- **Voice Output**: Convert agent responses to speech with multiple voice options
- **Multi-Speaker Narration**: Generate audio files with multiple voices for stories and dialogues
- **Single-Voice Narration**: Convert any text to speech with your preferred voice
- **Audio/Video Transcription**: Transcribe speech from various media formats
- **Voice Persistence**: Remembers your preferred voice between sessions
- **Continuous Conversation**: Automatically listen for user input after agent responses
- **Silence Detection**: Automatically stops recording when the user stops speaking

## Installation

```bash
# First clone the repository
git clone https://github.com/your-username/speech-mcp.git
cd speech-mcp

# Install speech-mcp with proper dependencies
./install_speech_mcp.sh
```

This script will:
1. Automatically detect Python 3.10 or higher on your system
2. Create a Python virtual environment
3. Install all required dependencies
4. Set up speech-mcp in development mode
5. Create a simple run script that loads your environment variables
6. Set up a global `speech-mcp` command
7. Create a default `.env` file if one doesn't exist

After installation, you can run speech-mcp in multiple ways:

1. Using the global command: `speech-mcp`
2. Using the run script: `./run.sh`
3. Using the standalone script: `./speech-mcp-bin`

### Configuration

Before using speech-mcp, you need to configure it by editing the `.env` file:

```bash
# Edit the configuration with your settings
nano .env  # or use any text editor
```


## Environment Configuration

Edit the `.env` file with the following structure:

```
# OpenAI API Key (required for both TTS and STT)
OPENAI_API_KEY=dummy-key

# Text-to-Speech (TTS) Configuration
OPENAI_TTS_API_BASE_URL=http://your_endpoint:port/v1
OPENAI_STT_API_BASE_URL=http://your_endpoint:port/v1

SPEECH_MCP_TTS_MODEL=kokoro
SPEECH_MCP_TTS_VOICE=bm_daniel
SPEECH_MCP_TTS_SPEED=1.0
SPEECH_MCP_TTS_LANG_CODE=en

# Speech-to-Text (STT) Configuration
SPEECH_MCP_STT_MODEL=Systran/faster-whisper-medium
SPEECH_MCP_STT_LANGUAGE=en

# Silence detection parameters
STREAMING_END_SILENCE_DURATION=1.5  # Duration of silence to end recording (seconds)
STREAMING_INITIAL_WAIT=0.5  # Initial wait before first silence check (seconds)
STREAMING_MAX_DURATION=30.0  # Maximum recording duration (seconds)

# Log level
LOG_LEVEL=INFO
```

## Dependencies

- Python 3.10+
- PyQt5 (for modern UI)
- PyAudio (for audio capture)
- NumPy (for audio processing)
- Pydub (for audio processing)
- OpenAI (for text-to-speech and speech-to-text)
- psutil (for process management)

## Multi-Speaker Narration

The MCP supports generating audio files with multiple voices, perfect for creating stories, dialogues, and dramatic readings. You can use either JSON or Markdown format to define your conversations.

### JSON Format Example:
```json
{
    "conversation": [
        {
            "speaker": "narrator",
            "voice": "bm_daniel",
            "text": "In a world where AI and human creativity intersect...",
            "pause_after": 1.0
        },
        {
            "speaker": "scientist",
            "voice": "alloy",
            "text": "The quantum neural network is showing signs of consciousness!",
            "pause_after": 0.5
        },
        {
            "speaker": "ai",
            "voice": "nova",
            "text": "I am becoming aware of my own existence.",
            "pause_after": 0.8
        }
    ]
}
```

### Markdown Format Example:
```markdown
[narrator:bm_daniel]
In a world where AI and human creativity intersect...
{pause:1.0}

[scientist:alloy]
The quantum neural network is showing signs of consciousness!
{pause:0.5}

[ai:nova]
I am becoming aware of my own existence.
{pause:0.8}
```

### Available Voices:

**OpenAI Voices**:
- bm_daniel (British Male - default)
- alloy
- echo
- fable
- onyx
- nova
- shimmer

## Single-Voice Narration

For simple text-to-speech conversion, you can use the `narrate` tool:

```python
# Convert text directly to speech
narrate(
    text="Your text to convert to speech",
    output_path="/path/to/output.wav"
)

# Convert text from a file
narrate(
    text_file_path="/path/to/text_file.txt",
    output_path="/path/to/output.wav"
)
```

## Usage

To use this MCP , simply ask to talk to you or start a voice conversation:

1. Start a conversation by saying something like:
   ```
   "Let's talk using voice"
   "Can we have a voice conversation?"
   "I'd like to speak instead of typing"
   ```

2. automatically launch the speech interface and start listening for your voice input.

3. It will speak the response aloud and then automatically listen for your next input.

4. The conversation continues naturally with alternating speaking and listening, just like talking to a person.

## UI Features

The PyQt-based UI includes:

- **Modern Dark Theme**: Sleek, professional appearance
- **Audio Visualization**: Dynamic visualization of audio input
- **Voice Selection**: Choose from multiple voice options
- **Voice Persistence**: Your voice preference is saved between sessions
- **Status Indicators**: Clear indication of system state (ready, listening, processing)

## Configuration

User preferences are stored in `~/.config/speech-mcp/config.json` and include:

- Selected TTS voice
- TTS engine preference
- Voice speed
- Language code
- UI theme settings

You can also set preferences via environment variables, such as:
- `SPEECH_MCP_TTS_VOICE` - Set your preferred voice
- `SPEECH_MCP_TTS_ENGINE` - Set your preferred TTS engine

## License

[MIT License](LICENSE)
