# Speech Transcription Guide

The speech-mcp extension provides powerful transcription capabilities with support for timestamps and speaker detection.

## Basic Usage

1. Basic transcription:
```python
result = transcribe("video.mp4")
```

2. Transcription with timestamps:
```python
result = transcribe("video.mp4", include_timestamps=True)
```

3. Transcription with speaker detection:
```python
result = transcribe("video.mp4", detect_speakers=True)
```

## Output Files

The transcription is saved to two files:
- `{input_name}.transcript.txt`: Contains the transcription text
- `{input_name}.metadata.json`: Contains detailed metadata

### Example Transcript with Speakers

```
SPEAKER-AWARE TRANSCRIPT
=====================

[Speaker: SPEAKER_00]
[00:00:01] First person speaking here.

[Speaker: SPEAKER_01]
[00:00:05] Second person responds.

[Speaker: SPEAKER_00]
[00:00:10] First person continues the conversation.
```

### Example Metadata

```json
{
  "engine": "openai",
  "model": "Systran/faster-whisper-medium",
  "time_taken": 45.23,
  "language": "en",
  "language_probability": 0.98,
  "duration": 180.5,
  "has_timestamps": true,
  "has_speakers": true,
  "speakers": {
    "SPEAKER_00": {
      "talk_time": 95.5,
      "segments": 8,
      "first_appearance": 0.0,
      "last_appearance": 175.2
    },
    "SPEAKER_01": {
      "talk_time": 85.0,
      "segments": 7,
      "first_appearance": 5.3,
      "last_appearance": 160.8
    }
  },
  "speaker_changes": 15,
  "average_turn_duration": 12.03
}
```

## Supported Formats

Audio formats:
- WAV (.wav)
- MP3 (.mp3)
- M4A (.m4a)
- FLAC (.flac)
- AAC (.aac)
- OGG (.ogg)

Video formats:
- MP4 (.mp4)
- MOV (.mov)
- AVI (.avi)
- MKV (.mkv)
- WebM (.webm)

For video files, the audio track is automatically extracted for transcription.

## Speaker Detection

The speaker detection feature uses a lightweight heuristic approach that:

1. Detects potential speaker changes based on:
   - Pauses between segments (> 1 second)
   - Dialogue indicators in text ("said", "asked", etc.)
   - Punctuation patterns (?, !, :)

2. Tracks speaker statistics:
   - Talk time per speaker
   - Number of segments
   - First and last appearances
   - Speaker changes
   - Average turn duration

3. Limitations:
   - Basic heuristic approach, not as accurate as dedicated diarization
   - May not handle overlapping speech well
   - Cannot identify actual speakers, only distinguish between them
   - Works best with clear turn-taking in conversations

## Best Practices

1. Use high-quality audio input when possible
2. For long videos:
   - Files are processed locally
   - Output is saved to disk to handle large transcripts
   - Progress updates are provided during processing

3. When using speaker detection:
   - Works best with clear audio
   - Minimal background noise
   - Clear pauses between speakers
   - Natural conversation flow

4. Metadata provides:
   - Processing statistics
   - Language detection confidence
   - Timing information
   - Speaker analytics (if enabled)

## Example Usage in Code

```python
# Basic transcription
result = transcribe("interview.mp4")
print(result)  # Shows file locations and basic stats

# With timestamps
result = transcribe("meeting.mp4", include_timestamps=True)
print(result)  # Shows file locations and timing info

# With speaker detection
result = transcribe("conversation.mp4", detect_speakers=True)
print(result)  # Shows file locations and speaker stats

# The output files can be found at:
# - interview.transcript.txt
# - interview.metadata.json
```