"""livecaption: local real-time English transcription + Chinese translation CLI.

For macOS / Apple Silicon.

ASR uses mlx-audio to run NVIDIA nemotron-3.5-asr-streaming-0.6b (true streaming, Apple GPU),
endpoint detection uses Silero VAD; translation uses mlx-lm to run Hy-MT2. Audio sources support
microphone and system audio (meeting output).
"""

__version__ = "0.1.0"
