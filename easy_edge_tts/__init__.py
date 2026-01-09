"""
easy-edge-tts - High-level TTS with voice rotation and mood selection.

A simple, powerful wrapper around Edge TTS and ElevenLabs for content creators.

Usage:
    from easy_edge_tts import speak, EdgeTTS, VoiceRotator

    # Simple one-liner
    await speak("Hello world", "output.mp3")

    # With voice rotation
    rotator = VoiceRotator()
    tts = rotator.get_tts_for_mood("dramatic")
    await tts.generate("The tension was unbearable...", "output.mp3")
"""

from .tts import (
    EdgeTTS,
    ElevenLabsTTS,
    TTSResult,
    TTSResultWithSentences,
    SentenceTiming,
)
from .rotator import VoiceRotator, get_voice_rotator
from .voices import EDGE_VOICES, VOICE_MOODS
from .utils import get_audio_duration

__version__ = "0.3.0"
__all__ = [
    "EdgeTTS",
    "ElevenLabsTTS",
    "TTSResult",
    "TTSResultWithSentences",
    "SentenceTiming",
    "VoiceRotator",
    "get_voice_rotator",
    "speak",
    "EDGE_VOICES",
    "VOICE_MOODS",
    "get_audio_duration",
]


async def speak(
    text: str,
    output_path: str,
    voice: str = "en-US-GuyNeural",
    rate: str = "+0%",
    pitch: str = "+0Hz",
) -> TTSResult:
    """
    Quick helper to generate speech with Edge TTS.

    Args:
        text: Text to speak
        output_path: Where to save audio file
        voice: Voice name (default: Guy, a natural male US voice)
        rate: Speed adjustment (e.g., "+10%", "-20%")
        pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz")

    Returns:
        TTSResult with path and duration

    Example:
        >>> await speak("Hello world", "hello.mp3")
        >>> await speak("Fast speech", "fast.mp3", rate="+20%")
    """
    from pathlib import Path
    tts = EdgeTTS(voice=voice)
    return await tts.generate(text, Path(output_path), rate=rate, pitch=pitch)
