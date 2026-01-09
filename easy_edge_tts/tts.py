"""
Voice Forge - TTS Engines

Text-to-Speech backends:
- EdgeTTS: Free, high-quality Microsoft voices
- ElevenLabsTTS: Premium, most natural voices (requires API key)
"""

import logging
from pathlib import Path
from dataclasses import dataclass

from .voices import get_voice_id, EDGE_VOICES
from .utils import get_audio_duration, estimate_duration

logger = logging.getLogger(__name__)


@dataclass
class SentenceTiming:
    """Timing information for a sentence boundary from TTS."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds

    @property
    def duration(self) -> float:
        """Duration of the sentence in seconds."""
        return self.end - self.start

    def __repr__(self) -> str:
        return f"SentenceTiming({self.text[:30]!r}..., {self.start:.2f}s-{self.end:.2f}s)"


@dataclass
class TTSResult:
    """Result from TTS generation."""
    audio_path: Path
    duration: float
    voice: str
    backend: str

    def __repr__(self) -> str:
        return f"TTSResult({self.audio_path.name}, {self.duration:.1f}s, {self.voice})"


@dataclass
class TTSResultWithSentences:
    """Result from TTS generation with sentence-level timing."""
    audio_path: Path
    duration: float
    voice: str
    backend: str
    sentences: list[SentenceTiming]

    def __repr__(self) -> str:
        return f"TTSResultWithSentences({self.audio_path.name}, {self.duration:.1f}s, {len(self.sentences)} sentences)"

    def get_sentence_at_time(self, time: float) -> SentenceTiming | None:
        """Get the sentence being spoken at a given time."""
        for sentence in self.sentences:
            if sentence.start <= time < sentence.end:
                return sentence
        return None

    def to_subtitle_segments(self) -> list[dict]:
        """Convert sentence timings to subtitle segment format."""
        return [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in self.sentences
        ]


class EdgeTTS:
    """
    Microsoft Edge TTS - FREE and high quality.

    Uses the edge-tts library. Install with: pip install edge-tts

    Attributes:
        voice: Voice ID or short name (e.g., "guy", "en-US-GuyNeural")

    Example:
        >>> tts = EdgeTTS(voice="aria")
        >>> result = await tts.generate("Hello world!", "output.mp3")
        >>> print(f"Generated {result.duration:.1f}s of audio")
    """

    def __init__(self, voice: str = "guy"):
        """
        Initialize Edge TTS.

        Args:
            voice: Voice short name (e.g., "guy", "jenny", "aria")
                   or full ID (e.g., "en-US-GuyNeural")
        """
        self.voice = get_voice_id(voice)
        self._voice_name = voice

    async def generate(
        self,
        text: str,
        output_path: Path | str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> TTSResult:
        """
        Generate speech from text.

        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            rate: Speed adjustment ("-50%" to "+100%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz")

        Returns:
            TTSResult with path, duration, and metadata

        Raises:
            ImportError: If edge-tts is not installed
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Run: pip install edge-tts"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate audio
        communicate = edge_tts.Communicate(
            text,
            self.voice,
            rate=rate,
            pitch=pitch,
        )

        await communicate.save(str(output_path))

        # Get actual duration
        duration = get_audio_duration(output_path)
        if duration <= 0:
            duration = estimate_duration(text)

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            voice=self.voice,
            backend="edge-tts"
        )

    async def generate_with_timestamps(
        self,
        text: str,
        output_path: Path | str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> tuple[TTSResult, list[dict]]:
        """
        Generate speech with word-level timestamps.

        Useful for creating synchronized subtitles.

        Args:
            text: Text to convert
            output_path: Where to save audio
            rate: Speed adjustment
            pitch: Pitch adjustment

        Returns:
            Tuple of (TTSResult, list of word timings)
            Each timing dict has: {"text": str, "start": float, "end": float}
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(text, self.voice, rate=rate, pitch=pitch)

        timestamps = []
        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    timestamps.append({
                        "text": chunk["text"],
                        "start": chunk["offset"] / 10_000_000,  # Convert to seconds
                        "end": (chunk["offset"] + chunk["duration"]) / 10_000_000,
                    })

        duration = get_audio_duration(output_path)
        if duration <= 0:
            duration = estimate_duration(text)

        result = TTSResult(
            audio_path=output_path,
            duration=duration,
            voice=self.voice,
            backend="edge-tts"
        )

        return result, timestamps

    async def generate_with_sentences(
        self,
        text: str,
        output_path: Path | str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> TTSResultWithSentences:
        """
        Generate speech with sentence-level timing information.

        This is ideal for synchronizing on-screen text with narration,
        as sentences provide natural reading boundaries.

        Args:
            text: Text to convert
            output_path: Where to save audio
            rate: Speed adjustment
            pitch: Pitch adjustment

        Returns:
            TTSResultWithSentences containing audio path, duration, and
            a list of SentenceTiming objects with start/end times for each sentence.

        Example:
            >>> tts = EdgeTTS(voice="aria")
            >>> result = await tts.generate_with_sentences(
            ...     "Hello world. This is a test.",
            ...     "output.mp3"
            ... )
            >>> for sentence in result.sentences:
            ...     print(f"{sentence.start:.2f}s: {sentence.text}")
            0.00s: Hello world.
            0.95s: This is a test.
        """
        try:
            import edge_tts
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        communicate = edge_tts.Communicate(text, self.voice, rate=rate, pitch=pitch)

        sentences: list[SentenceTiming] = []
        with open(output_path, "wb") as f:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    f.write(chunk["data"])
                elif chunk["type"] == "SentenceBoundary":
                    start = chunk["offset"] / 10_000_000  # Convert to seconds
                    end = (chunk["offset"] + chunk["duration"]) / 10_000_000
                    sentences.append(SentenceTiming(
                        text=chunk["text"],
                        start=start,
                        end=end,
                    ))

        duration = get_audio_duration(output_path)
        if duration <= 0:
            duration = estimate_duration(text)

        return TTSResultWithSentences(
            audio_path=output_path,
            duration=duration,
            voice=self.voice,
            backend="edge-tts",
            sentences=sentences,
        )

    @classmethod
    def list_voices(cls) -> list[str]:
        """List available voice short names."""
        return list(EDGE_VOICES.keys())

    def __repr__(self) -> str:
        return f"EdgeTTS(voice={self._voice_name!r})"


class ElevenLabsTTS:
    """
    ElevenLabs TTS - Premium, most natural voices.

    Requires an API key from https://elevenlabs.io

    Example:
        >>> tts = ElevenLabsTTS(api_key="your-key", voice_id="...")
        >>> result = await tts.generate("Hello world!", "output.mp3")
    """

    DEFAULT_MODEL = "eleven_monolingual_v1"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model: str = None,
    ):
        """
        Initialize ElevenLabs TTS.

        Args:
            api_key: Your ElevenLabs API key
            voice_id: Voice ID from your ElevenLabs account
            model: Model ID (default: eleven_monolingual_v1)
        """
        self.api_key = api_key
        self.voice_id = voice_id
        self.model = model or self.DEFAULT_MODEL

    async def generate(
        self,
        text: str,
        output_path: Path | str,
        stability: float = 0.5,
        similarity_boost: float = 0.75,
    ) -> TTSResult:
        """
        Generate speech using ElevenLabs API.

        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            stability: Voice stability (0-1, higher = more consistent)
            similarity_boost: Voice clarity (0-1, higher = clearer)

        Returns:
            TTSResult with path, duration, and metadata
        """
        import httpx

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}"

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                url,
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": self.model,
                    "voice_settings": {
                        "stability": stability,
                        "similarity_boost": similarity_boost,
                    }
                }
            )
            response.raise_for_status()
            output_path.write_bytes(response.content)

        duration = get_audio_duration(output_path)
        if duration <= 0:
            duration = estimate_duration(text)

        return TTSResult(
            audio_path=output_path,
            duration=duration,
            voice=self.voice_id,
            backend="elevenlabs"
        )

    def __repr__(self) -> str:
        return f"ElevenLabsTTS(voice_id={self.voice_id[:8]}...)"
