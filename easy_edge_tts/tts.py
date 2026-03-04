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
class WordTiming:
    """Timing information for a word boundary from TTS."""
    text: str
    start: float  # Start time in seconds
    end: float    # End time in seconds

    @property
    def duration(self) -> float:
        """Duration of the word in seconds."""
        return self.end - self.start

    def __repr__(self) -> str:
        return f"WordTiming({self.text!r}, {self.start:.2f}s-{self.end:.2f}s)"


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


@dataclass
class TTSResultWithTimings:
    """
    Result from TTS generation with both sentence and word-level timing.

    This provides the most flexible option for subtitle generation:
    - Use sentences for natural grouping
    - Use words for precise timing when splitting long sentences
    """
    audio_path: Path
    duration: float
    voice: str
    backend: str
    sentences: list[SentenceTiming]
    words: list[WordTiming]

    def __repr__(self) -> str:
        return f"TTSResultWithTimings({self.audio_path.name}, {self.duration:.1f}s, {len(self.sentences)} sentences, {len(self.words)} words)"

    def get_sentence_at_time(self, time: float) -> SentenceTiming | None:
        """Get the sentence being spoken at a given time."""
        for sentence in self.sentences:
            if sentence.start <= time < sentence.end:
                return sentence
        return None

    def get_word_at_time(self, time: float) -> WordTiming | None:
        """Get the word being spoken at a given time."""
        for word in self.words:
            if word.start <= time < word.end:
                return word
        return None

    def get_words_in_range(self, start: float, end: float) -> list[WordTiming]:
        """Get all words within a time range."""
        return [w for w in self.words if w.start >= start and w.end <= end]

    def get_words_for_sentence(self, sentence: SentenceTiming) -> list[WordTiming]:
        """Get all words that belong to a sentence."""
        return self.get_words_in_range(sentence.start, sentence.end)

    def to_subtitle_segments(self) -> list[dict]:
        """Convert sentence timings to subtitle segment format."""
        return [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in self.sentences
        ]

    def to_chunked_segments(
        self,
        max_chars: int = 70,
        max_lines: int = 2,
        chars_per_line: int = 35,
    ) -> list[dict]:
        """
        Convert to display-ready subtitle segments that fit on screen.

        Long sentences are split at word boundaries to ensure all content
        is displayed while maintaining sync with audio.

        Args:
            max_chars: Maximum total characters per segment
            max_lines: Maximum lines per segment
            chars_per_line: Characters per line (max_chars = max_lines * chars_per_line)

        Returns:
            List of subtitle segments with start, end, text keys
        """
        max_chars = max_lines * chars_per_line
        segments = []

        for sentence in self.sentences:
            # If sentence fits, use it directly
            if len(sentence.text) <= max_chars:
                segments.append({
                    "start": sentence.start,
                    "end": sentence.end,
                    "text": sentence.text
                })
            else:
                # Split long sentence using word boundaries
                words = self.get_words_for_sentence(sentence)
                if not words:
                    # Fallback: split by time proportion if no word boundaries
                    segments.extend(self._split_by_proportion(sentence, max_chars))
                else:
                    segments.extend(self._split_by_words(sentence, words, max_chars))

        return segments

    def _split_by_words(
        self,
        sentence: SentenceTiming,
        words: list[WordTiming],
        max_chars: int
    ) -> list[dict]:
        """Split a sentence into chunks using word boundaries."""
        chunks = []
        current_chunk_words = []
        current_length = 0

        for word in words:
            word_len = len(word.text)

            # Check if adding this word exceeds limit
            if current_length + word_len + (1 if current_chunk_words else 0) > max_chars:
                # Save current chunk
                if current_chunk_words:
                    chunks.append({
                        "start": current_chunk_words[0].start,
                        "end": current_chunk_words[-1].end,
                        "text": " ".join(w.text for w in current_chunk_words)
                    })
                current_chunk_words = [word]
                current_length = word_len
            else:
                current_chunk_words.append(word)
                current_length += word_len + (1 if len(current_chunk_words) > 1 else 0)

        # Don't forget the last chunk
        if current_chunk_words:
            chunks.append({
                "start": current_chunk_words[0].start,
                "end": current_chunk_words[-1].end,
                "text": " ".join(w.text for w in current_chunk_words)
            })

        return chunks

    def _split_by_proportion(
        self,
        sentence: SentenceTiming,
        max_chars: int
    ) -> list[dict]:
        """Fallback: split sentence by time proportion when no word boundaries."""
        text = sentence.text
        chunks = []

        # Split text into chunks
        words = text.split()
        current_chunk = []
        current_length = 0

        chunk_texts = []
        for word in words:
            if current_length + len(word) + 1 > max_chars and current_chunk:
                chunk_texts.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1

        if current_chunk:
            chunk_texts.append(" ".join(current_chunk))

        # Distribute time proportionally
        total_chars = sum(len(c) for c in chunk_texts)
        current_time = sentence.start

        for chunk_text in chunk_texts:
            proportion = len(chunk_text) / total_chars
            chunk_duration = sentence.duration * proportion
            chunks.append({
                "start": current_time,
                "end": current_time + chunk_duration,
                "text": chunk_text
            })
            current_time += chunk_duration

        return chunks


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
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> TTSResult:
        """
        Generate speech from text.

        Args:
            text: Text to convert to speech
            output_path: Where to save the audio file
            rate: Speed adjustment ("-50%" to "+100%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz")
            max_retries: Maximum retry attempts on failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 2.0)

        Returns:
            TTSResult with path, duration, and metadata

        Raises:
            ImportError: If edge-tts is not installed
        """
        import asyncio

        try:
            import edge_tts
            from edge_tts.exceptions import NoAudioReceived
        except ImportError:
            raise ImportError(
                "edge-tts not installed. Run: pip install edge-tts"
            )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Log input for debugging
        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"[EdgeTTS] generate: voice={self.voice}, text_len={len(text)}")

        if not text or not text.strip():
            logger.error(f"[EdgeTTS] EMPTY TEXT provided!")
            raise ValueError("Cannot generate TTS for empty text")

        last_error = None

        for attempt in range(max_retries):
            try:
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

                logger.info(f"[EdgeTTS] Generated {duration:.1f}s audio")

                return TTSResult(
                    audio_path=output_path,
                    duration=duration,
                    voice=self.voice,
                    backend="edge-tts"
                )

            except NoAudioReceived as e:
                last_error = e
                logger.warning(f"[EdgeTTS] NoAudioReceived attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"[EdgeTTS] Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[EdgeTTS] All retries failed. Text: {text_preview!r}")

            except Exception as e:
                last_error = e
                logger.error(f"[EdgeTTS] Error: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))
                else:
                    logger.error(f"[EdgeTTS] All retries failed")

        raise last_error

    async def generate_with_timestamps(
        self,
        text: str,
        output_path: Path | str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> tuple[TTSResult, list[dict]]:
        """
        Generate speech with word-level timestamps.

        Useful for creating synchronized subtitles.

        Args:
            text: Text to convert
            output_path: Where to save audio
            rate: Speed adjustment
            pitch: Pitch adjustment
            max_retries: Maximum retry attempts on failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 2.0)

        Returns:
            Tuple of (TTSResult, list of word timings)
            Each timing dict has: {"text": str, "start": float, "end": float}
        """
        import asyncio

        try:
            import edge_tts
            from edge_tts.exceptions import NoAudioReceived
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"[EdgeTTS] generate_with_timestamps: voice={self.voice}, text_len={len(text)}")

        if not text or not text.strip():
            logger.error(f"[EdgeTTS] EMPTY TEXT provided!")
            raise ValueError("Cannot generate TTS for empty text")

        last_error = None

        for attempt in range(max_retries):
            try:
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

                logger.info(f"[EdgeTTS] Generated {duration:.1f}s with {len(timestamps)} word timestamps")

                result = TTSResult(
                    audio_path=output_path,
                    duration=duration,
                    voice=self.voice,
                    backend="edge-tts"
                )

                return result, timestamps

            except NoAudioReceived as e:
                last_error = e
                logger.warning(f"[EdgeTTS] NoAudioReceived attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"[EdgeTTS] Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[EdgeTTS] All retries failed. Text: {text_preview!r}")

            except Exception as e:
                last_error = e
                logger.error(f"[EdgeTTS] Error: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

        raise last_error

    async def generate_with_sentences(
        self,
        text: str,
        output_path: Path | str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        max_retries: int = 3,
        retry_delay: float = 2.0,
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
            max_retries: Maximum retry attempts on failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 2.0)

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
        import asyncio

        try:
            import edge_tts
            from edge_tts.exceptions import NoAudioReceived
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        text_preview = text[:100] + "..." if len(text) > 100 else text
        logger.info(f"[EdgeTTS] generate_with_sentences: voice={self.voice}, text_len={len(text)}")

        if not text or not text.strip():
            logger.error(f"[EdgeTTS] EMPTY TEXT provided!")
            raise ValueError("Cannot generate TTS for empty text")

        last_error = None

        for attempt in range(max_retries):
            try:
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

                logger.info(f"[EdgeTTS] Generated {duration:.1f}s with {len(sentences)} sentences")

                return TTSResultWithSentences(
                    audio_path=output_path,
                    duration=duration,
                    voice=self.voice,
                    backend="edge-tts",
                    sentences=sentences,
                )

            except NoAudioReceived as e:
                last_error = e
                logger.warning(f"[EdgeTTS] NoAudioReceived attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"[EdgeTTS] Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[EdgeTTS] All retries failed. Text: {text_preview!r}")

            except Exception as e:
                last_error = e
                logger.error(f"[EdgeTTS] Error: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay * (attempt + 1))

        raise last_error

    async def generate_with_timings(
        self,
        text: str,
        output_path: Path | str,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> TTSResultWithTimings:
        """
        Generate speech with both sentence and word-level timing.

        This is the most flexible option for subtitle generation:
        - Sentences provide natural grouping boundaries
        - Words enable precise splitting of long sentences

        Use result.to_chunked_segments() to get display-ready subtitles
        that fit on screen while staying in sync with audio.

        Args:
            text: Text to convert
            output_path: Where to save audio
            rate: Speed adjustment
            pitch: Pitch adjustment
            max_retries: Maximum retry attempts on failure (default: 3)
            retry_delay: Delay between retries in seconds (default: 2.0)

        Returns:
            TTSResultWithTimings containing audio path, duration,
            sentence timings, and word timings.

        Example:
            >>> tts = EdgeTTS(voice="aria")
            >>> result = await tts.generate_with_timings(
            ...     "This is a very long sentence that needs splitting.",
            ...     "output.mp3"
            ... )
            >>> # Get display-ready chunks (max 70 chars each)
            >>> segments = result.to_chunked_segments(max_chars=70)
            >>> for seg in segments:
            ...     print(f"{seg['start']:.2f}s: {seg['text']}")
        """
        import asyncio

        try:
            import edge_tts
            from edge_tts.exceptions import NoAudioReceived
        except ImportError:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Log input for debugging
        text_preview = text[:100] + "..." if len(text) > 100 else text
        text_preview_clean = text_preview.replace('\n', ' ')
        logger.info(f"[EdgeTTS] generate_with_timings called")
        logger.info(f"[EdgeTTS] Voice: {self.voice}, Rate: {rate}, Pitch: {pitch}")
        logger.info(f"[EdgeTTS] Text length: {len(text)} chars")
        logger.info(f"[EdgeTTS] Text preview: {text_preview_clean!r}")

        # Check for empty or whitespace-only text
        if not text or not text.strip():
            logger.error(f"[EdgeTTS] EMPTY TEXT provided! text={text!r}")
            raise ValueError("Cannot generate TTS for empty text")

        last_error = None

        for attempt in range(max_retries):
            try:
                logger.info(f"[EdgeTTS] Attempt {attempt + 1}/{max_retries}...")

                communicate = edge_tts.Communicate(text, self.voice, rate=rate, pitch=pitch)

                sentences: list[SentenceTiming] = []
                words: list[WordTiming] = []
                audio_bytes_received = 0

                with open(output_path, "wb") as f:
                    async for chunk in communicate.stream():
                        if chunk["type"] == "audio":
                            f.write(chunk["data"])
                            audio_bytes_received += len(chunk["data"])
                        elif chunk["type"] == "SentenceBoundary":
                            start = chunk["offset"] / 10_000_000
                            end = (chunk["offset"] + chunk["duration"]) / 10_000_000
                            sentences.append(SentenceTiming(
                                text=chunk["text"],
                                start=start,
                                end=end,
                            ))
                        elif chunk["type"] == "WordBoundary":
                            start = chunk["offset"] / 10_000_000
                            end = (chunk["offset"] + chunk["duration"]) / 10_000_000
                            words.append(WordTiming(
                                text=chunk["text"],
                                start=start,
                                end=end,
                            ))

                logger.info(f"[EdgeTTS] Stream completed: {audio_bytes_received} bytes, {len(sentences)} sentences, {len(words)} words")

                duration = get_audio_duration(output_path)
                if duration <= 0:
                    duration = estimate_duration(text)
                    logger.warning(f"[EdgeTTS] Could not read duration from file, estimated: {duration:.1f}s")
                else:
                    logger.info(f"[EdgeTTS] Audio duration: {duration:.1f}s")

                return TTSResultWithTimings(
                    audio_path=output_path,
                    duration=duration,
                    voice=self.voice,
                    backend="edge-tts",
                    sentences=sentences,
                    words=words,
                )

            except NoAudioReceived as e:
                last_error = e
                logger.warning(f"[EdgeTTS] NoAudioReceived on attempt {attempt + 1}/{max_retries}")
                logger.warning(f"[EdgeTTS] Voice: {self.voice}")
                logger.warning(f"[EdgeTTS] Text ({len(text)} chars): {text_preview_clean!r}")

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)  # Exponential backoff
                    logger.info(f"[EdgeTTS] Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[EdgeTTS] All {max_retries} attempts failed with NoAudioReceived")
                    logger.error(f"[EdgeTTS] FULL TEXT THAT FAILED:\n{text}")

            except Exception as e:
                last_error = e
                logger.error(f"[EdgeTTS] Unexpected error on attempt {attempt + 1}: {type(e).__name__}: {e}")

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.info(f"[EdgeTTS] Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"[EdgeTTS] All {max_retries} attempts failed")
                    logger.error(f"[EdgeTTS] FULL TEXT THAT FAILED:\n{text}")

        # If we get here, all retries failed
        raise last_error

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
