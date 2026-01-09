"""Tests for easy-edge-tts."""

import pytest
from pathlib import Path
from easy_edge_tts import (
    EdgeTTS,
    VoiceRotator,
    EDGE_VOICES,
    VOICE_MOODS,
    get_audio_duration,
    SentenceTiming,
    TTSResultWithSentences,
)
from easy_edge_tts.voices import get_voice_id, list_voices, get_voices_for_mood


class TestVoices:
    """Test voice definitions."""

    def test_edge_voices_exist(self):
        """Verify voice definitions."""
        assert len(EDGE_VOICES) > 10
        assert "guy" in EDGE_VOICES
        assert "jenny" in EDGE_VOICES
        assert "aria" in EDGE_VOICES

    def test_get_voice_id_short_name(self):
        """Test getting voice ID from short name."""
        assert get_voice_id("guy") == "en-US-GuyNeural"
        assert get_voice_id("aria") == "en-US-AriaNeural"

    def test_get_voice_id_full_name(self):
        """Test passing through full voice ID."""
        assert get_voice_id("en-US-GuyNeural") == "en-US-GuyNeural"

    def test_list_voices(self):
        """Test listing voices."""
        voices = list_voices()
        assert "guy" in voices
        assert "jenny" in voices

    def test_list_voices_category(self):
        """Test listing voices by category."""
        male_voices = list_voices("male")
        assert "guy" in male_voices
        assert "jenny" not in male_voices

    def test_voice_moods_exist(self):
        """Verify mood mappings."""
        assert "dramatic" in VOICE_MOODS
        assert "happy" in VOICE_MOODS
        assert "aita" in VOICE_MOODS

    def test_get_voices_for_mood(self):
        """Test mood-based voice selection."""
        voices = get_voices_for_mood("dramatic")
        assert len(voices) > 0
        assert all(v in EDGE_VOICES for v in voices)


class TestEdgeTTS:
    """Test EdgeTTS class."""

    def test_init_short_name(self):
        """Test initialization with short name."""
        tts = EdgeTTS(voice="guy")
        assert tts.voice == "en-US-GuyNeural"

    def test_init_full_name(self):
        """Test initialization with full voice ID."""
        tts = EdgeTTS(voice="en-US-AriaNeural")
        assert tts.voice == "en-US-AriaNeural"

    def test_list_voices(self):
        """Test class method to list voices."""
        voices = EdgeTTS.list_voices()
        assert len(voices) > 10

    def test_repr(self):
        """Test string representation."""
        tts = EdgeTTS(voice="aria")
        assert "aria" in repr(tts)


class TestVoiceRotator:
    """Test VoiceRotator class."""

    def test_init_default(self):
        """Test default initialization."""
        rotator = VoiceRotator()
        assert len(rotator.voices) > 0

    def test_init_custom_voices(self):
        """Test custom voice list."""
        rotator = VoiceRotator(voices=["guy", "jenny"])
        assert rotator.voices == ["guy", "jenny"]

    def test_get_random_voice(self):
        """Test random voice selection."""
        rotator = VoiceRotator(voices=["guy", "jenny", "aria"])
        voice = rotator.get_random_voice()
        assert voice in ["guy", "jenny", "aria"]

    def test_get_random_tts(self):
        """Test getting random TTS instance."""
        rotator = VoiceRotator()
        tts = rotator.get_random_tts()
        assert isinstance(tts, EdgeTTS)

    def test_get_next_voice_rotation(self):
        """Test sequential rotation."""
        rotator = VoiceRotator(voices=["guy", "jenny", "aria"])

        v1 = rotator.get_next_voice()
        v2 = rotator.get_next_voice()
        v3 = rotator.get_next_voice()
        v4 = rotator.get_next_voice()

        assert v1 == "guy"
        assert v2 == "jenny"
        assert v3 == "aria"
        assert v4 == "guy"  # Wraps around

    def test_reset_rotation(self):
        """Test resetting rotation."""
        rotator = VoiceRotator(voices=["guy", "jenny"])

        rotator.get_next_voice()
        rotator.get_next_voice()
        rotator.reset_rotation()

        assert rotator.get_next_voice() == "guy"

    def test_get_voice_for_mood(self):
        """Test mood-based selection."""
        rotator = VoiceRotator()
        voice = rotator.get_voice_for_mood("dramatic")
        assert voice in VOICE_MOODS["dramatic"]

    def test_get_tts_for_mood(self):
        """Test getting TTS for mood."""
        rotator = VoiceRotator()
        tts = rotator.get_tts_for_mood("happy")
        assert isinstance(tts, EdgeTTS)

    def test_list_moods(self):
        """Test listing available moods."""
        rotator = VoiceRotator()
        moods = rotator.list_moods()
        assert "dramatic" in moods
        assert "happy" in moods


class TestUtils:
    """Test utility functions."""

    def test_get_audio_duration_nonexistent(self):
        """Test duration detection with nonexistent file."""
        duration = get_audio_duration("/nonexistent/file.mp3")
        assert duration == 0.0


class TestSentenceTiming:
    """Test SentenceTiming dataclass."""

    def test_creation(self):
        """Test creating a SentenceTiming."""
        timing = SentenceTiming(text="Hello world.", start=0.0, end=1.5)
        assert timing.text == "Hello world."
        assert timing.start == 0.0
        assert timing.end == 1.5

    def test_duration_property(self):
        """Test duration calculation."""
        timing = SentenceTiming(text="Test", start=1.0, end=3.5)
        assert timing.duration == 2.5

    def test_repr(self):
        """Test string representation."""
        timing = SentenceTiming(text="Hello world.", start=0.0, end=1.5)
        assert "Hello world." in repr(timing)
        assert "0.00s" in repr(timing)


class TestTTSResultWithSentences:
    """Test TTSResultWithSentences dataclass."""

    def test_creation(self):
        """Test creating a result with sentences."""
        sentences = [
            SentenceTiming("Hello.", 0.0, 1.0),
            SentenceTiming("World.", 1.0, 2.0),
        ]
        result = TTSResultWithSentences(
            audio_path=Path("/tmp/test.mp3"),
            duration=2.0,
            voice="en-US-GuyNeural",
            backend="edge-tts",
            sentences=sentences,
        )
        assert len(result.sentences) == 2
        assert result.duration == 2.0

    def test_get_sentence_at_time(self):
        """Test finding sentence at specific time."""
        sentences = [
            SentenceTiming("Hello.", 0.0, 1.0),
            SentenceTiming("World.", 1.0, 2.0),
        ]
        result = TTSResultWithSentences(
            audio_path=Path("/tmp/test.mp3"),
            duration=2.0,
            voice="en-US-GuyNeural",
            backend="edge-tts",
            sentences=sentences,
        )

        sentence = result.get_sentence_at_time(0.5)
        assert sentence.text == "Hello."

        sentence = result.get_sentence_at_time(1.5)
        assert sentence.text == "World."

        sentence = result.get_sentence_at_time(3.0)
        assert sentence is None

    def test_to_subtitle_segments(self):
        """Test converting to subtitle format."""
        sentences = [
            SentenceTiming("Hello.", 0.0, 1.0),
            SentenceTiming("World.", 1.0, 2.0),
        ]
        result = TTSResultWithSentences(
            audio_path=Path("/tmp/test.mp3"),
            duration=2.0,
            voice="en-US-GuyNeural",
            backend="edge-tts",
            sentences=sentences,
        )

        segments = result.to_subtitle_segments()
        assert len(segments) == 2
        assert segments[0] == {"start": 0.0, "end": 1.0, "text": "Hello."}
        assert segments[1] == {"start": 1.0, "end": 2.0, "text": "World."}

    def test_repr(self):
        """Test string representation."""
        sentences = [SentenceTiming("Hello.", 0.0, 1.0)]
        result = TTSResultWithSentences(
            audio_path=Path("/tmp/test.mp3"),
            duration=1.0,
            voice="en-US-GuyNeural",
            backend="edge-tts",
            sentences=sentences,
        )
        assert "1 sentences" in repr(result)
