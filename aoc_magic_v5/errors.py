"""Exception hierarchy for aoc_magic_v3.

All exceptions raised by this package must be one of these.
Nothing else inherits from AocError.
"""


class AocError(Exception):
    """Base exception for aoc_magic_v3."""


class TTSError(AocError):
    """Raised on TTS synthesis failure."""


class EngineUnavailable(TTSError):
    """Raised when a requested TTS engine is not available."""


class VcError(AocError):
    """Raised on voice-conversion failure."""


class DspError(AocError):
    """Raised on DSP processing failure."""


class ConfigError(AocError):
    """Raised on configuration errors."""
