"""Exceptions with user-friendly messages. The UI shows str(error) directly."""


class PipelineError(Exception):
    """Base class: anything that should stop the pipeline with a clear message."""


class InvalidInput(PipelineError):
    pass


class InvalidYouTubeURL(InvalidInput):
    pass


class TranscriptError(PipelineError):
    """Captions could not be fetched.

    `reason` is one of: disabled, not_found, empty, blocked, unavailable,
    age_restricted, invalid, other.
    """

    # Reasons where downloading the audio and running speech-to-text may still work.
    STT_RECOVERABLE = {"disabled", "not_found", "empty", "blocked", "other"}

    def __init__(self, message: str, reason: str = "other"):
        super().__init__(message)
        self.reason = reason

    @property
    def stt_can_help(self) -> bool:
        return self.reason in self.STT_RECOVERABLE


class AudioError(PipelineError):
    pass


class LLMError(PipelineError):
    pass
