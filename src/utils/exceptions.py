"""Application errors with messages intended for the user interface."""

from __future__ import annotations


class TabDiffuseError(Exception):
    """Expected failure with a human-readable explanation.

    ``message`` is safe to show in the UI. It must not contain raw dataset rows.
    """

    def __init__(self, message: str) -> None:
        self.message = message
        super().__init__(message)
