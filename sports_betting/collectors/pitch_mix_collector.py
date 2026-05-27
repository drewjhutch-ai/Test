"""
Pitch mix change detector.
MLB Stats API has no pitch-type breakdown data. Returns empty dict always.
Layer 21 handles an empty result gracefully with a neutral score.
"""
from __future__ import annotations


def get_pitch_mix_changes() -> dict[str, dict]:
    """MLB Stats API has no pitch-type data. Layer 21 scores neutral when empty."""
    return {}
