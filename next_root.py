#!/usr/bin/env python3
"""next_root — pick the next chord root by scale degree.

Small helper used by the DAW bus / voice_ui flows to translate a spoken
phrase like "next root up a fifth" or "go to the relative minor" into a
concrete pitch class, so an LLM intent can be resolved without guessing.

Public API:
    next_root(current, degree=1, mode="major", direction=1) -> str
    parse_direction(phrase) -> int
"""

import re

SCALES = {
    "major": [0, 2, 4, 5, 7, 9, 11],
    "minor": [0, 2, 3, 5, 7, 8, 10],
    "dorian": [0, 2, 3, 5, 7, 9, 10],
    "phrygian": [0, 1, 3, 5, 7, 8, 10],
    "lydian": [0, 2, 4, 6, 7, 9, 11],
    "mixolydian": [0, 2, 4, 5, 7, 9, 10],
    "locrian": [0, 1, 3, 5, 6, 8, 10],
}

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]
NAME_TO_PC = {n: i for i, n in enumerate(PITCH_NAMES)}
NAME_TO_PC.update({"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10})
NAME_TO_PC.update({n.lower(): pc for n, pc in list(NAME_TO_PC.items())})

DEGREE_WORDS = {
    "unison": 1, "tonic": 1, "root": 1,
    "second": 2, "third": 3, "fourth": 4, "fifth": 5,
    "sixth": 6, "seventh": 7, "octave": 8,
    "relative": 6,
}

_UP_WORDS = {"up", "higher", "raise", "ascend", "next"}
_DOWN_WORDS = {"down", "lower", "drop", "descend", "previous", "prev"}


def parse_direction(phrase):
    """Return +1 for up/next, -1 for down/previous; default +1."""
    if not phrase:
        return 1
    tokens = re.findall(r"[a-z]+", phrase.lower())
    for t in tokens:
        if t in _DOWN_WORDS:
            return -1
    return 1


def _parse_degree(phrase):
    """Extract a scale-degree step from a natural-language phrase."""
    if not phrase:
        return 1
    low = phrase.lower()
    m = re.search(r"\b(\d+)\b", low)
    if m:
        try:
            return max(1, min(8, int(m.group(1))))
        except ValueError:
            pass
    for word, deg in DEGREE_WORDS.items():
        if word in low:
            return deg
    return 1


def next_root(current, degree=1, mode="major", direction=1):
    """Return the pitch-class name reached by stepping `degree` scale
    degrees from `current` in `mode`. direction=+1 up, -1 down."""
    if isinstance(degree, str):
        degree = _parse_degree(degree)
    if current not in NAME_TO_PC:
        raise ValueError("unknown pitch: %r" % current)
    scale = SCALES.get(mode.lower())
    if scale is None:
        raise ValueError("unknown mode: %r" % mode)

    tonic = NAME_TO_PC[current]
    steps = (int(degree) - 1) * int(direction)
    octave, idx = divmod(steps, 7)
    semis = scale[idx % 7] + 12 * octave
    if steps < 0 and idx != 0:
        semis = scale[(steps % 7)] + 12 * (steps // 7)
    pc = (tonic + semis) % 12
    return PITCH_NAMES[pc]


def _selftest():
    assert next_root("C", 1, "major") == "C"
    assert next_root("C", 2, "major") == "D"
    assert next_root("C", 5, "major") == "G"
    assert next_root("C", 6, "major") == "A"
    assert next_root("C", 3, "major", direction=-1) == "A"
    assert next_root("A", 3, "minor") == "C"
    assert parse_direction("go down a fourth") == -1
    assert parse_direction("next chord") == 1
    assert _parse_degree("up a fifth") == 5
    assert _parse_degree("to the relative minor") == 6
    print("next_root selftest OK")


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        _selftest()
    else:
        cur = sys.argv[1]
        deg = sys.argv[2] if len(sys.argv) > 2 else 1
        mode = sys.argv[3] if len(sys.argv) > 3 else "major"
        direction = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        print(next_root(cur, deg, mode, direction))
