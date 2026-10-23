"""V3 CLI accepts all V2 flags and new flags."""

from __future__ import annotations

from aoc_magic_v3.cli import build_parser


def test_parser_accepts_v2_flags():
    p = build_parser()
    args = p.parse_args(
        [
            "hello world",
            "--out", "x.wav",
            "--ref", "a.wav", "b.wav",
            "--pitch", "1.5",
            "--speed", "1.1",
            "--tts-rate", "180",
            "--tts-volume", "0.8",
            "--sr", "22050",
            "--comp-threshold", "-18",
            "--comp-ratio", "3",
            "--drive", "1.0",
            "--topk", "4",
            "--no-dsp",
        ]
    )
    assert args.text == "hello world"
    assert str(args.out) == "x.wav"
    assert args.pitch == 1.5
    assert args.speed == 1.1
    assert args.tts_rate == 180
    assert args.tts_volume == 0.8
    assert args.sr == 22050
    assert args.no_dsp is True
    assert args.ref == ["a.wav", "b.wav"]


def test_parser_new_flags():
    p = build_parser()
    args = p.parse_args(
        [
            "hi",
            "--tts-engine", "pyttsx3",
            "--vc-engine", "none",
            "--device", "cpu",
            "--seed", "42",
            "--log-level", "debug",
        ]
    )
    assert args.tts_engine == "pyttsx3"
    assert args.vc_engine == "none"
    assert args.device == "cpu"
    assert args.seed == 42
    assert args.log_level == "debug"


def test_list_engines_no_text_required():
    p = build_parser()
    args = p.parse_args(["--list-engines"])
    assert args.list_engines is True
    assert args.text is None


def test_defaults_match_v2():
    p = build_parser()
    args = p.parse_args(["hello"])
    assert args.tts_engine == "auto"
    assert args.vc_engine == "auto"
    assert args.seed == 0
    assert args.sr == 44100
    assert args.pitch == 2.0
    assert args.speed == 1.15
    assert args.tts_rate == 260
    assert args.drive == 1.5
    assert str(args.out) == "aoc_style_final.wav"
