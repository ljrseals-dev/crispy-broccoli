"""Tkinter GUI for aoc_magic_v3.

Launch with: python -m aoc_magic_v3.gui
Must not import torch, TTS, kokoro_onnx, or piper directly.
Preserves every widget from V2 GUI and adds engine/device dropdowns.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import List

from aoc_magic_v3.config import Config
from aoc_magic_v3.logging import configure_logging
from aoc_magic_v3.pipeline import run_pipeline


class AocMagicGui:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("AOC Magic V3 – Offline Voice Generator")
        root.geometry("700x720")
        root.resizable(True, True)

        self.output_file = tk.StringVar(value="aoc_style_final.wav")
        self.ref_files: List[str] = []
        self.dsp_enabled = tk.BooleanVar(value=True)
        self.pitch = tk.DoubleVar(value=2.0)
        self.speed = tk.DoubleVar(value=1.15)
        self.tts_rate = tk.IntVar(value=260)
        self.tts_volume = tk.DoubleVar(value=1.0)
        self.comp_threshold = tk.DoubleVar(value=-20.0)
        self.comp_ratio = tk.DoubleVar(value=4.0)
        self.drive = tk.DoubleVar(value=1.5)
        self.topk = tk.IntVar(value=4)
        self.sample_rate = tk.IntVar(value=44100)
        self.tts_engine_var = tk.StringVar(value="auto")
        self.vc_engine_var = tk.StringVar(value="auto")
        self.device_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="Ready")

        self._build()

    def _build(self) -> None:
        # Text
        frame_text = ttk.LabelFrame(self.root, text="Text to speak", padding=10)
        frame_text.pack(fill="both", expand=True, padx=10, pady=5)
        self.text_entry = tk.Text(frame_text, height=5, wrap="word")
        self.text_entry.pack(fill="both", expand=True)
        self.text_entry.insert("1.0", "We need bold action now!")

        # References
        frame_refs = ttk.LabelFrame(self.root, text="Reference voices (optional)", padding=10)
        frame_refs.pack(fill="x", padx=10, pady=5)
        self.ref_listbox = tk.Listbox(frame_refs, height=3)
        self.ref_listbox.pack(side="left", fill="both", expand=True, padx=(0, 5))
        btn_refs = ttk.Frame(frame_refs)
        btn_refs.pack(side="right", fill="y")
        ttk.Button(btn_refs, text="Add", command=self._add_refs).pack(fill="x", pady=2)
        ttk.Button(btn_refs, text="Remove", command=self._remove_ref).pack(fill="x", pady=2)
        ttk.Button(btn_refs, text="Clear", command=self._clear_refs).pack(fill="x", pady=2)

        # DSP params
        frame_params = ttk.LabelFrame(self.root, text="Character DSP", padding=10)
        frame_params.pack(fill="x", padx=10, pady=5)
        self._param_row(frame_params, "Pitch (semitones)", self.pitch, -12, 12, 0.1)
        self._param_row(frame_params, "Speed (1.0=normal, >1 faster)", self.speed, 0.5, 2.0, 0.05)
        self._param_row(frame_params, "TTS Rate (words/min)", self.tts_rate, 50, 500, 1)
        self._param_row(frame_params, "TTS Volume (0-1)", self.tts_volume, 0.0, 1.0, 0.05)
        self._param_row(frame_params, "Compression Threshold (dB)", self.comp_threshold, -60, 0, 0.5)
        self._param_row(frame_params, "Compression Ratio", self.comp_ratio, 1, 20, 0.5)
        self._param_row(frame_params, "Drive (dB)", self.drive, 0, 10, 0.5)
        self._param_row(frame_params, "kNN-VC topk", self.topk, 1, 10, 1)

        frame_sr = ttk.Frame(frame_params)
        frame_sr.pack(fill="x", pady=2)
        ttk.Label(frame_sr, text="Sample Rate:").pack(side="left")
        ttk.Combobox(
            frame_sr,
            textvariable=self.sample_rate,
            values=[16000, 22050, 32000, 44100, 48000],
            state="readonly",
            width=10,
        ).pack(side="left", padx=5)
        ttk.Checkbutton(frame_sr, text="Enable DSP", variable=self.dsp_enabled).pack(
            side="left", padx=20
        )

        # Engine selectors (V3)
        frame_eng = ttk.LabelFrame(self.root, text="Engines (V3)", padding=10)
        frame_eng.pack(fill="x", padx=10, pady=5)
        row = ttk.Frame(frame_eng)
        row.pack(fill="x")
        ttk.Label(row, text="TTS Engine").pack(side="left")
        ttk.Combobox(
            row,
            textvariable=self.tts_engine_var,
            values=["auto", "pyttsx3", "kokoro", "xtts", "piper"],
            state="readonly",
            width=12,
        ).pack(side="left", padx=5)
        ttk.Label(row, text="VC Engine").pack(side="left", padx=(15, 0))
        ttk.Combobox(
            row,
            textvariable=self.vc_engine_var,
            values=["auto", "mkl", "knn", "none"],
            state="readonly",
            width=10,
        ).pack(side="left", padx=5)
        ttk.Label(row, text="Device").pack(side="left", padx=(15, 0))
        ttk.Combobox(
            row,
            textvariable=self.device_var,
            values=["auto", "cpu", "cuda", "mps"],
            state="readonly",
            width=8,
        ).pack(side="left", padx=5)

        # Output
        frame_out = ttk.LabelFrame(self.root, text="Output", padding=10)
        frame_out.pack(fill="x", padx=10, pady=5)
        ttk.Entry(frame_out, textvariable=self.output_file).pack(
            side="left", fill="x", expand=True, padx=(0, 5)
        )
        ttk.Button(frame_out, text="Browse...", command=self._browse_out).pack(side="right")

        # Run
        frame_run = ttk.Frame(self.root)
        frame_run.pack(fill="x", padx=10, pady=10)
        self.run_button = ttk.Button(
            frame_run, text="Generate Magic", command=self._run, width=18
        )
        self.run_button.pack(side="left")
        ttk.Button(frame_run, text="List Engines", command=self._list_engines).pack(
            side="left", padx=6
        )
        self.progress = ttk.Progressbar(frame_run, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        ttk.Label(self.root, textvariable=self.status_var, anchor="w").pack(
            fill="x", padx=10, pady=(0, 10)
        )

    def _param_row(self, parent, label, var, min_val, max_val, step):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=32).pack(side="left")
        ttk.Spinbox(
            row, from_=min_val, to=max_val, increment=step, textvariable=var, width=10
        ).pack(side="left", padx=5)

    def _add_refs(self):
        files = filedialog.askopenfilenames(
            title="Select reference audio files",
            filetypes=[("Audio files", "*.wav *.mp3 *.flac *.ogg"), ("All files", "*.*")],
        )
        for f in files:
            if f not in self.ref_files:
                self.ref_files.append(f)
        self._update_ref_list()

    def _remove_ref(self):
        sel = self.ref_listbox.curselection()
        if sel:
            self.ref_files.pop(sel[0])
            self._update_ref_list()

    def _clear_refs(self):
        self.ref_files = []
        self._update_ref_list()

    def _update_ref_list(self):
        self.ref_listbox.delete(0, tk.END)
        for f in self.ref_files:
            self.ref_listbox.insert(tk.END, os.path.basename(f))

    def _browse_out(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav")],
            initialfile=self.output_file.get(),
        )
        if path:
            self.output_file.set(path)

    def _list_engines(self):
        import io
        from contextlib import redirect_stdout
        from aoc_magic_v3.cli import _print_engines

        buf = io.StringIO()
        with redirect_stdout(buf):
            _print_engines()
        messagebox.showinfo("Engines", buf.getvalue())

    def _run(self):
        text = self.text_entry.get("1.0", tk.END).strip()
        if not text:
            messagebox.showerror("Error", "Please enter some text to speak.")
            return
        out = self.output_file.get().strip()
        if not out:
            messagebox.showerror("Error", "Please specify an output file path.")
            return

        refs = [Path(p) for p in self.ref_files]
        cfg = Config(
            text=text,
            out=Path(out),
            ref=refs,
            tts_rate=self.tts_rate.get(),
            tts_volume=self.tts_volume.get(),
            sample_rate=self.sample_rate.get(),
            pitch=self.pitch.get(),
            speed=self.speed.get(),
            comp_threshold=self.comp_threshold.get(),
            comp_ratio=self.comp_ratio.get(),
            drive_db=self.drive.get(),
            topk=self.topk.get(),
            tts_engine=self.tts_engine_var.get(),
            vc_engine=self.vc_engine_var.get(),
            use_dsp=self.dsp_enabled.get(),
            device=self.device_var.get(),
        )

        self.run_button.config(state="disabled")
        self.progress.start()
        self.status_var.set("Processing...")

        def worker():
            try:
                configure_logging("info")
                result = run_pipeline(cfg)
                self.root.after(0, lambda: self._ok(result))
            except Exception as exc:
                self.root.after(0, lambda: self._err(exc))

        threading.Thread(target=worker, daemon=True).start()

    def _ok(self, out: Path):
        self.progress.stop()
        self.run_button.config(state="normal")
        self.status_var.set("Done!")
        messagebox.showinfo("Success", f"Output saved to:\n{out}")

    def _err(self, exc: Exception):
        self.progress.stop()
        self.run_button.config(state="normal")
        self.status_var.set(f"Error: {exc}")
        messagebox.showerror("Error", str(exc))


def main() -> None:
    root = tk.Tk()
    AocMagicGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()
