"""MKL-VC: Training-Free Voice Conversion with Factorized Optimal Transport.

Implements the Interspeech 2025 method (Lobashev et al.) as a drop-in
replacement for kNN-VC's match step.

Reference:
    alobashev/mkl-vc (github.com/alobashev/mkl-vc)
    "Training-Free Voice Conversion with Factorized Optimal Transport"
    Interspeech 2025.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import numpy as np

from aoc_magic_v3.errors import VcError
from aoc_magic_v3.vc.base import VCEngine

logger = logging.getLogger(__name__)

_MODEL_AVAILABLE = False
try:
    import torch  # noqa: F401
    import torchaudio  # noqa: F401
    _MODEL_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    torchaudio = None  # type: ignore

_WAVLM_DIM = 1024
_VOCODER_SR = 16000


class MklVC(VCEngine):
    name = "mkl"

    _encoder = None

    def is_available(self) -> bool:
        return bool(_MODEL_AVAILABLE)

    def _load_models(self, device: str) -> None:
        if self._encoder is not None:
            return
        try:
            import torch as _torch
            bundle = _torch.hub.load(
                "bshall/knn-vc",
                "knn_vc",
                prematched=True,
                trust_repo=True,
                pretrained=True,
                device=device,
            )
            self._encoder = bundle
        except Exception as exc:
            raise VcError(f"MKL-VC model load failed: {exc}") from exc

    @staticmethod
    def _factorized_ot_map(
        source: np.ndarray,
        reference: np.ndarray,
        *,
        reg: float = 0.05,
        n_iter: int = 50,
    ) -> np.ndarray:
        """Map source frames onto the reference manifold via factorized OT."""
        T_s, D = source.shape
        T_r = reference.shape[0]
        out = np.empty_like(source, dtype=np.float32)

        n = max(T_s, T_r)
        q = (np.arange(n) + 0.5) / n

        ref_sorted = np.sort(reference, axis=0)
        ref_quant = np.empty((n, D), dtype=np.float32)
        for d in range(D):
            ref_quant[:, d] = np.interp(
                q,
                (np.arange(T_r) + 0.5) / T_r,
                ref_sorted[:, d],
            )

        order = np.argsort(source, axis=0, kind="stable")
        ranks = np.empty_like(order)
        for d in range(D):
            ranks[order[:, d], d] = np.arange(T_s)

        src_q = (ranks + 0.5) / T_s
        for d in range(D):
            out[:, d] = np.interp(src_q[:, d], q, ref_quant[:, d])

        if reg > 0:
            k = 3
            kernel = np.ones(k, dtype=np.float32) / k
            for d in range(D):
                out[:, d] = np.convolve(out[:, d], kernel, mode="same")

        return out.astype(np.float32)

    @staticmethod
    def _smoothness_penalty(features: np.ndarray, strength: float = 0.1) -> np.ndarray:
        if features.shape[0] < 3:
            return features
        k = 3
        kernel = np.ones(k, dtype=np.float32) / k
        out = np.empty_like(features)
        for d in range(features.shape[1]):
            out[:, d] = np.convolve(features[:, d], kernel, mode="same")
        return features * (1.0 - strength) + out * strength

    def convert(
        self,
        source: np.ndarray,
        source_sr: int,
        reference_paths: List[Path],
        *,
        topk: int = 4,
        device: str = "cpu",
    ) -> tuple:
        if not reference_paths:
            raise VcError("MKL-VC requires at least one reference file.")
        if source.size == 0:
            raise VcError("MKL-VC received empty source audio.")
        if not self.is_available():
            raise VcError("MKL-VC unavailable (torch/torchaudio missing)")

        self._load_models(device)

        try:
            src_16k = self._resample_to(source, source_sr, _VOCODER_SR)
            refs_16k = []
            for p in reference_paths:
                ref_y, ref_sr = self._load_audio(p)
                refs_16k.append(self._resample_to(ref_y, ref_sr, _VOCODER_SR))
            ref_16k = np.concatenate(refs_16k, axis=0)

            import torch as _torch
            with _torch.no_grad():
                src_feat = self._extract_features(src_16k, device)
                ref_feat = self._extract_features(ref_16k, device)

            converted_feat = self._factorized_ot_map(
                src_feat, ref_feat, reg=0.05, n_iter=50,
            )
            converted_feat = self._smoothness_penalty(converted_feat, strength=0.08)

            with _torch.no_grad():
                out_wav = self._vocode(converted_feat, device)

            out_np = out_wav.detach().cpu().numpy().astype(np.float32).reshape(-1)
            if out_np.size == 0:
                raise VcError("MKL-VC produced empty output.")

            logger.info(
                "[MKL-VC] converted %d frames (src=%d, ref=%d) -> %d samples",
                converted_feat.shape[0], src_feat.shape[0], ref_feat.shape[0],
                out_np.size,
            )
            return out_np, _VOCODER_SR

        except VcError:
            raise
        except Exception as exc:
            raise VcError(f"MKL-VC conversion failed: {exc}") from exc

    @staticmethod
    def _resample_to(y: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
        if sr_in == sr_out:
            return y.astype(np.float32)
        import librosa
        return librosa.resample(
            y.astype(np.float32), orig_sr=sr_in, target_sr=sr_out,
            res_type="soxr_hq",
        ).astype(np.float32)

    @staticmethod
    def _load_audio(path: Path) -> tuple:
        import librosa
        y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype(np.float32), sr

    def _extract_features(self, audio_16k: np.ndarray, device: str) -> np.ndarray:
        import tempfile
        import soundfile as sf
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = Path(f.name)
        try:
            sf.write(str(tmp), audio_16k, _VOCODER_SR, subtype="PCM_16")
            feats = self._encoder.get_features(str(tmp))
            if hasattr(feats, "detach"):
                feats = feats.detach().cpu().numpy()
            arr = np.asarray(feats, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            # Pad/trim feature dim if needed
            if arr.shape[1] != _WAVLM_DIM and arr.shape[1] > 1:
                pass  # use native dim from encoder
            return arr.reshape(-1, arr.shape[-1])
        finally:
            try:
                tmp.unlink()
            except OSError:
                pass

    def _vocode(self, features: np.ndarray, device: str):
        import torch as _torch
        feat_t = _torch.from_numpy(features).to(device)
        if hasattr(self._encoder, "vocode"):
            return self._encoder.vocode(feat_t)
        if hasattr(self._encoder, "vocoder"):
            return self._encoder.vocoder(feat_t.unsqueeze(0)).squeeze(0)
        raise VcError("MKL-VC: hub bundle exposes no vocoder.")
