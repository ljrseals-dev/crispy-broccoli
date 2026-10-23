"""Legacy kNN-VC engine (MIT).

Loads via torch.hub when available. Models cached under ~/.cache/aoc_magic/.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from aoc_magic_v3.errors import VcError
from aoc_magic_v3.vc.base import VCEngine

logger = logging.getLogger(__name__)

_TORCH_AVAILABLE = False
_TORCHAUDIO_AVAILABLE = False
try:
    import torch  # noqa: F401
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore

try:
    import torchaudio  # noqa: F401
    _TORCHAUDIO_AVAILABLE = True
except ImportError:
    pass

_model_cache: Optional[object] = None


def _load_knn_vc(device: str = "cpu"):
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    if not _TORCH_AVAILABLE:
        raise VcError("torch is required for kNN-VC")
    try:
        # One-time torch.hub fetch allowed by §7.4 / §2
        knn = torch.hub.load(
            "bshall/knn-vc",
            "knn_vc",
            prematched=True,
            trust_repo=True,
            verbose=False,
        )
        knn = knn.to(device)
        _model_cache = knn
        return knn
    except Exception as exc:
        raise VcError(f"Failed to load kNN-VC: {exc}") from exc


class KnnVC(VCEngine):
    name = "knn"

    def is_available(self) -> bool:
        return _TORCH_AVAILABLE and _TORCHAUDIO_AVAILABLE

    def convert(
        self,
        source: np.ndarray,
        source_sr: int,
        reference_paths: List[Path],
        *,
        topk: int = 4,
        device: str = "cpu",
    ) -> tuple:
        if not self.is_available():
            raise VcError("kNN-VC unavailable (torch/torchaudio missing)")
        if not reference_paths:
            raise VcError("kNN-VC requires at least one reference path")

        try:
            import torch
            import librosa

            model = _load_knn_vc(device)
            # Resample source to 16 kHz (WavLM expectation)
            src = source.astype(np.float32)
            if source_sr != 16000:
                src = librosa.resample(src, orig_sr=source_sr, target_sr=16000, res_type="soxr_hq")

            query = torch.from_numpy(src).float().to(device)

            matching_set = []
            for ref_path in reference_paths:
                ref, ref_sr = librosa.load(str(ref_path), sr=None, mono=True)
                if ref_sr != 16000:
                    ref = librosa.resample(ref, orig_sr=ref_sr, target_sr=16000, res_type="soxr_hq")
                matching_set.append(torch.from_numpy(ref.astype(np.float32)).float().to(device))

            with torch.inference_mode():
                out = model(query, matching_set, topk=topk)

            audio = out.squeeze().cpu().numpy().astype(np.float32)
            # HiFi-GAN output is typically 16 kHz for kNN-VC
            return audio, 16000
        except VcError:
            raise
        except Exception as exc:
            raise VcError(f"kNN-VC conversion failed: {exc}") from exc
