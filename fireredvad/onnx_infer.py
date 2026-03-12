import logging
import os
from dataclasses import dataclass

import numpy as np

from fireredvad.core.audio_feat import AudioFeat
from fireredvad.core.vad_postprocessor import VadPostprocessor

logger = logging.getLogger(__name__)

CUDA_RUNTIME_HELP = (
    "Install GPU runtimes with `uv sync --extra onnx-gpu` "
    "(the `onnx-gpu` extra installs `onnxruntime-gpu[cuda,cudnn]`), "
    "or install CUDA 12.8-compatible CUDA 12.x + cuDNN 9.x system runtimes."
)


def _require_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise ImportError(
            "onnxruntime is required for ONNX inference. "
            "Install it with `uv sync --extra onnx` "
            "or `uv sync --extra onnx-gpu`."
        ) from exc
    return ort


def _resolve_model_paths(model_dir_or_path, default_model_name):
    package_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

    if os.path.splitext(model_dir_or_path)[1].lower() == ".onnx":
        model_candidates = [
            model_dir_or_path,
            os.path.join(package_root, "pretrained_models", "onnx_models", os.path.basename(model_dir_or_path)),
        ]
        model_candidates = list(dict.fromkeys(os.path.normpath(p) for p in model_candidates))
        for model_path in model_candidates:
            cmvn_path = os.path.join(os.path.dirname(model_path), "cmvn.ark")
            if os.path.isfile(model_path) and os.path.isfile(cmvn_path):
                return model_path, cmvn_path
        raise FileNotFoundError(
            "ONNX model not found. Tried: "
            + ", ".join(model_candidates)
        )

    dir_candidates = [
        model_dir_or_path,
        os.path.join(model_dir_or_path, "onnx_models"),
        os.path.join(model_dir_or_path, "pretrained_models", "onnx_models"),
        os.path.join(package_root, model_dir_or_path),
        os.path.join(package_root, "onnx_models"),
        os.path.join(package_root, "pretrained_models", "onnx_models"),
    ]
    dir_candidates = list(dict.fromkeys(os.path.normpath(p) for p in dir_candidates))

    for model_dir in dir_candidates:
        model_path = os.path.join(model_dir, default_model_name)
        cmvn_path = os.path.join(model_dir, "cmvn.ark")
        if os.path.isfile(model_path) and os.path.isfile(cmvn_path):
            return model_path, cmvn_path

    raise FileNotFoundError(
        "ONNX model directory not found. "
        f"Expected `{default_model_name}` and `cmvn.ark`. Tried: "
        + ", ".join(dir_candidates)
    )


def _build_session(model_path, use_gpu=False):
    ort = _require_onnxruntime()
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    available_providers = ort.get_available_providers()

    if use_gpu:
        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception as exc:
                logger.warning("Failed to preload CUDA/cuDNN DLLs: %s", exc)

        if "CUDAExecutionProvider" not in available_providers:
            logger.warning("CUDAExecutionProvider is unavailable. Falling back to CPUExecutionProvider.")
        else:
            try:
                session = ort.InferenceSession(
                    model_path,
                    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    sess_options=session_options,
                )
            except Exception as exc:
                logger.warning(
                    "CUDAExecutionProvider failed to initialize: %s Falling back to CPUExecutionProvider. %s",
                    exc,
                    CUDA_RUNTIME_HELP,
                )
            else:
                if "CUDAExecutionProvider" not in session.get_providers():
                    logger.warning(
                        "CUDAExecutionProvider was requested but is inactive. Active providers: %s. %s",
                        session.get_providers(),
                        CUDA_RUNTIME_HELP,
                    )
                return session

    return ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
        sess_options=session_options,
    )


def _run_chunked(session, feats, chunk_max_frame):
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    if feats.shape[0] <= chunk_max_frame:
        outputs = session.run([output_name], {input_name: feats[None, :, :].astype(np.float32)})[0]
        return outputs[0]

    logger.warning("Too long input, split every %s frames", chunk_max_frame)
    chunk_probs = []
    for start in range(0, feats.shape[0], chunk_max_frame):
        end = start + chunk_max_frame
        chunk = feats[start:end]
        outputs = session.run([output_name], {input_name: chunk[None, :, :].astype(np.float32)})[0]
        chunk_probs.append(outputs[0])
    return np.concatenate(chunk_probs, axis=0)


@dataclass
class FireRedVadOnnxConfig:
    use_gpu: bool = False
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    min_speech_frame: int = 20
    max_speech_frame: int = 2000  # 20s
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000  # 300s

    def __post_init__(self):
        if self.speech_threshold < 0 or self.speech_threshold > 1:
            raise ValueError("speech_threshold must be in [0, 1]")
        if self.min_speech_frame <= 0:
            raise ValueError("min_speech_frame must be positive")


class FireRedVadOnnx:
    @classmethod
    def from_pretrained(cls, model_dir_or_path, config=None):
        if config is None:
            config = FireRedVadOnnxConfig()

        model_path, cmvn_path = _resolve_model_paths(model_dir_or_path, "fireredvad_vad.onnx")
        session = _build_session(model_path, config.use_gpu)
        audio_feat = AudioFeat(cmvn_path)
        vad_postprocessor = VadPostprocessor(
            config.smooth_window_size,
            config.speech_threshold,
            config.min_speech_frame,
            config.max_speech_frame,
            config.min_silence_frame,
            config.merge_silence_frame,
            config.extend_speech_frame,
        )
        return cls(audio_feat, session, vad_postprocessor, config)

    def __init__(self, audio_feat, session, vad_postprocessor, config):
        self.audio_feat = audio_feat
        self.session = session
        self.vad_postprocessor = vad_postprocessor
        self.config = config

    def detect(self, audio, do_postprocess=True):
        feat, dur = self.audio_feat.extract(audio, return_tensor=False)
        probs = _run_chunked(self.session, feat, self.config.chunk_max_frame).squeeze(-1)

        if not do_postprocess:
            return None, probs

        decisions = self.vad_postprocessor.process(probs.tolist())
        starts_ends_s = self.vad_postprocessor.decision_to_segment(decisions, dur)
        result = {
            "dur": round(dur, 3),
            "timestamps": starts_ends_s,
        }
        if isinstance(audio, str):
            result["wav_path"] = audio
        return result, probs


@dataclass
class FireRedAedOnnxConfig:
    use_gpu: bool = False
    smooth_window_size: int = 5
    speech_threshold: float = 0.4
    singing_threshold: float = 0.5
    music_threshold: float = 0.5
    min_event_frame: int = 20
    max_event_frame: int = 2000  # 20s
    min_silence_frame: int = 20
    merge_silence_frame: int = 0
    extend_speech_frame: int = 0
    chunk_max_frame: int = 30000  # 300s


class FireRedAedOnnx:
    IDX2EVENT = {0: "speech", 1: "singing", 2: "music"}

    @classmethod
    def from_pretrained(cls, model_dir_or_path, config=None):
        if config is None:
            config = FireRedAedOnnxConfig()

        model_path, cmvn_path = _resolve_model_paths(model_dir_or_path, "fireredvad_aed.onnx")
        session = _build_session(model_path, config.use_gpu)
        audio_feat = AudioFeat(cmvn_path)

        event2postprocessor = {}
        for event in cls.IDX2EVENT.values():
            threshold = getattr(config, f"{event}_threshold")
            event2postprocessor[event] = VadPostprocessor(
                config.smooth_window_size,
                threshold,
                config.min_event_frame,
                config.max_event_frame,
                config.min_silence_frame,
                config.merge_silence_frame,
                config.extend_speech_frame,
            )
        return cls(audio_feat, session, event2postprocessor, config)

    def __init__(self, audio_feat, session, event2postprocessor, config):
        self.audio_feat = audio_feat
        self.session = session
        self.event2postprocessor = event2postprocessor
        self.config = config

    def detect(self, audio):
        feat, dur = self.audio_feat.extract(audio, return_tensor=False)
        probs = _run_chunked(self.session, feat, self.config.chunk_max_frame)
        if probs.ndim != 2 or probs.shape[-1] != len(self.IDX2EVENT):
            raise ValueError(f"Unexpected AED output shape: {probs.shape}")

        event2starts_ends_s = {}
        event2raw_ratio = {}
        for idx, event in self.IDX2EVENT.items():
            threshold = getattr(self.config, f"{event}_threshold")
            postprocessor = self.event2postprocessor[event]
            event_probs = probs[:, idx].tolist()
            decision = postprocessor.process(event_probs)
            starts_ends_s = postprocessor.decision_to_segment(decision, dur)
            event2starts_ends_s[event] = starts_ends_s
            raw_ratio = sum(int(p >= threshold) for p in event_probs) / len(event_probs) if event_probs else 0
            event2raw_ratio[event] = round(raw_ratio, 3)

        result = {
            "dur": round(dur, 3),
            "event2timestamps": event2starts_ends_s,
            "event2ratio": event2raw_ratio,
        }
        if isinstance(audio, str):
            result["wav_path"] = audio
        return result, probs
