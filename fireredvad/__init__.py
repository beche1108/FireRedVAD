# Copyright 2026 Xiaohongshu. (Author: Kaituo Xu, Wenpeng Li, Kai Huang, Kun Liu)

from importlib import import_module

__version__ = "0.0.2"

_LAZY_IMPORTS = {
    "FireRedVad": ("fireredvad.vad", "FireRedVad"),
    "FireRedVadConfig": ("fireredvad.vad", "FireRedVadConfig"),
    "FireRedAed": ("fireredvad.aed", "FireRedAed"),
    "FireRedAedConfig": ("fireredvad.aed", "FireRedAedConfig"),
    "FireRedStreamVad": ("fireredvad.stream_vad", "FireRedStreamVad"),
    "FireRedStreamVadConfig": ("fireredvad.stream_vad", "FireRedStreamVadConfig"),
}


def __getattr__(name):
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def non_stream_vad(wav_path, model_dir="pretrained_models/FireRedVAD/VAD", **kwargs):
    """Quick VAD inference"""
    from fireredvad.vad import FireRedVad, FireRedVadConfig

    config = FireRedVadConfig(**kwargs)
    vad = FireRedVad.from_pretrained(model_dir, config)
    result, probs = vad.detect(wav_path)
    return result


def stream_vad_full(wav_path, model_dir="pretrained_models/FireRedVAD/Stream-VAD", **kwargs):
    """Quick Stream VAD inference"""
    from fireredvad.stream_vad import FireRedStreamVad, FireRedStreamVadConfig

    config = FireRedStreamVadConfig(**kwargs)
    svad = FireRedStreamVad.from_pretrained(model_dir, config)
    frame_results, result = svad.detect_full(wav_path)
    return frame_results, result


def non_stream_aed(wav_path, model_dir="pretrained_models/FireRedVAD/AED", **kwargs):
    """Quick AED inference"""
    from fireredvad.aed import FireRedAed, FireRedAedConfig

    config = FireRedAedConfig(**kwargs)
    aed = FireRedAed.from_pretrained(model_dir, config)
    result, probs = aed.detect(wav_path)
    return result


def non_stream_vad_onnx(audio, model_dir="onnx_models", **kwargs):
    """Quick ONNX VAD inference"""
    from fireredvad.onnx_infer import FireRedVadOnnx, FireRedVadOnnxConfig

    config = FireRedVadOnnxConfig(**kwargs)
    vad = FireRedVadOnnx.from_pretrained(model_dir, config)
    result, probs = vad.detect(audio)
    return result


def non_stream_aed_onnx(audio, model_dir="onnx_models", **kwargs):
    """Quick ONNX AED inference"""
    from fireredvad.onnx_infer import FireRedAedOnnx, FireRedAedOnnxConfig

    config = FireRedAedOnnxConfig(**kwargs)
    aed = FireRedAedOnnx.from_pretrained(model_dir, config)
    result, probs = aed.detect(audio)
    return result


def analyze_video_with_onnx(video_path, model_dir="onnx_models", output_dir=None, **kwargs):
    """Split video by VAD and classify segments with AED."""
    from fireredvad.video_pipeline import FireRedVideoPipeline, FireRedVideoPipelineConfig

    pipeline = FireRedVideoPipeline.from_pretrained(
        model_dir=model_dir,
        pipeline_config=FireRedVideoPipelineConfig(**kwargs),
    )
    return pipeline.analyze(video_path, output_dir=output_dir)


__all__ = [
    '__version__',
    'FireRedVad',
    'FireRedVadConfig',
    'FireRedAed',
    'FireRedAedConfig', 
    'FireRedStreamVad',
    'FireRedStreamVadConfig',
    'non_stream_vad',
    'stream_vad_full',
    'non_stream_aed',
    'non_stream_vad_onnx',
    'non_stream_aed_onnx',
    'analyze_video_with_onnx',
]
