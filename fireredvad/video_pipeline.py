import json
import logging
import os
from dataclasses import asdict, dataclass

import numpy as np
import soundfile as sf

from fireredvad.onnx_infer import (
    FireRedAedOnnx,
    FireRedAedOnnxConfig,
    FireRedVadOnnx,
    FireRedVadOnnxConfig,
)

logger = logging.getLogger(__name__)


def _require_av():
    try:
        import av
    except ImportError as exc:
        raise ImportError(
            "PyAV is required for video ingestion. Install it with "
            "`uv sync --extra onnx` or "
            "`uv sync --extra onnx-gpu`."
        ) from exc
    return av


@dataclass
class FireRedVideoPipelineConfig:
    include_gap_segments: bool = True
    min_gap_duration_s: float = 0.3
    speech_ratio_threshold: float = 0.55
    singing_ratio_threshold: float = 0.35
    music_ratio_threshold: float = 0.45
    music_background_threshold: float = 0.2
    save_audio_segments: bool = True
    save_full_audio: bool = True


class VideoAudioExtractor:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def extract(self, video_path):
        av = _require_av()

        with av.open(video_path) as container:
            audio_stream = next(iter(container.streams.audio), None)
            if audio_stream is None:
                raise ValueError(f"No audio stream found in video: {video_path}")

            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=self.sample_rate,
            )
            chunks = []
            for packet in container.demux(audio_stream):
                for frame in packet.decode():
                    resampled_frames = resampler.resample(frame)
                    if resampled_frames is None:
                        continue
                    if not isinstance(resampled_frames, list):
                        resampled_frames = [resampled_frames]
                    for resampled in resampled_frames:
                        chunks.append(np.asarray(resampled.to_ndarray()).reshape(-1))

            flushed_frames = resampler.resample(None)
            if flushed_frames is not None:
                if not isinstance(flushed_frames, list):
                    flushed_frames = [flushed_frames]
                for flushed in flushed_frames:
                    chunks.append(np.asarray(flushed.to_ndarray()).reshape(-1))

        if not chunks:
            raise ValueError(f"Decoded audio is empty: {video_path}")

        audio = np.concatenate(chunks).astype(np.int16, copy=False)
        return audio, self.sample_rate


class FireRedVideoPipeline:
    BUSINESS_ACTIONS = {
        "speech": "transcribe",
        "speech_with_music": "transcribe_with_bgm",
        "singing": "review_for_lyrics",
        "music": "skip_as_bgm",
        "silence": "drop",
        "uncertain": "manual_review",
    }

    @classmethod
    def from_pretrained(
        cls,
        model_dir,
        vad_config=None,
        aed_config=None,
        pipeline_config=None,
    ):
        vad = FireRedVadOnnx.from_pretrained(model_dir, vad_config or FireRedVadOnnxConfig())
        aed = FireRedAedOnnx.from_pretrained(model_dir, aed_config or FireRedAedOnnxConfig())
        return cls(
            vad=vad,
            aed=aed,
            extractor=VideoAudioExtractor(),
            config=pipeline_config or FireRedVideoPipelineConfig(),
        )

    def __init__(self, vad, aed, extractor, config):
        self.vad = vad
        self.aed = aed
        self.extractor = extractor
        self.config = config

    def analyze(self, video_path, output_dir=None):
        wav_np, sample_rate = self.extractor.extract(video_path)
        audio = (wav_np, sample_rate)

        vad_result, _ = self.vad.detect(audio)
        aed_result, _ = self.aed.detect(audio)
        timeline = self._build_timeline(vad_result["timestamps"], aed_result["event2timestamps"], vad_result["dur"])

        result = {
            "input_path": video_path,
            "audio_duration": vad_result["dur"],
            "pipeline_config": asdict(self.config),
            "vad": vad_result,
            "aed": aed_result,
            "timeline": timeline,
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(video_path))[0]
            if self.config.save_full_audio:
                audio_path = os.path.join(output_dir, f"{stem}.wav")
                sf.write(audio_path, wav_np, samplerate=sample_rate)
                result["audio_path"] = audio_path
            if self.config.save_audio_segments:
                segment_dir = os.path.join(output_dir, "segments")
                os.makedirs(segment_dir, exist_ok=True)
                segment_paths = self._save_audio_segments(wav_np, sample_rate, timeline, segment_dir, stem)
                timeline = [
                    {**item, "audio_segment_path": segment_paths[item["segment_id"]]}
                    for item in timeline
                ]
                result["timeline"] = timeline
            manifest_path = os.path.join(output_dir, f"{stem}.segments.json")
            with open(manifest_path, "w", encoding="utf-8") as fout:
                json.dump(result, fout, ensure_ascii=False, indent=2)
            result["manifest_path"] = manifest_path

        return result

    def _build_timeline(self, speech_segments, event2timestamps, total_duration):
        speech_segments = sorted(speech_segments)
        timeline = []
        last_end = 0.0
        segment_index = 0

        for start_s, end_s in speech_segments:
            if self.config.include_gap_segments and start_s - last_end >= self.config.min_gap_duration_s:
                timeline.append(self._build_segment(segment_index, "gap", last_end, start_s, event2timestamps))
                segment_index += 1
            timeline.append(self._build_segment(segment_index, "speech", start_s, end_s, event2timestamps))
            segment_index += 1
            last_end = end_s

        if self.config.include_gap_segments and total_duration - last_end >= self.config.min_gap_duration_s:
            timeline.append(self._build_segment(segment_index, "gap", last_end, total_duration, event2timestamps))
        elif not speech_segments:
            timeline.append(self._build_segment(segment_index, "gap", 0.0, total_duration, event2timestamps))

        return timeline

    def _build_segment(self, index, segment_type, start_s, end_s, event2timestamps):
        ratios = {
            event: round(self._overlap_ratio(start_s, end_s, timestamps), 3)
            for event, timestamps in event2timestamps.items()
        }
        label = self._decide_label(segment_type, ratios)
        return {
            "segment_id": index,
            "segment_type": segment_type,
            "start": round(start_s, 3),
            "end": round(end_s, 3),
            "duration": round(end_s - start_s, 3),
            "label": label,
            "event_ratios": ratios,
            "business_action": self.BUSINESS_ACTIONS[label],
        }

    def _decide_label(self, segment_type, ratios):
        if ratios["singing"] >= self.config.singing_ratio_threshold:
            return "singing"
        if segment_type == "speech":
            if (
                ratios["speech"] >= self.config.speech_ratio_threshold
                and ratios["music"] >= self.config.music_background_threshold
            ):
                return "speech_with_music"
            if ratios["speech"] >= self.config.speech_ratio_threshold:
                return "speech"
            if ratios["music"] >= self.config.music_ratio_threshold:
                return "music"
            return "uncertain"

        if ratios["music"] >= self.config.music_ratio_threshold:
            return "music"
        if ratios["speech"] >= self.config.speech_ratio_threshold:
            return "speech"
        return "silence"

    def _overlap_ratio(self, start_s, end_s, timestamps):
        duration = max(0.0, end_s - start_s)
        if duration <= 0:
            return 0.0

        overlap = 0.0
        for event_start, event_end in timestamps:
            left = max(start_s, event_start)
            right = min(end_s, event_end)
            if right > left:
                overlap += right - left
        return min(overlap / duration, 1.0)

    def _save_audio_segments(self, wav_np, sample_rate, timeline, segment_dir, stem):
        segment_paths = {}
        for item in timeline:
            start = int(item["start"] * sample_rate)
            end = int(item["end"] * sample_rate)
            segment_name = f"{stem}_{item['segment_id']:04d}_{item['label']}.wav"
            segment_path = os.path.join(segment_dir, segment_name)
            sf.write(segment_path, wav_np[start:end], samplerate=sample_rate)
            segment_paths[item["segment_id"]] = segment_path
        return segment_paths
