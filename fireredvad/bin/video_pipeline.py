#!/usr/bin/env python3

import argparse
import json
import logging

from fireredvad.onnx_infer import FireRedAedOnnxConfig, FireRedVadOnnxConfig
from fireredvad.video_pipeline import FireRedVideoPipeline, FireRedVideoPipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)
logger = logging.getLogger("fireredvad.bin.video_pipeline")


def build_parser():
    parser = argparse.ArgumentParser(description="Split a video with FireRedVAD ONNX and classify segments with AED ONNX.")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--model_dir", type=str, default="pretrained_models/onnx_models")
    parser.add_argument("--output_dir", type=str, default="out/video_pipeline")
    parser.add_argument("--use_gpu", type=int, default=0)
    parser.add_argument("--include_gap_segments", type=int, default=1)
    parser.add_argument("--min_gap_duration_s", type=float, default=0.3)
    parser.add_argument("--speech_ratio_threshold", type=float, default=0.55)
    parser.add_argument("--singing_ratio_threshold", type=float, default=0.35)
    parser.add_argument("--music_ratio_threshold", type=float, default=0.45)
    parser.add_argument("--music_background_threshold", type=float, default=0.2)
    parser.add_argument("--save_audio_segments", type=int, default=1)
    parser.add_argument("--save_full_audio", type=int, default=1)

    parser.add_argument("--smooth_window_size", type=int, default=5)
    parser.add_argument("--speech_threshold", type=float, default=0.4)
    parser.add_argument("--singing_threshold", type=float, default=0.5)
    parser.add_argument("--music_threshold", type=float, default=0.5)
    parser.add_argument("--min_speech_frame", type=int, default=20)
    parser.add_argument("--max_speech_frame", type=int, default=2000)
    parser.add_argument("--min_event_frame", type=int, default=20)
    parser.add_argument("--max_event_frame", type=int, default=2000)
    parser.add_argument("--min_silence_frame", type=int, default=20)
    parser.add_argument("--merge_silence_frame", type=int, default=0)
    parser.add_argument("--extend_speech_frame", type=int, default=0)
    parser.add_argument("--chunk_max_frame", type=int, default=30000)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    vad_config = FireRedVadOnnxConfig(
        use_gpu=bool(args.use_gpu),
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        min_speech_frame=args.min_speech_frame,
        max_speech_frame=args.max_speech_frame,
        min_silence_frame=args.min_silence_frame,
        merge_silence_frame=args.merge_silence_frame,
        extend_speech_frame=args.extend_speech_frame,
        chunk_max_frame=args.chunk_max_frame,
    )
    aed_config = FireRedAedOnnxConfig(
        use_gpu=bool(args.use_gpu),
        smooth_window_size=args.smooth_window_size,
        speech_threshold=args.speech_threshold,
        singing_threshold=args.singing_threshold,
        music_threshold=args.music_threshold,
        min_event_frame=args.min_event_frame,
        max_event_frame=args.max_event_frame,
        min_silence_frame=args.min_silence_frame,
        merge_silence_frame=args.merge_silence_frame,
        extend_speech_frame=args.extend_speech_frame,
        chunk_max_frame=args.chunk_max_frame,
    )
    pipeline_config = FireRedVideoPipelineConfig(
        include_gap_segments=bool(args.include_gap_segments),
        min_gap_duration_s=args.min_gap_duration_s,
        speech_ratio_threshold=args.speech_ratio_threshold,
        singing_ratio_threshold=args.singing_ratio_threshold,
        music_ratio_threshold=args.music_ratio_threshold,
        music_background_threshold=args.music_background_threshold,
        save_audio_segments=bool(args.save_audio_segments),
        save_full_audio=bool(args.save_full_audio),
    )

    pipeline = FireRedVideoPipeline.from_pretrained(
        model_dir=args.model_dir,
        vad_config=vad_config,
        aed_config=aed_config,
        pipeline_config=pipeline_config,
    )
    result = pipeline.analyze(args.video_path, output_dir=args.output_dir)
    logger.info("Result: %s", json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
