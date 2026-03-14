"""
CLI for compiling a calibrated capture into MuJoCo artifacts.
"""
import argparse

from perception.pipeline import PipelineConfig
from world_model.photo_to_sim import PhotoToSimCompiler, PhotoToSimCompilerConfig


def main():
    parser = argparse.ArgumentParser(description="Compile a capture manifest to MuJoCo")
    parser.add_argument("manifest", help="Path to capture manifest JSON")
    parser.add_argument(
        "--output-dir",
        default="artifacts/photo_to_sim",
        help="Directory for exported scene artifacts",
    )
    parser.add_argument(
        "--vlm-url",
        default="http://localhost:8000/v1",
        help="OpenAI-compatible VLM endpoint",
    )
    parser.add_argument(
        "--vlm-model",
        default="Qwen/Qwen3-VL-8B",
        help="Vision-language model name",
    )
    parser.add_argument(
        "--sim-tagger",
        action="store_true",
        help="Use the deterministic sim tagger instead of a VLM",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.10,
        help="Minimum object confidence kept in the compiled scene",
    )
    args = parser.parse_args()

    config = PhotoToSimCompilerConfig(
        vlm_api_base=args.vlm_url,
        vlm_model=args.vlm_model,
        pipeline=PipelineConfig(
            use_sim_tagger=args.sim_tagger,
            depth_fallback=True,
            min_confidence=args.min_confidence,
        ),
    )
    compiler = PhotoToSimCompiler(config=config)
    compiled = compiler.compile_manifest(args.manifest, args.output_dir)

    print(f"[Compiler] Scene name: {compiled.scene_name}")
    print(f"[Compiler] Objects: {len(compiled.scene_graph.get_all_objects())}")
    if compiled.artifact_paths:
        print(f"[Compiler] Scene graph: {compiled.artifact_paths.scene_graph_json}")
        print(f"[Compiler] MuJoCo XML: {compiled.artifact_paths.mujoco_xml}")
        print(f"[Compiler] Randomization: {compiled.artifact_paths.randomization_json}")
        print(f"[Compiler] Report: {compiled.artifact_paths.compiler_report_json}")


if __name__ == "__main__":
    main()
