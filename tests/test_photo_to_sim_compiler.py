"""
Smoke test for the capture-to-sim compiler.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from perception.pipeline import PipelineConfig
from world_model.photo_to_sim import PhotoToSimCompiler, PhotoToSimCompilerConfig


def main():
    repo_root = Path(__file__).resolve().parent.parent
    manifest = repo_root / "test_data" / "photo_to_sim_manifest.json"

    config = PhotoToSimCompilerConfig(
        pipeline=PipelineConfig(
            use_sim_tagger=True,
            depth_fallback=True,
            min_confidence=0.0,
        )
    )
    compiler = PhotoToSimCompiler(config=config)

    with tempfile.TemporaryDirectory() as tmp_dir:
        compiled = compiler.compile_manifest(manifest, tmp_dir)

        assert compiled.artifact_paths is not None
        assert compiled.artifact_paths.scene_graph_json.exists()
        assert compiled.artifact_paths.mujoco_xml.exists()
        assert compiled.artifact_paths.randomization_json.exists()
        assert compiled.artifact_paths.compiler_report_json.exists()

        xml = compiled.artifact_paths.mujoco_xml.read_text(encoding="utf-8")
        assert "<mujoco model=" in xml
        assert "support_surface" in xml

    print("photo_to_sim compiler smoke test passed")


if __name__ == "__main__":
    main()
