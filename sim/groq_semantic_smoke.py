import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim.groq_semantic_client import DEFAULT_MODEL, GroqSemanticClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a Groq multimodal semantic-decision test on a single image."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a local image file.",
    )
    parser.add_argument(
        "--mission",
        default="Drive from the marked start point to the marked destination while staying safe.",
        help="Mission text passed to the semantic runtime prompt.",
    )
    parser.add_argument(
        "--context",
        default="",
        help="Extra scene or route context appended to the prompt.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Groq model ID.",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/groq_semantic",
        help="Directory where the JSON artifact will be saved.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    client = GroqSemanticClient(model=args.model)
    decision = client.semantic_decision(
        image_path=str(image_path),
        mission=args.mission,
        extra_context=args.context,
    )

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_path = output_dir / f"groq_semantic_smoke_{timestamp}.json"

    payload = {
        "timestamp_utc": timestamp,
        "image_path": str(image_path),
        "mission": args.mission,
        "context": args.context,
        "model": decision["model"],
        "parsed": decision["parsed"],
        "usage": decision["usage"],
        "raw_text": decision["raw_text"],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(json.dumps(payload, indent=2))
    print(f"\nSaved artifact: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
