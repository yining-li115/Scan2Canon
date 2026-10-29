"""Canonicalize a mesh from verified OA V2 predictions and open visual semantics."""

import argparse
from pathlib import Path

from experiments.canonicalization.runtime import build_open_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path, help="Verified mesh render bundle")
    parser.add_argument("predictions", type=Path, help="Matching OA V2 inference directory")
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("gemini", "claude", "responses"), required=True)
    parser.add_argument("--model", help="Model ID; Gemini falls back to GEMINI_MODEL")
    parser.add_argument("--observations", type=Path, help="Verified original-photo bundle")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--max-calls", type=int, choices=(2, 3), default=3)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = build_open_controller(args.backend, args.blender, args.model).run(
            args.views,
            args.predictions,
            args.output,
            args.observations,
            args.resume,
            args.max_calls,
        )
        print(result / "canonicalization.json")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Open canonicalization did not complete: {error}\n")


if __name__ == "__main__":
    main()
