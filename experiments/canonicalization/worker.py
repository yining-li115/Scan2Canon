"""Run one resumable mesh canonicalization job."""

import argparse
from pathlib import Path

from experiments.canonicalization.runtime import build_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("category")
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("gemini", "responses", "local"), required=True)
    parser.add_argument("--model", help="Model ID; Gemini falls back to GEMINI_MODEL")
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--max-calls", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        controller = build_controller(args.backend, args.blender, args.model, args.weights)
        result = controller.run(
            args.mesh, args.category, args.output, args.observations, args.resume, args.max_calls
        )
        print(result / "canonicalization.json")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Canonicalization did not complete: {error}\n")


if __name__ == "__main__":
    main()
