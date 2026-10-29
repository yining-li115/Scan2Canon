"""Compare two saved instance assessments using a bounded cloud visual policy."""

import argparse
from pathlib import Path
from uuid import UUID

from experiments.canonicalization.runtime import build_policy
from experiments.instance_matching.runner import run
from scan2canon.runtime import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=UUID)
    parser.add_argument("right", type=UUID)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("claude", "gemini", "responses"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--max-calls", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            build_runtime(args.data_dir).library,
            build_policy(args.backend, args.model),
            args.left,
            args.right,
            args.output,
            resume=args.resume,
            max_calls=args.max_calls,
        )
        print(result)
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        parser.exit(1, f"Instance matching stopped: {error}\n")


if __name__ == "__main__":
    main()
