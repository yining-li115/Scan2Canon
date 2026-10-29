"""Run one visual agent with bounded evidence actions and resumable capture requests."""

import argparse
from pathlib import Path

from experiments.canonicalization.agent_controller import submit_capture
from experiments.canonicalization.runtime import build_agent_controller


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("views", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("gemini", "claude", "responses"), required=True)
    parser.add_argument("--model", help="Pin one model; disables the Gemini fallback list")
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--additional-observations", type=Path)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--max-calls", choices=(2, 3, 4), type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--proposal-method", choices=("decoded", "distribution"), default="decoded")
    parser.add_argument(
        "--allow-shape-only",
        action="store_true",
        help="Explicit geometry-only baseline; no missing appearance is synthesized",
    )
    args = parser.parse_args()
    try:
        controller = build_agent_controller(
            args.backend,
            args.blender,
            args.model,
            allow_shape_only=args.allow_shape_only,
            proposal_method=args.proposal_method,
        )
        if args.additional_observations:
            if not args.resume:
                raise ValueError("Use --resume to submit evidence to a pending capture request")
            submit_capture(args.output, args.additional_observations)
        result = controller.run(
            args.views,
            args.predictions,
            args.output,
            args.observations,
            args.resume,
            args.max_calls,
        )
        print(result / "canonicalization.json")
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Canonicalization agent did not complete: {error}\n")


if __name__ == "__main__":
    main()
