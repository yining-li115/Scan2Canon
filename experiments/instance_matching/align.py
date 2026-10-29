"""Apply a stored shared-frame proposal to one of its source instances."""

import argparse
from pathlib import Path
from uuid import UUID

from experiments.artifacts import sha256
from experiments.canonicalization.runtime import build_shared_controller
from experiments.instance_matching.runner import verified_images
from scan2canon.runtime import build_runtime


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("object_id", type=UUID)
    parser.add_argument("proposal_id", type=UUID)
    parser.add_argument("views", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--backend", choices=("claude", "gemini", "responses"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--max-calls", type=int, choices=(2, 3, 4), default=4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        library = build_runtime(args.data_dir).library
        item = library.get_object(args.object_id)
        proposal = next((m for m in library.matches(item.id) if m.id == args.proposal_id), None)
        if proposal is None:
            raise ValueError("No matching proposal for this instance")
        evidence = Path(proposal.evidence.location)
        if not evidence.is_file() or sha256(evidence) != proposal.evidence.sha256:
            raise ValueError("Shared-frame evidence is missing or modified")
        for identity in (proposal.left_assessment_id, proposal.right_assessment_id):
            verified_images(library.get_assessment(identity))
        controller = build_shared_controller(
            args.backend, args.blender, proposal, item.source_mesh_sha256, args.model
        )
        print(
            controller.run(
                args.views,
                args.predictions,
                args.output,
                args.observations,
                args.resume,
                args.max_calls,
            )
        )
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        parser.exit(1, f"Shared-frame alignment stopped: {error}\n")


if __name__ == "__main__":
    main()
