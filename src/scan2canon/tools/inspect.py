import json

from scan2canon.domain import StageResult
from scan2canon.ports import ToolContext


class InspectCapture:
    """Report decoded inputs; this does not certify coverage or reconstruction quality."""

    version = "inspect/1"

    def execute(self, context: ToolContext) -> StageResult:
        capture = context.capture
        unique_count = len({frame.sha256 for frame in capture.frames})
        if unique_count < 3:
            raise ValueError("At least three distinct image files are required")
        report = {
            "capture_id": str(capture.id),
            "frame_count": len(capture.frames),
            "unique_file_count": unique_count,
            "checks": {"image_decoding": "passed", "view_coverage": "not_evaluated"},
        }
        key = context.artifacts.write(
            context.job.id, "capture-report.json", json.dumps(report).encode()
        )
        return StageResult(
            outputs={"capture_report": key},
            metrics={"frame_count": float(len(capture.frames))},
        )
