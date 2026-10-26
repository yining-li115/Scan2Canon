import argparse
import logging
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel

from scan2canon.runtime import build_runtime
from scan2canon.worker import run_once


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(prog="scan2canon")
    command.add_argument("--data-dir", type=Path, default=None)
    commands = command.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("import", help="Import 3–200 JPEG/PNG/HEIC/HEIF images")
    capture.add_argument("directory", type=Path)
    submit = commands.add_parser("submit", help="Queue an imported capture")
    submit.add_argument("capture_id", type=UUID)
    submit.add_argument("--max-attempts", type=int, default=12)
    for name in ("status", "events", "retry"):
        operation = commands.add_parser(name)
        operation.add_argument("job_id", type=UUID)
    worker = commands.add_parser("worker", help="Process queued jobs, then exit")
    worker.add_argument("--once", action="store_true", help="Process at most one job")
    return command


def emit(value: BaseModel) -> None:
    print(value.model_dump_json(indent=2))


def main() -> None:
    command = parser()
    args = command.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    try:
        runtime = build_runtime(args.data_dir)
        match args.command:
            case "import":
                emit(runtime.service.import_capture(args.directory))
            case "submit":
                emit(runtime.service.submit(args.capture_id, args.max_attempts))
            case "status":
                emit(runtime.jobs.get(args.job_id))
            case "events":
                for event in runtime.jobs.events(args.job_id):
                    emit(event)
            case "retry":
                emit(runtime.jobs.retry(args.job_id))
            case "worker":
                while (job := run_once(runtime)) is not None:
                    emit(job)
                    if args.once:
                        break
    except (OSError, ValueError, KeyError, RuntimeError) as error:
        command.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
