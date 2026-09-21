#!/usr/bin/env python3
"""Rewrite depth-video paths in a Cosmos batch JSONL spec."""

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-spec", type=Path, required=True)
    parser.add_argument("--depth-video-dir", type=Path, required=True)
    parser.add_argument("--output-spec", type=Path, required=True)
    parser.add_argument(
        "--check-files",
        action="store_true",
        help="Fail if any referenced depth-video basename is missing.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_spec = args.input_spec.expanduser().resolve()
    output_spec = args.output_spec.expanduser().resolve()
    depth_video_dir = args.depth_video_dir.expanduser().resolve()

    if input_spec == output_spec:
        raise ValueError("--output-spec must differ from --input-spec")

    records = []
    missing = []
    with input_spec.open("r", encoding="utf-8") as src:
        for line_number, line in enumerate(src, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            try:
                depth = record["control_overrides"]["depth"]
                source_path = Path(depth["input_control"])
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Missing control_overrides.depth.input_control on line {line_number}"
                ) from error

            target_path = depth_video_dir / source_path.name
            depth["input_control"] = str(target_path)
            if args.check_files and not target_path.is_file():
                missing.append(target_path)
            records.append(record)

    if missing:
        examples = "\n".join(f"  {path}" for path in missing[:10])
        raise FileNotFoundError(
            f"Missing {len(missing)} depth videos. First missing paths:\n{examples}"
        )

    output_spec.parent.mkdir(parents=True, exist_ok=True)
    with output_spec.open("w", encoding="utf-8") as dst:
        for record in records:
            dst.write(json.dumps(record, ensure_ascii=True) + "\n")

    print(f"Wrote {len(records)} records to {output_spec}")


if __name__ == "__main__":
    main()
