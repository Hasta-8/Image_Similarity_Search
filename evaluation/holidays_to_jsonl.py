"""Convert INRIA Holidays ground truth into ImgSimiSearch evaluation JSONL.

INRIA Holidays (Jegou, Douze and Schmid, ECCV 2008) ships its ground truth as
``eval_holidays/perfect_result.dat``. Each line is a query image followed by
``rank filename`` pairs for its relevant images:

    147800.jpg 0 147802.jpg 1 147801.jpg

This script writes one JSON object per query in the format that
``backend/evaluate_retrieval.py`` expects:

    {"query_path": "147800.jpg", "relevant_paths": ["147802.jpg", "147801.jpg"]}

Paths are relative to the Holidays ``jpg/`` directory, which is the dataset
root used to build the candidate index.

Usage:
    python evaluation/holidays_to_jsonl.py \
        --perfect-result /path/to/eval_holidays/perfect_result.dat \
        --images-dir /path/to/INRIA_Holidays/jpg \
        --output /path/to/INRIA_Holidays/ground_truth.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

IMAGE_NAME = re.compile(r"^\d{6}\.jpg$")


def parse_perfect_result(path: Path) -> list[dict]:
    records = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        query, pairs = fields[0], fields[1:]
        if len(pairs) % 2:
            raise ValueError(f"line {line_number}: expected 'rank filename' pairs")
        relevant = pairs[1::2]
        for name in [query, *relevant]:
            if not IMAGE_NAME.match(name):
                raise ValueError(f"line {line_number}: unexpected image name {name!r}")
        if query in seen:
            raise ValueError(f"line {line_number}: duplicate query {query}")
        if not relevant:
            raise ValueError(f"line {line_number}: query {query} has no relevant images")
        if query in relevant or len(set(relevant)) != len(relevant):
            raise ValueError(f"line {line_number}: self or duplicate match for {query}")
        # Holidays groups share the first four digits of the file name.
        if any(name[:4] != query[:4] for name in relevant):
            raise ValueError(f"line {line_number}: relevant image outside group {query[:4]}")
        seen.add(query)
        records.append({"query_path": query, "relevant_paths": relevant})
    if not records:
        raise ValueError(f"{path} contains no queries")
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--perfect-result", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path,
                        help="Holidays jpg/ directory; every referenced file must exist")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    try:
        records = parse_perfect_result(args.perfect_result)
    except (OSError, ValueError) as exc:
        print(f"Conversion failed: {exc}", file=sys.stderr)
        return 2

    missing = [
        name
        for record in records
        for name in [record["query_path"], *record["relevant_paths"]]
        if not (args.images_dir / name).is_file()
    ]
    if missing:
        print(f"Conversion failed: {len(missing)} referenced images are missing, "
              f"e.g. {missing[:3]}", file=sys.stderr)
        return 2
    if args.output.exists():
        print(f"Conversion failed: output already exists: {args.output}", file=sys.stderr)
        return 2

    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")

    relevant_total = sum(len(record["relevant_paths"]) for record in records)
    print(f"Wrote {len(records)} queries with {relevant_total} relevant images to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
