"""Download the demo dataset: bird and animal photos from Open Images V7.

Usage:
    python scripts/fetch_demo_dataset.py --output dataset [--per-class 10]

Picks photos from the Open Images *validation* split that human annotators
labelled with one of the classes below, skipping any photo that is also
labelled with a person or that needs rotating. Selection is deterministic
(sorted image IDs), so re-running gives the same set.

Open Images lists these photos as CC BY 2.0; each photo's author, licence and
source page are written to ``<output>/CREDITS.csv``. The indexer ignores that
file because it only reads .jpg/.jpeg/.png.

Standard library only.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

META = {
    "classes.csv": "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv",
    "labels.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-human-imagelabels-boxable.csv",
    "images.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}
IMAGE_URL = "https://open-images-dataset.s3.amazonaws.com/validation/{}.jpg"

CLASSES = [
    "Owl", "Parrot", "Duck", "Penguin", "Dog",
    "Cat", "Horse", "Elephant", "Giraffe", "Zebra",
]
PEOPLE = ["Person", "Man", "Woman", "Boy", "Girl", "Human face", "Human body"]
UA = {"User-Agent": "ImgSimiSearch-demo-dataset/1.0"}


def fetch(url: str, dest: Path) -> None:
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return
        except Exception as exc:  # network hiccup: back off and retry
            if attempt == 4:
                raise
            print(f"retry {url}: {exc}", file=sys.stderr)
            time.sleep(5 * (attempt + 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output", type=Path, required=True, help="dataset folder to fill")
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--cache", type=Path, default=Path(".openimages-cache"),
                    help="where the metadata CSVs are kept")
    args = ap.parse_args()

    args.cache.mkdir(parents=True, exist_ok=True)
    for name, url in META.items():
        if not (args.cache / name).exists():
            print("metadata:", name)
            fetch(url, args.cache / name)

    with open(args.cache / "classes.csv", newline="") as f:
        mid_of = {name: mid for mid, name in csv.reader(f)}
    missing = [c for c in CLASSES + PEOPLE if c not in mid_of]
    if missing:
        raise SystemExit(f"classes not in Open Images: {missing}")
    wanted = {mid_of[c]: c for c in CLASSES}
    people = {mid_of[c] for c in PEOPLE}

    has_person: set[str] = set()
    by_class: dict[str, set[str]] = defaultdict(set)
    with open(args.cache / "labels.csv", newline="") as f:
        for row in csv.DictReader(f):
            if row["Confidence"] != "1":
                continue
            if row["LabelName"] in people:
                has_person.add(row["ImageID"])
            elif row["LabelName"] in wanted:
                by_class[wanted[row["LabelName"]]].add(row["ImageID"])

    info: dict[str, dict[str, str]] = {}
    with open(args.cache / "images.csv", newline="") as f:
        for row in csv.DictReader(f):
            info[row["ImageID"]] = row

    chosen: dict[str, str] = {}
    for cls in CLASSES:
        n = 0
        for image_id in sorted(by_class[cls]):
            row = info.get(image_id)
            if (image_id in has_person or image_id in chosen or row is None
                    or row.get("Rotation", "") not in ("", "0", "0.0")):
                continue
            chosen[image_id] = cls
            n += 1
            if n == args.per_class:
                break
        print(f"{cls:10s} {n}")

    args.output.mkdir(parents=True, exist_ok=True)
    for k, (image_id, cls) in enumerate(sorted(chosen.items()), 1):
        dest = args.output / f"{image_id}.jpg"
        if not dest.exists():
            fetch(IMAGE_URL.format(image_id), dest)
        if k % 100 == 0:
            print(f"downloaded {k}/{len(chosen)}")

    with open(args.output / "CREDITS.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "class", "author", "license", "source", "author_profile"])
        for image_id, cls in sorted(chosen.items()):
            row = info[image_id]
            w.writerow([f"{image_id}.jpg", cls, row["Author"], row["License"],
                        row["OriginalLandingURL"], row["AuthorProfileURL"]])
    print(f"done: {len(chosen)} photos in {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
