"""Rank INRIA Holidays queries, write an official-format result file, and time each stage.

For every query in the ground-truth JSONL this script decodes the image, embeds
it with the pinned CLIP service, and ranks all indexed images with FAISS. It
writes:

* ``<out-dir>/clip_result.dat``: full rankings in the format read by INRIA's
  ``eval_holidays/holidays_map.py`` (``query 0 name 1 name ...``), so the mAP
  can be checked with the official scorer;
* ``<out-dir>/latency.json``: p50/p95/mean per stage (decode, CLIP, FAISS, total).

Queries run one at a time after a warm-up, which matches how the API serves
searches. Run it from the repository root:

    python evaluation/holidays_benchmark.py \
        --candidate-dir /path/to/INRIA_Holidays/candidate_index \
        --images-dir /path/to/INRIA_Holidays/jpg \
        --ground-truth /path/to/INRIA_Holidays/ground_truth.jsonl \
        --out-dir /path/to/INRIA_Holidays
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from PIL import Image  # noqa: E402

from embeddings import get_embedding_service  # noqa: E402
from faiss_index import load_faiss_index, load_metadata, search_index  # noqa: E402


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--images-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    index = load_faiss_index(str(args.candidate_dir / "faiss_index.bin"))
    metadata = load_metadata(str(args.candidate_dir / "metadata.json"))
    queries = [
        json.loads(line)["query_path"]
        for line in args.ground_truth.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    service = get_embedding_service(device="cpu")
    service.embed_image(Image.new("RGB", (224, 224)))  # warm-up, not timed

    timings: dict[str, list[float]] = {
        "decode": [], "clip_embed": [], "faiss_search_full_ranking": [], "total": []
    }
    lines = []
    for query in queries:
        t0 = time.perf_counter()
        with Image.open(args.images_dir / query) as raw:
            image = raw.convert("RGB")
        t1 = time.perf_counter()
        vector = service.embed_image(image)
        t2 = time.perf_counter()
        results = search_index(index, vector, top_k=index.ntotal)
        t3 = time.perf_counter()
        for stage, seconds in (
            ("decode", t1 - t0),
            ("clip_embed", t2 - t1),
            ("faiss_search_full_ranking", t3 - t2),
            ("total", t3 - t0),
        ):
            timings[stage].append(seconds * 1000)
        ranked = [metadata[image_id] for image_id, _score in results]
        lines.append(
            query + " " + " ".join(f"{rank} {name}" for rank, name in enumerate(ranked))
        )

    (args.out_dir / "clip_result.dat").write_text("\n".join(lines) + "\n", encoding="utf-8")
    summary: dict = {
        stage: {
            "p50_ms": round(statistics.median(values), 1),
            "p95_ms": round(percentile(values, 95), 1),
            "mean_ms": round(statistics.mean(values), 1),
        }
        for stage, values in timings.items()
    }
    summary["queries"] = len(queries)
    summary["index_vectors"] = index.ntotal
    (args.out_dir / "latency.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
