# Retrieval-quality evaluation

Retrieval quality is measured on the public **INRIA Holidays** benchmark
(results in the main README). This directory has the tools to reproduce that
result and to evaluate any other labelled dataset.

| File | Purpose |
|---|---|
| `ground_truth.schema.json` | JSONL format for labelled query/relevance pairs |
| `holidays_to_jsonl.py` | Converts INRIA Holidays ground truth into that format |
| `holidays_benchmark.py` | Ranks every query, writes INRIA-format results, and measures per-stage latency |

## Reproducing the INRIA Holidays benchmark

Keep the dataset outside the repository. INRIA holds the copyright to the
images. If you report results, cite Jégou, Douze and Schmid, "Hamming Embedding
and Weak Geometry Consistency for Large Scale Image Search", ECCV 2008.

1. Download from the [official page](https://lear.inrialpes.fr/~jegou/data.php.html):
   `jpg1.tar.gz` and `jpg2.tar.gz` (images, 2.8 GB in total) and
   `eval_holidays.tgz` (ground truth and official scorer). Extract them into
   one directory, `INRIA_Holidays/`, so that you have `INRIA_Holidays/jpg/`
   (1,491 images) and `INRIA_Holidays/eval_holidays/`.

2. Convert the ground truth (500 queries, 991 relevant images). Run from the
   repository root:

   ```bash
   H=/path/to/INRIA_Holidays
   python evaluation/holidays_to_jsonl.py \
     --perfect-result $H/eval_holidays/perfect_result.dat \
     --images-dir $H/jpg --output $H/ground_truth.jsonl
   ```

3. Build an evaluation-only index. Never promote it.

   ```bash
   cd backend
   python rebuild_index.py --dataset $H/jpg --output-dir $H/candidate_index --device cpu
   ```

4. Compute the metrics. `K = 1490` ranks the whole gallery, so its AP is the
   standard full-ranking mAP.

   ```bash
   python evaluate_retrieval.py --candidate-dir $H/candidate_index \
     --dataset $H/jpg --ground-truth $H/ground_truth.jsonl --ks 1,5,10,1490
   ```

5. Measure latency and write rankings for the official scorer:

   ```bash
   cd ..
   python evaluation/holidays_benchmark.py --candidate-dir $H/candidate_index \
     --images-dir $H/jpg --ground-truth $H/ground_truth.jsonl --out-dir $H
   ```

   Score `$H/clip_result.dat` with `eval_holidays/holidays_map.py`. That script
   is Python 2. Under Python 3 it needs syntax-only changes (`print` calls and
   `list(zip(...))`). A correct port reproduces INRIA's reference values:
   `perfect_result.dat` gives 1.00000 and `he_wgc_result_200806.dat` gives
   0.74626.

The official scorer computes AP with trapezoidal interpolation, while
`evaluate_retrieval.py` uses standard non-interpolated AP. On the same
rankings these give 0.8437 and 0.8631. Quote the official value when comparing
with published Holidays results.

## Evaluating your own dataset

`ground_truth.schema.json` defines one JSON object per JSONL line:

- `query_path`: an existing dataset-relative POSIX path;
- `relevant_paths`: one or more distinct existing indexed paths selected by a
  human according to the project's documented visual/semantic relevance rule;
- the query image itself must not appear in `relevant_paths`.

Collect labels across multiple queries and categories, including clear relevant
and hard-negative cases. Define the relevance rule before annotation, use at
least two annotators for ambiguous pairs where practical, and resolve
disagreements before calculating metrics. Keep the label file outside generated
candidate/release directories.

After labels exist, run from `backend/`:

```bash
python evaluate_retrieval.py \
  --candidate-dir candidate_index \
  --dataset ../dataset \
  --ground-truth ../evaluation/ground_truth.jsonl \
  --ks 1,5,10
```

The harness checks the candidate, dataset fingerprint, model revision, paths,
and stable IDs. It excludes exact query self-matches from rankings. It reports
macro Precision@K, Recall@K, AP@K (mean average precision over queries), and
MRR@K. Precision divides by K; AP@K divides by `min(number_of_relevant_items,
K)`. These definitions are implemented in `backend/retrieval_metrics.py`.

The metric unit tests use synthetic rankings solely to verify formulas; they are
not relevance labels or a retrieval benchmark.
