# Retrieval-quality evaluation data

No human-labeled query/relevance pairs exist yet, so no retrieval-quality
result is claimed. Create `ground_truth.jsonl` in this directory to enable
evaluation.

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
