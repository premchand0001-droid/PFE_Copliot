# Retrieval relevance labels (Section 4.3)

`retrieval_labeling.csv` - 238 passages for 20 queries, scale 0 = not relevant, 1 = related / partial, 2 = directly answers.

**Provenance (state this in the report):** the first-pass suggestions were drafted by an AI assistant; the project owner then reviewed every row (including the 34 flagged for spot-check) and confirmed or corrected the label, so the file is reported as **human-verified**. Labels were pooled from BM25, dense, hybrid and hybrid+rerank results, so the usual pooling bias applies.

**How the notebook uses it:** Section 4.3a copies this file to `outputs/retrieval_labeling.csv` when no filled labels exist there yet (from the cloned repo, or from GitHub), but only if at least 80% of the labelled passages exist in the current index. If the corpus was rebuilt and the passages differ, the labels are not used and a fresh sheet is generated for labelling.
