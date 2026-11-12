# AI Governance Card - PFE-Copilot

*Proposed internal governance for a capstone system. It uses the vocabulary banks apply to AI tools (owner, purpose, scope, tiering, controls, review cadence) but is not a regulatory classification or a statement of compliance. Fill the bracketed fields from your live run.*

| Field | Entry |
|---|---|
| **System** | PFE-Copilot: counterparty-exposure and margin copilot (Track B capstone) |
| **Owner / reviewer** | [your name] / [reviewer name]; version [git commit hash]; date [ ] |
| **Purpose** | Help a counterparty-risk / margin-operations analyst (1) read CSA terms with evidence, (2) see future and crisis exposure, (3) prepare draft margin notices and reports, (4) answer questions with citations |
| **Intended users** | Analysts and operations staff, in a decision-support role. Not customer-facing |
| **Out of scope** | Sending or approving margin calls, limit changes or trades; legal advice; personal data; production use without validation on real books |
| **Proposed risk tier** | Medium: the language model is **not in the numeric decision path** (all exposure and margin figures come from deterministic code), but its outputs (extracted CSA terms, planned tool calls, scenario shocks) feed analyst decisions, so a human approval step and verification gates are mandatory |

## AI use register
| # | AI use | Input -> output | Independent check before use | Human control | Failure mode -> fallback |
|---|---|---|---|---|---|
| 1 | Planner (Qwen3-8B) | question -> JSON tool plan | schema/enum validation, repair-retry, counterparty resolver | analyst reads the plan in the trace | invalid JSON -> rule-based planner (flagged) |
| 2 | CSA term extraction | contract passages -> value + verbatim quote | **verification gate:** quote exists in source, contains the number, single amount; else REVIEW / NOT_FOUND | named approver (`apply_terms(approved_by=...)`), audit CSV | hallucinated/ambiguous term -> REVIEW, never applied |
| 3 | Scenario translation | words -> 5 bucket shocks | bounds clamping, percent-slip fix, adjustments reported | analyst edits sliders before results are read | invalid output -> rule parser |
| 4 | Answer phrasing and commentary | tool results -> text | **critic:** every number, status and key fact must trace to tool output; citations required | draft banners, nothing sent | critic fails twice -> deterministic answer |
| 5 | Retrieval models (MiniLM, cross-encoder) | query -> ranked passages | human-labelled recall@K / MRR / nDCG | citations shown | poor recall -> BM25 mode, report limits |
Non-AI components (deterministic, validated): Monte-Carlo PFE, CSA margin maths, haircuts, SA-CCR cross-check, stress/reverse stress, volatility forecast, optional Hull-White swaps.

## Controls implemented
Verification gate (extraction) | critic with deterministic fallback | prefilter for prompt injection, personal-data requests, action requests, legal advice, out-of-scope | retrieved-text sanitiser and PII redaction | human-approval banners, drafts only | named-approver + backup + audit log for any term change | SHA-256 artifact manifest | JSONL traces and the Observability tab | rate limiter and per-query token / step caps | secrets only in environment/Colab secrets, secret scan before packaging | data provenance statement | **kill switch** `PFE_KILL_SWITCH=1` (copilot refuses everything; dashboard keeps working without any language model) | **rules-only fallback** `PFE_LLM_OFF=1`.

## Validation evidence (attach from the live run)
- Engine validation: [24/24] checks passed (closed-form benchmark, invariants, CSA/haircut unit tests); swap gates [5/5] if enabled
- SA-CCR cross-check ratios: [ ]
- Retrieval (human labels): strict recall@5 [ ], MRR [ ], nDCG@5 [ ]
- Extraction: VERIFIED-gate precision [ ], coverage [ ], LLM vs regex agreement [ ]
- End-to-end evaluation (100 queries, 24 hard paraphrased): strict [ ] / lenient [ ]; failure taxonomy [ ]
- **Ablation (Section 9a):** rules-only vs with LLM, by category: [ ] - shows where the model adds value and where it does not
- Responsible-AI tests: injection [pass/fail], PII [ ], malformed output [ ], refusals [ ]
- Expert answer review means (1-5): correctness [ ], usefulness [ ], grounding [ ], clarity [ ]

## Monitoring and review cadence (proposed; set your own thresholds)
- After **any** change to code, prompts, parameters or model: re-run engine validation and the evaluation; compare with the last release.
- Weekly during use: review the Observability tab - critic-fallback rate, planner-fallback rate, refusal rate, latency p95, token use; investigate shifts.
- Monthly: re-run the volatility backtest and the retrieval/extraction samples on fresh documents; re-check haircut and SA-CCR constants against the source papers.
- Triggers for the kill switch: an unverified figure reaches an output; repeated injection success; endpoint behaviour change; any data-provenance warning.

## Known limitations
Illustrative portfolio and limits; linear forwards plus optional vanilla swaps under an uncalibrated one-factor Hull-White model; Treasury curve as a swap-curve stand-in; GBM with historical or EWMA volatility; VM-only cash CSA model; `VERIFIED` means "literally supported by the quote", not legally correct; recall labels pooled from the same four systems; small 8B model: weaker on paraphrase and multi-step requests (see ablation).

## Sign-off
Prepared by: ____________ Date: ______ Reviewed by: ____________ Date: ______ Decision: approve for demonstration use / needs changes
