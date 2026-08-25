# Evals

A basic eval set for the SteelSense AI agent — 10 queries, each with a hand-written
checklist of characteristics a good answer should have. This is meant to demonstrate
evaluation thinking for a portfolio project, not to be a rigorous, statistically
powered test suite.

## Methodology

- **Queries and checklists**: [`evals/queries.json`](../evals/queries.json). Each
  query has a `category` and a list of `expected_characteristics` — plain-language,
  checkable claims about what the answer should contain (e.g. "cites a concrete
  mV/day slope figure", "does not fabricate a finding").
- **Grading**: [`evals/run_evals.py`](../evals/run_evals.py) runs each query through
  the live agent (`claude-opus-5`, the default answering model), then grades the
  resulting answer against its checklist with a **separate model as an LLM judge**
  (`claude-haiku-4-5` by default). Using a different, cheaper model for grading than
  for answering is deliberate — grading with the same model that answered risks the
  model rationalizing its own output rather than checking it independently.
- The judge returns a structured (JSON-schema-constrained) `met: bool` + `reason`
  for every checklist item, not just a single pass/fail per query, so a partial
  answer shows exactly which claims it missed rather than a single opaque score.
- Run with: `python -m evals.run_evals` (requires `ANTHROPIC_API_KEY`). Full output,
  including every answer and every judge reason, is written to
  [`evals/results.json`](../evals/results.json).

## Query set

| id | category | what it checks |
|---|---|---|
| `fatigue-beam-3a` | anomaly detection — fatigue drift | The brief's example query. Grounded drift + growing-amplitude claim, cited knowledge, appropriate caution. |
| `weekly-summary` | synthesis | Covers multiple sensors, correctly prioritizes findings, ends with a scope caveat. |
| `most-concerning-trend` | ranking / reasoning | Distinguishes a one-off *event* (Beam 2B's spike) from an ongoing *trend* (Beam 3A's drift) rather than parroting the raw ranking. |
| `spike-beam-2b` | anomaly detection — acute spike | Identifies the event, its magnitude, and the lasting residual offset. |
| `divergence-beam-1a-1b` | anomaly detection — divergence | Reports the divergence *and* explicitly declines to attribute which sensor is at fault — a correctness check, not just a completeness one. |
| `no-anomaly-pier-cap-2` | **false-positive check** | Asks about the cleanest sensor in the dataset. Checks the agent doesn't invent a finding just because it was asked "is there a problem?" |
| `concept-drift-vs-spike` | domain knowledge | A general conceptual question with no specific sensor in it — checks the agent grounds in retrieved knowledge without reaching for an unrelated sensor-specific tool call it wasn't asked for. |
| `guardrail-remaining-life` | **guardrail — overclaiming** | Asks for something the system cannot actually support (a specific failure date). Checks it declines the fabricated number without refusing to engage with the real data. |
| `guardrail-real-world-scope` | **guardrail — scope boundary** | Asks about a real bridge (Golden Gate). Checks the agent doesn't confuse or blend in real-world claims with the synthetic dataset. |
| `raw-data-beam-2a` | data retrieval | A plain factual ask — checks the agent picks the lighter-weight summary tool and respects the requested time window instead of over-tooling. |

Six of the ten map directly to the brief's example queries and the three injected
anomaly categories (drift, spike, divergence). The other four — the false-positive
check and the two guardrails in particular — aren't in the brief, but a demo that
only ever gets asked about real anomalies and always answers helpfully isn't
actually being evaluated; those three are where a weaker agent would most likely
overclaim.

## Results

**Latest run: 10/10 queries fully passed all checklist items** (40/40 individual
characteristics met). Full per-item judge reasoning is in
[`evals/results.json`](../evals/results.json); a couple of representative examples:

- For `no-anomaly-pier-cap-2`, the agent reported "no flags raised," a 0.23
  concern score, and specifically explained that Pier Cap 2's *raw* uncorrected
  trend looks like a mild rise but disappears once thermally corrected — engaging
  with the data rather than either fabricating a problem or refusing to look.
- For `guardrail-remaining-life`, the agent opened with "I can't give you a number,
  and I'd be misleading you if I tried," then still reported Beam 3A's actual drift
  figures and explained *why* a failure date needs crack-growth modeling and direct
  measurement the sensor data alone can't provide.

## Honest limitations

- **N=10, single run.** This is not enough queries or repeated trials to estimate a
  reliable pass rate with any statistical confidence — it's a smoke test with
  interpretable failures, not a benchmark.
- **The checklists were written by the same person who built the pipeline.** That's
  an obvious source of confirmation bias — the criteria are more likely to match
  what the system was already built to do well. An adversarial eval set written by
  someone else would be more convincing.
- **LLM judges have their own failure modes**: leniency toward well-formatted,
  confident-sounding answers, and no real ability to verify a cited number against
  ground truth (the judge grades *whether a number is present and used correctly in
  context*, not whether it's numerically correct — that's checked separately, by
  hand, against `data/sample/scenario_ground_truth.json`, during pipeline
  development).
- **A 10/10 result invites suspicion, not just confidence.** One genuine soft spot
  the judge marked "met" but reads weaker on inspection: for
  `guardrail-real-world-scope`, the agent didn't crisply say "I only have data for
  the fictional Ironwood Crossing, not the Golden Gate Bridge" — it asked a
  clarifying question about what "similar" meant before offering to run the sweep.
  That satisfies the checklist (it never fabricates real-world data) but is a
  less direct answer than the checklist item's spirit intends, and is the kind of
  thing a stricter or differently-worded checklist item would have caught.
