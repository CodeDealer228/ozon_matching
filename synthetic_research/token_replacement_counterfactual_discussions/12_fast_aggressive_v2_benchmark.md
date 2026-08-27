# Fast-aggressive v2: benchmark before the 12k run

## Goal

Measure real wall time and accepted-operation yield on exactly 200 new, balanced
human-train positive source pairs. The benchmark is only a throughput and quality
measurement. If the resulting ETA is acceptable, the next run targets the full
12,000 synthetic pairs; there is no intermediate training experiment on 1,000–2,000
pairs.

## Why v1 was stopped

The first aggressive design made three sequential generator calls, optional web
research calls, and a final validator call for every source. At the observed rate,
the full target required multiple days. The validator prompt also initially
misinterpreted an identity-changing replacement as an error; that issue was fixed in
commit `2f9fffc`, but the per-source architecture remained too slow.

## Fast v2 architecture

1. Build a replacement bank locally from identity-bearing tokens observed in the
   canonical human-train positive source pack.
2. Rank candidate values by category, semantic format, occurrence in real train
   cards, and lexical product-context similarity.
3. Send eight same-category source pairs in one structured generator request.
4. Send all proposed operations from that batch in one independent structured
   validator request.
5. Materialize each accepted operation as one two-sided positive and one one-sided
   negative pair.

This changes the request count from roughly four or five calls per source to at most
two calls per eight sources. Web search is not performed per source. Real train
observations supply replacement candidates; the model may always skip.

## Benchmark configuration

- source pairs: 200, balanced as 10 per category;
- batch size: 8;
- workers: 12;
- generator model pool: the four highest-ranked currently available general models,
  with exact quota/model-unavailable rotation;
- validator: a different model from the same high-ranked pool;
- source tokens considered per pair: 4;
- real-train replacement candidates per source token: 6;
- maximum accepted operations per source: 2;
- reasoning: disabled;
- durable submitted/terminal journal: enabled;
- retry of submitted, terminal, or uncertain sources: disabled.

## Provider contract

Alibaba Singapore receives only category, names, compact attributes, two retrieved
human-train RAG examples and allowed candidate operations. Local item IDs, labels,
validation rows and secrets are never included in the payload.

## Required benchmark report

The manifest records:

- wall-clock API time;
- attempted, completed and terminal sources;
- accepted operations and resulting positive/negative pair counts;
- sources/hour and operations/source;
- token and prompt-cache usage;
- model provenance and exact errors;
- projected sources and hours for 12,000 pairs;
- whether the remaining strict source pool is large enough for the target.

The ETA is accepted only after a representative output audit. A fast run that emits
implausible replacements is not considered a successful benchmark.

## Benchmark v2.0 finding and v2.1 correction

The first 200-source batched run proved the speed target but failed the quality gate.
It finished its API work in 36.85 seconds and produced 154 locally applicable
operations, but the initial bank matched only broad semantic formats. Manual review
found cross-product substitutions such as a fishing brand proposed for a radio.
Those rows are retained only as a diagnostic artifact and are not approved for the
training dataset.

Version 2.1 keeps the same two batched model calls but tightens candidate construction:

- infer the attribute role (brand, article/model, count, volume, weight, dimensions,
  power, resolution, format) from both source cards;
- require the same role for the replacement value;
- prefer values extracted from the retrieved hard-negative human-train pair;
- require contextual title overlap for global-bank fallbacks;
- attach the real evidence product name to every candidate operation;
- reject partial multi-word brands, country/manufacturer noise and non-applicable
  numeric fragments locally.

A second disjoint 200-source run is required before using an ETA for the full 12k
generation.

## Disjoint v2.1 benchmark result

The role-aware run used 200 new sources, 10 per category, and completed all provider
work in 47.28 seconds. Local bank preparation took another 52.31 seconds and is paid
once per process. Because each category had 10 rows, the benchmark used 40 batches
(8+2 per category); long full runs mostly use full eight-row batches.

Results before the final conservative dependency filter:

- attempted sources: 200;
- completed sources: 190;
- terminal sources: 10, both Electronics validator batches affected by a safe-to-fix
  `status=skip` plus stale operation IDs formatting inconsistency;
- accepted operations: 163;
- materialized pairs: 326 (163 positive + 163 negative);
- locally non-applicable operations: 0;
- observed throughput: 15,229 attempted sources/hour;
- raw operations/source: 0.815.

The representative per-category audit was materially better than v2.0: examples
included `domino -> contex` for condoms, `h1n -> q8n-4k` for Zoom recorders,
`100штук -> 200штук` for ball-pit balls and `саи160к -> саи250а` for Resanta welding
machines. It also exposed a smaller remaining failure mode: changing a brand while
leaving a brand-specific article, or changing an article to one belonging to another
brand.

The full-run code therefore rejects those identity-dependency conflicts before the
model sees them. Applying the same rule conservatively to the completed benchmark
retains 127 operations, or 0.635 operations per attempted source. At that lower-bound
yield, 6,000 operations (12,000 balanced pairs) require about 9,449 source attempts.
After excluding diagnostic and benchmark sources, the strict source pack still has
about 11,460 unused sources, so the total target remains feasible.

### ETA

- mathematical API estimate at the conservative yield: about 37 minutes;
- practical estimate including bank construction, uneven category tails, checkpoint
  writes and ordinary model rotation: **45–75 minutes**;
- risk envelope if several provider quotas expire during the run: up to roughly two
  hours.

This estimate is based on actual end-to-end provider latency, not a theoretical call
count. The 12k run must stop per category at exactly 300 accepted operations and must
not include the v2.0 diagnostic rows.
