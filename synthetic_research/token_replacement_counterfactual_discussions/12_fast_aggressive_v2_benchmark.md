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
