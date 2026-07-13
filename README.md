# YouTube Bot Detection Pipeline

A two-stage statistical + ML pipeline that scores YouTube channels for
bot/fraudulent-audience risk, sourced live from the Brandconnect API.

## Overview

The pipeline works in two stages:

1. **Stage 1 — Statistical scoring** (`scorer.py`): compares each channel
   against a peer group (same niche, similar subscriber count) using robust
   z-scores across engagement, comment sparsity, audience entropy, and
   growth-efficiency features. Produces a `suspicion_score`.
2. **Stage 2 — Refinement** (`refiner.py`): trains a PCA + RBF-kernel SVM on
   the extremes of the Stage 1 distribution (top 5% / bottom 30%) to learn a
   non-linear decision boundary, then re-scores every channel against it.
   Produces a `refined_score` and a `risk_band` (`Minimal` / `Low` /
   `Elevated` / `High`).

Data no longer comes from a CSV export. It's pulled directly from
Brandconnect via the project's existing extractor and persisted to MongoDB,
then read back and mapped into the pipeline's internal `ChannelRecord`
format.

## Project structure

```
project_root/
├── main.py                     # entry point / usage example
├── models/
│   └── bot_detection/
│       ├── __init__.py
│       ├── channel_record.py   # ChannelRecord dataclass
│       ├── record_builder.py   # Brandconnect/Mongo -> ChannelRecord
│       ├── scorer.py           # Stage 1: statistical peer-relative scoring
│       ├── refiner.py          # Stage 2: PCA + SVM refinement
│       └── pipeline.py         # orchestrates the two stages
├── logs/
├── youtube/
│   └── extract/
│       └── brandconnect/
│           ├── extractors/
│           │   └── channel_extractor.py   # pulls from Brandconnect API
│           └── crud/
│               ├── channelpublicdata_crud.py
│               └── channelinsights_crud.py
└── README.md
```

## Data flow

1. **Extraction** — `ChannelExtractor.extract_data(channel_ids)` calls the
   Brandconnect API and persists results to MongoDB via
   `ChannelPublicDataCRUD` and `ChannelInsightCRUD`.
2. **Read-back** — `ChannelRecordBuilder` reads the persisted documents
   straight from their Mongo collections (`get_collection().find_one(...)`),
   taking the most recently `collected_at` insight document per channel.
3. **Mapping** — the builder converts the raw Mongo documents into a
   `ChannelRecord`:
   - Core metrics (`views`, `likes`, `comments`, `engagement_rate`) come
     from `insight_doc["OVERALL"]`.
   - `subscriber_count` / `video_count` / `topic_categories` prefer the
     public-data document, falling back to `OVERALL`'s duplicate fields.
   - `age_dist` is remapped from the API's camelCase keys (e.g.
     `age18To24`) to the pipeline's internal snake_case buckets (e.g.
     `age18_24`).
   - `country_dist` uses each entry's top-level `region_code` /
     `percentage`; nested sub-region detail is dropped (finer grain than
     the entropy calculation needs).
   - `interests` is left empty — it's not used anywhere in scoring.
4. **Scoring** — `StatisticalScorer.score_population` /
   `score_single` (Stage 1), then `WeightRefiner.refine` /
   `refine_single` (Stage 2).

## Usage

### Bootstrap / retrain the baseline

Run this once to establish a baseline population, and periodically after
that to keep it current as your channel pool grows:

```python
from models.bot_detection import BotDetectorPipeline

pipeline = BotDetectorPipeline()
results = pipeline.run_population(channel_ids, refresh=True)
```

`refresh=True` triggers live extraction from Brandconnect before scoring;
pass `refresh=False` to re-score using data already in Mongo.

### Score a single new influencer

Once a baseline has been trained, score new channels one at a time without
retraining:

```python
result = pipeline.score_new_influencer("UCFlYkvx37oSWzC1Pqy4CHDA")
print(result)
# {
#   "channel_id": "...",
#   "suspicion_score": 0.42,
#   "features": {...},
#   "anomalies": {...},
#   "refined_score": 0.81,
#   "risk_band": "Elevated"
# }
```

`score_new_influencer` also defaults to `refresh=True` (live pull), and
returns `None` if the channel's public/insight data can't be found or
built.