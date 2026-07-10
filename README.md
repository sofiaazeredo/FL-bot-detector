# YouTube Suspicious Engagement Scoring

A lightweight Python implementation for estimating the likelihood that a YouTube channel exhibits artificially inflated engagement by comparing its metrics against similar creators.

The model is not a bot detector. Instead, it computes an anomaly score based on how unusual a channel's engagement patterns are relative to its peers.

---

## Overview

Each channel is compared against a cluster of similar channels defined by:

* **Content niche**
* **Order of magnitude of subscriber count** (log10 buckets)

For every feature, the script computes a **robust z-score** using the peer group's **median** and **Median Absolute Deviation (MAD)**, making it resistant to outliers and heavy-tailed engagement distributions.

Individual anomaly signals are combined into a normalized **Suspicion Score** between **0.0** and **1.0**.

---

## Features Used

### 1. Comment Sparsity

Measures whether a channel receives significantly fewer comments per view than similar creators.

* Lower than peers → more suspicious
* Higher than peers → not penalized

---

### 2. Engagement Rate Anomaly

Measures likes per view.

Both unusually low and unusually high engagement can indicate abnormal behavior.

---

### 3. Cross-Video Engagement Dispersion

Computes the coefficient of variation of engagement rates across videos.

Very consistent or highly inconsistent engagement compared to peers may indicate manipulation.

Requires at least **5 videos**.

---

### 4. Subscribers per Video

Calculates:

Subscribers ÷ Number of Uploaded Videos

Channels with unusually high subscriber counts relative to their upload history receive a small penalty.

This feature has intentionally low weight because it is affected by channel age, upload strategy and content type.

---

### 5. Audience Country Entropy

Computes the normalized Shannon entropy of the audience country distribution.

Very concentrated audiences may indicate purchased or low-quality traffic.

---

### 6. Audience Age Entropy

Computes the normalized Shannon entropy of the audience age distribution.

Extremely concentrated age distributions may be considered suspicious depending on the peer baseline.

---

## Clustering

Peer groups are built using:

```
(niche, log10(subscriber_count))
```

Example:

```
("gaming", 5)
```

represents gaming channels with approximately 100k–999k subscribers.

If too few peers exist, the algorithm falls back to:

1. Same subscriber bucket across all niches
2. Entire dataset

This prevents unstable statistics in very small clusters.

---

## Robust Statistics

Instead of using mean and standard deviation, the model uses:

* Median
* Median Absolute Deviation (MAD)

Advantages:

* Resistant to viral outliers
* Resistant to purchased engagement
* More stable on skewed social-media metrics

Robust z-scores are capped to avoid instability when peer distributions are extremely tight.

---

## Suspicion Score

Each feature contributes a normalized score between **0** and **1**.

Only features successfully computed for the target channel contribute to the final score.

Weights are automatically renormalized when features are missing.

The final score is also bounded to **[0,1]**.

---

## Risk Bands

| Score     | Risk     |
| --------- | -------- |
| 0.00–0.19 | Minimal  |
| 0.20–0.44 | Low      |
| 0.45–0.74 | Elevated |
| 0.75–1.00 | High     |

These thresholds are heuristic and intended as starting points.

---

## Output

The scorer returns a `ScoreResult` containing:

* Channel ID
* Suspicion score
* Risk band
* Individual component scores
* Caveats (missing data, fallback clusters, etc.)
* Peer cluster used
* Peer cluster size

---

## Intended Use

This project is designed as an anomaly detection tool, not as definitive evidence of artificial engagement.

A high suspicion score indicates that a channel differs significantly from similar creators according to the available metrics and should be considered for further manual review.
