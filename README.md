# Multi-Stage Influencer Bot Detector Pipeline

An unsupervised anomaly detection framework designed to identify bot-farmed and engagement-inflated channels. The system operates in two distinct mathematical stages: peer-relative continuous statistical profiling followed by a non-linear machine learning refinement layer.

## Architecture

The framework relies on a decoupled, two-stage evaluation pipeline to isolate anomalies cleanly without relying on brittle, manually defined rules:

### Stage 1: Continuous Statistical Profiling

* **Dynamic Peer Matching:** Instead of splitting channels into rigid subscriber buckets (which creates edge artifacts), channels are matched dynamically with their closest 30 size-peers inside their target niche using a log-distance metric.
* **Robust Z-Scores:** Calculates peer-relative deviations using Median and Median Absolute Deviation (MAD) to stay resilient against heavy-tailed organic data.
* **Sigmoid Anomaly Compression:** Maps infinite Z-scores into a clean $[0, 1]$ interval using a non-linear sigmoid transform. This preserves the distinction between high anomalies ($Z = 6$) and extreme bot attacks ($Z = 50$) without letting massive outliers break the boundaries.
* **Global Baseline Corrections:** Employs Laplace smoothing on interaction ratios to cleanly distinguish dead accounts from bot footprints, alongside globally normalized Shannon entropy metrics for country/age spreads.

### Stage 2: Non-Linear Machine Learning Refinement

* **Global Variance Mapping:** Standardizes data and fits Principal Component Analysis (PCA) on the *entire* population to learn the true coordinate space of normal traffic variance.
* **Support Vector Machine (SVM) Decision Boundary:** Extracts pseudo-labels from the extreme tails of Stage 1 scores ($5\%$ highest vs $30\%$ lowest) and trains a non-linear RBF Kernel SVM. This breaks linear circularity and uncovers hidden, complex multi-feature interactions.

---

## Project Structure

```text
├── main.py             # Pipeline orchestrator & file I/O wrapper
├── models.py           # Dataclasses (ChannelRecord) and CSV loaders
├── scorer.py           # Feature Extraction & Stage 1 Statistical Scorer
├── refiner.py          # PCA + RBF SVM Stage 2 Classifier
└── README.md           # Documentation

```

---

## Getting Started

### Prerequisites

Ensure you have the required dependencies installed:

```bash
pip install numpy scikit-learn

```

### 1. Batch Processing Execution (Initial Run)

To score an entire population dataset at once, run the pipeline directly. This will load the CSV, compute baseline metrics, run both scoring stages, and output a refined CSV containing suspicion scores and risk bands (`Minimal`, `Low`, `Elevated`, `High`).

```bash
python main.py

```

### 2. Live / Real-Time Single Influencer Scoring

The pipeline allows you to evaluate a brand-new influencer on the fly **without needing to re-run or re-train the entire database**.

To score a single record live, you leverage the saved memory states (`self.scaler`, `self.pca`, `self.clf`) and compare the target directly against the pre-loaded baseline data pool:

```python
from models import CSVLoader
from scorer import StatisticalScorer
from refiner import WeightRefiner

# 1. Load your core baseline population once into memory
loader = CSVLoader("youtube_sample_100k.csv")
baseline_records = loader.load()

# Precompute global baselines
global_countries = set(c for r in baseline_records for c in r.country_dist.keys())
from scorer import FeatureExtractor
baseline_features = [FeatureExtractor.extract(r, global_countries) for r in baseline_records]

# 2. Initialize and calibrate/train the pipeline
scorer = StatisticalScorer()
refiner = WeightRefiner()

stage1_batch = scorer.score_population(baseline_records)
_ = refiner.refine(stage1_batch)  # Locks global model weights into memory

# 3. Score any incoming single 'ChannelRecord' in milliseconds
# (Assuming `new_influencer` is a ChannelRecord object fetched from an API/Form)
stage1_single = scorer.score_single(
    target_record=new_influencer,
    baseline_records=baseline_records,
    baseline_features=baseline_features,
    global_countries=global_countries
)

final_result = refiner.refine_single(stage1_single)

print(f"Risk Level: {final_result['risk_band']} | Score: {final_result['refined_score']}")

```

---

## Outputs

The output CSV (or individual real-time dictionary payload) will yield:

* `suspicion_score`: The base weighted average anomaly score from Stage 1.
* `refined_score`: Calibrated probability metric ($0.0$ to $1.0$) from the Stage 2 SVM classifier mapping proximity to known bot patterns.
* `risk_band`: Human-readable categorization based on structural threshold limits.