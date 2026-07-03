from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

  
@dataclass
class VideoRecord:
    video_id: str
    views: int
    likes: int
    comments: int


@dataclass
class ChannelRecord:
    channel_id: str
    niche: str                    
    subscriber_count: int
    view_count: int              
    video_count: int
    videos: List[VideoRecord] = field(default_factory=list)

    audience_country_distribution: Optional[Dict[str, float]] = None
    audience_age_distribution: Optional[Dict[str, float]] = None


@dataclass
class ScoreResult:
    channel_id: str
    suspicion_score: float                 
    risk_band: str                         
    component_scores: Dict[str, float]
    caveats: List[str]
    cluster_key: Tuple[str, int]
    cluster_size: int


def _median_mad(values: List[float]) -> Tuple[float, float]:
    """Median and MAD (scaled to be a consistent estimator of std dev
    under normality, factor 1.4826), robust to the heavy-tailed
    distributions engagement metrics actually have."""
    med = statistics.median(values)
    abs_dev = [abs(v - med) for v in values]
    mad = statistics.median(abs_dev) * 1.4826
    return med, mad


def _robust_z(x: float, median: float, mad: float, cap: float = 6.0) -> float:
    """Robust z-score, capped to avoid blow-up when MAD is near zero
    (common in small or unusually homogeneous clusters)."""
    if mad <= 1e-9:
        if abs(x - median) < 1e-9:
            return 0.0
        return math.copysign(cap, x - median)
    z = (x - median) / mad
    return max(-cap, min(cap, z))


def _normalized_entropy(dist: Dict[str, float]) -> Optional[float]:
    """Shannon entropy normalized by log(k) so it's comparable across
    channels that report different numbers of categories — raw entropy
    is not comparable across varying k, normalizing is what makes this
    usable as a cross-channel feature."""
    values = [v for v in dist.values() if v > 0]
    if len(values) < 2:
        return None
    total = sum(values)
    probs = [v / total for v in values]
    h = -sum(p * math.log(p) for p in probs)
    h_max = math.log(len(probs))
    return h / h_max if h_max > 0 else None


def _follower_bucket(subscribers: int) -> int:
    """Log10 bucket. Justified because engagement rate has a well
    documented power-law decay with audience size — comparing a 5M-sub
    channel to a 5K-sub channel on raw engagement rate is not a fair
    baseline, comparing within a log-decade is."""
    return math.floor(math.log10(max(subscribers, 1)))


def extract_features(channel: ChannelRecord) -> Dict[str, Optional[float]]:
    videos = channel.videos
    total_views = sum(v.views for v in videos) or 1
    total_likes = sum(v.likes for v in videos)
    total_comments = sum(v.comments for v in videos)

    engagement_rate = total_likes / total_views
    comment_rate = total_comments / total_views

    # Cross-video dispersion of per-video engagement rate.
    dispersion_cv = None
    if len(videos) >= 5:
        rates = [v.likes / v.views for v in videos if v.views > 0]
        if len(rates) >= 5:
            mean_r = statistics.mean(rates)
            if mean_r > 1e-9:
                dispersion_cv = statistics.pstdev(rates) / mean_r

    subs_per_video = (
        channel.subscriber_count / channel.video_count
        if channel.video_count > 0 else None
    )

    country_entropy = (
        _normalized_entropy(channel.audience_country_distribution)
        if channel.audience_country_distribution else None
    )
    age_entropy = (
        _normalized_entropy(channel.audience_age_distribution)
        if channel.audience_age_distribution else None
    )

    return {
        "engagement_rate": engagement_rate,
        "comment_rate": comment_rate,
        "dispersion_cv": dispersion_cv,
        "subs_per_video": subs_per_video,
        "country_entropy": country_entropy,
        "age_entropy": age_entropy,
    }

def build_clusters(
    population: List[ChannelRecord],
) -> Dict[Tuple[str, int], List[Dict[str, Optional[float]]]]:
    clusters: Dict[Tuple[str, int], List[Dict[str, Optional[float]]]] = {}
    for ch in population:
        key = (ch.niche, _follower_bucket(ch.subscriber_count))
        clusters.setdefault(key, []).append(extract_features(ch))
    return clusters


def get_baseline(
    key: Tuple[str, int],
    clusters: Dict[Tuple[str, int], List[Dict[str, Optional[float]]]],
    min_size: int = 10,
) -> Tuple[List[Dict[str, Optional[float]]], Tuple[str, int]]:
    """Falls back from (niche, bucket) -> bucket-only -> global pool if
    the peer cluster is too small to give stable robust statistics."""
    niche, bucket = key
    if key in clusters and len(clusters[key]) >= min_size:
        return clusters[key], key

    bucket_only = [
        feats
        for (n, b), rows in clusters.items()
        if b == bucket
        for feats in rows
    ]
    if len(bucket_only) >= min_size:
        return bucket_only, ("__any_niche__", bucket)

    everyone = [feats for rows in clusters.values() for feats in rows]
    return everyone, ("__global__", -1)


BASE_WEIGHTS = {
    "comment_sparsity": 0.30,     # one-directional: only low comment_rate is suspicious
    "engagement_anomaly": 0.25,   # two-directional
    "dispersion_anomaly": 0.15,   # two-directional
    "subs_per_video": 0.10,       # one-directional, confounded -> low weight
    "country_entropy": 0.12,      # one-directional: only low entropy is suspicious
    "age_entropy": 0.08,          # one-directional
}


def score_channel(
    target: ChannelRecord,
    population: List[ChannelRecord],
    min_cluster_size: int = 10,
) -> ScoreResult:
    caveats = []

    clusters = build_clusters(population)
    key = (target.niche, _follower_bucket(target.subscriber_count))
    peers, used_key = get_baseline(key, clusters, min_cluster_size)
    if used_key != key:
        caveats.append(
            f"Peer cluster {key} too small; backed off to {used_key} "
            f"(n={len(peers)}). Results are less reliable."
        )

    target_feats = extract_features(target)

    component_scores: Dict[str, float] = {}
    active_weights: Dict[str, float] = {}

    def peer_values(field_name: str) -> List[float]:
        return [p[field_name] for p in peers if p.get(field_name) is not None]

    # comment sparsity — penalize only when BELOW peer median
    vals = peer_values("comment_rate")
    if len(vals) >= min_cluster_size and target_feats["comment_rate"] is not None:
        med, mad = _median_mad(vals)
        z = _robust_z(target_feats["comment_rate"], med, mad)
        component_scores["comment_sparsity"] = max(0.0, -z) / 6.0
        active_weights["comment_sparsity"] = BASE_WEIGHTS["comment_sparsity"]
    else:
        caveats.append("Insufficient peer data for comment_rate; signal skipped.")

    # engagement rate anomaly — either direction is suspicious
    vals = peer_values("engagement_rate")
    if len(vals) >= min_cluster_size and target_feats["engagement_rate"] is not None:
        med, mad = _median_mad(vals)
        z = _robust_z(target_feats["engagement_rate"], med, mad)
        component_scores["engagement_anomaly"] = abs(z) / 6.0
        active_weights["engagement_anomaly"] = BASE_WEIGHTS["engagement_anomaly"]
    else:
        caveats.append("Insufficient peer data for engagement_rate; signal skipped.")

    # cross-video dispersion anomaly — either direction
    vals = peer_values("dispersion_cv")
    if (
        len(vals) >= min_cluster_size
        and target_feats["dispersion_cv"] is not None
    ):
        med, mad = _median_mad(vals)
        z = _robust_z(target_feats["dispersion_cv"], med, mad)
        component_scores["dispersion_anomaly"] = abs(z) / 6.0
        active_weights["dispersion_anomaly"] = BASE_WEIGHTS["dispersion_anomaly"]
    else:
        caveats.append(
            "Dispersion signal skipped (needs >=5 videos for target and peers)."
        )

    # subs/video — penalize only when ABOVE peer median, low weight
    vals = peer_values("subs_per_video")
    if len(vals) >= min_cluster_size and target_feats["subs_per_video"] is not None:
        med, mad = _median_mad(vals)
        z = _robust_z(target_feats["subs_per_video"], med, mad)
        component_scores["subs_per_video"] = max(0.0, z) / 6.0
        active_weights["subs_per_video"] = BASE_WEIGHTS["subs_per_video"]
    else:
        caveats.append("Insufficient peer data for subs_per_video; signal skipped.")

    # audience entropy — core signals now that audienceProfile is collected.
    # Missing on the target or too sparse among peers is treated as a
    # per-record/per-cluster data gap, not a systemic unavailability.
    if target_feats["country_entropy"] is not None:
        vals = peer_values("country_entropy")
        if len(vals) >= min_cluster_size:
            med, mad = _median_mad(vals)
            z = _robust_z(target_feats["country_entropy"], med, mad)
            component_scores["country_entropy"] = max(0.0, -z) / 6.0
            active_weights["country_entropy"] = BASE_WEIGHTS["country_entropy"]
        else:
            caveats.append(
                "Not enough peers report audience_country_distribution; "
                "country_entropy signal skipped."
            )
    else:
        caveats.append(
            "Target channel has no audience_country_distribution on record; "
            "country_entropy signal skipped for this channel."
        )

    if target_feats["age_entropy"] is not None:
        vals = peer_values("age_entropy")
        if len(vals) >= min_cluster_size:
            med, mad = _median_mad(vals)
            z = _robust_z(target_feats["age_entropy"], med, mad)
            component_scores["age_entropy"] = max(0.0, -z) / 6.0
            active_weights["age_entropy"] = BASE_WEIGHTS["age_entropy"]
        else:
            caveats.append(
                "Not enough peers report audience_age_distribution; "
                "age_entropy signal skipped."
            )
    else:
        caveats.append(
            "Target channel has no audience_age_distribution on record; "
            "age_entropy signal skipped for this channel."
        )

    # Renormalize weights over whatever signals actually fired.
    weight_sum = sum(active_weights.values())
    if weight_sum > 0:
        weighted = sum(
            component_scores[k] * (w / weight_sum) for k, w in active_weights.items()
        )
    else:
        weighted = 0.0
        caveats.append("No signals could be computed — score is uninformative.")

    suspicion_score = weighted  # already bounded ~[0,1] by construction

    suspicion_score = round(min(1.0, suspicion_score), 4)

    if suspicion_score >= 0.75:
        band = "High"
    elif suspicion_score >= 0.45:
        band = "Elevated"
    elif suspicion_score >= 0.20:
        band = "Low"
    else:
        band = "Minimal"
        
    return ScoreResult(
        channel_id=target.channel_id,
        suspicion_score=suspicion_score,
        risk_band=band,
        component_scores={k: round(v, 4) for k, v in component_scores.items()},
        caveats=caveats,
        cluster_key=used_key,
        cluster_size=len(peers),
    )


if __name__ == "__main__":
    import random

    random.seed(0)

    ORGANIC_COUNTRY_POOL = ["BR", "US", "PT", "MX", "AR", "ES", "CO", "IN", "DE", "FR"]
    ORGANIC_AGE_BUCKETS = ["13-17", "18-24", "25-34", "35-44", "45-54", "55-64"]

    def make_organic_channel(i: int) -> ChannelRecord:
        subs = random.randint(50_000, 500_000)
        videos = []
        for v in range(20):
            views = random.randint(5_000, 60_000)
            likes = int(views * random.uniform(0.03, 0.07))
            comments = int(likes * random.uniform(0.02, 0.06))
            videos.append(VideoRecord(f"v{i}_{v}", views, likes, comments))

        # Organic audiences: spread across several countries, peaked age curve.
        countries = random.sample(ORGANIC_COUNTRY_POOL, k=6)
        country_dist = {c: random.uniform(5, 25) for c in countries}
        age_dist = {
            "13-17": random.uniform(3, 8),
            "18-24": random.uniform(25, 35),
            "25-34": random.uniform(25, 35),
            "35-44": random.uniform(10, 20),
            "45-54": random.uniform(3, 8),
            "55-64": random.uniform(1, 4),
        }

        return ChannelRecord(
            channel_id=f"organic_{i}",
            niche="tech",
            subscriber_count=subs,
            view_count=sum(v.views for v in videos),
            video_count=200 + i,
            videos=videos,
            audience_country_distribution=country_dist,
            audience_age_distribution=age_dist,
        )

    def make_suspicious_channel() -> ChannelRecord:
        subs = 300_000
        videos = []
        for v in range(20):
            views = random.randint(5_000, 60_000)
            likes = int(views * 0.05)          # engagement looks normal
            comments = int(likes * 0.001)       # but comments are near-dead
            videos.append(VideoRecord(f"sus_{v}", views, likes, comments))

        # Purchased-audience pattern: concentrated in 2 countries, flat age curve.
        country_dist = {"BR": 80.0, "IN": 15.0, "US": 5.0}
        age_dist = {b: 100.0 / len(ORGANIC_AGE_BUCKETS) for b in ORGANIC_AGE_BUCKETS}

        return ChannelRecord(
            channel_id="suspect_1",
            niche="tech",
            subscriber_count=subs,
            view_count=sum(v.views for v in videos),
            video_count=15,   # huge subs, almost no content -> high subs/video
            videos=videos,
            audience_country_distribution=country_dist,
            audience_age_distribution=age_dist,
        )

    population = [make_organic_channel(i) for i in range(50)]
    target = make_suspicious_channel()

    result = score_channel(target, population)

    print(f"channel_id:       {result.channel_id}")
    print(f"suspicion_score:  {result.suspicion_score}  ({result.risk_band})")
    print(f"cluster:          {result.cluster_key}  (n={result.cluster_size})")
    print("component_scores:")
    for k, v in result.component_scores.items():
        print(f"  - {k}: {v}")
    print("caveats:")
    for c in result.caveats:
        print(f"  - {c}")