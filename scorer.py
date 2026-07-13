import math
import statistics
from typing import Dict, List, Optional, Tuple
from .channel_record import ChannelRecord


class FeatureExtractor:
    # Define global category pools to fix the entropy normalization baseline
    GLOBAL_AGE_BUCKETS = ['age18_24', 'age25_34', 'age35_44', 'age45_54', 'age55_64', 'age65_plus']

    @staticmethod
    def extract(record: ChannelRecord, global_countries: set) -> Dict[str, float]:
        features = {}

        # Avoids assigning a flat 0.0 to both a channel with 0 views and a channel with 1M views and 0 comments.
        views_smoothed = record.views + 1.0
        features['engagement_rate'] = record.engagement_rate
        features['comment_rate'] = (record.comments + 0.01) / views_smoothed
        features['likes_per_view'] = (record.likes + 0.01) / views_smoothed

        # Pass the total possible baseline categories so the denominator is uniform across the population
        features['age_entropy'] = FeatureExtractor._normalized_entropy(
            record.age_dist, total_possible_categories=len(FeatureExtractor.GLOBAL_AGE_BUCKETS)
        )
        features['country_entropy'] = FeatureExtractor._normalized_entropy(
            record.country_dist, total_possible_categories=max(len(global_countries), 2)
        )

        # Growth/Efficiency smoothing
        features['subs_per_video'] = record.subscriber_count / (record.video_count + 1)

        return features

    @staticmethod
    def _normalized_entropy(dist: Dict[str, float], total_possible_categories: int) -> float:
        values = [v for v in dist.values() if v > 0]
        if not values or total_possible_categories < 2:
            return 1.0  # 1.0 represents max entropy / normal state

        total = sum(values)
        probs = [v / total for v in values]
        h = -sum(p * math.log(p) for p in probs)

        # Normalize using the global category count space instead of len(probs)
        h_max = math.log(total_possible_categories)
        return h / h_max if h_max > 0 else 1.0


class StatisticalScorer:
    def __init__(self, min_cluster_size: int = 15):
        self.min_cluster_size = min_cluster_size
        self.base_weights = {
            "engagement_anomaly": 0.3,
            "comment_sparsity": 0.25,
            "country_entropy": 0.2,
            "age_entropy": 0.15,
            "subs_per_video": 0.1
        }

    def score_population(self, records: List[ChannelRecord]) -> List[Dict]:
        # Build global country registry dynamically from the population to set a fair entropy baseline
        global_countries = set()
        for r in records:
            global_countries.update(r.country_dist.keys())

        # Extract features using the fixed global baseline dimensions
        all_features = [FeatureExtractor.extract(r, global_countries) for r in records]

        # Precompute log10 subscribers for smooth continuous peer selection
        log_subs = [math.log10(max(r.subscriber_count, 1)) for r in records]

        # Compute continuous peer-relative scores
        results = []
        for i, r in enumerate(records):
            target_log = log_subs[i]
            niche_indices = [idx for idx, rec in enumerate(records) if rec.niche == r.niche]
            pool_indices = niche_indices if len(niche_indices) >= self.min_cluster_size else list(range(len(records)))

            # Continuous window selection
            peer_indices = sorted(pool_indices, key=lambda idx: abs(log_subs[idx] - target_log))[:max(self.min_cluster_size, 30)]

            peer_features = [all_features[idx] for idx in peer_indices]
            score_data = self._compute_z_scores(all_features[i], peer_features)

            suspicion_score = self._calculate_final_score(score_data)

            results.append({
                "channel_id": r.channel_id,
                "suspicion_score": suspicion_score,
                "features": all_features[i],
                "anomalies": score_data
            })

        return results

    def _compute_z_scores(self, target: Dict[str, float], peers: List[Dict[str, float]]) -> Dict[str, float]:
        anomalies = {}

        def get_robust_z(val, peer_vals):
            if not peer_vals: return 0.0
            med = statistics.median(peer_vals)
            abs_dev = [abs(v - med) for v in peer_vals]
            mad = statistics.median(abs_dev) * 1.4826
            if mad < 1e-9: return 0.0
            return (val - med) / mad

        anomalies['engagement_anomaly'] = abs(get_robust_z(target['engagement_rate'], [p['engagement_rate'] for p in peers]))
        anomalies['comment_sparsity'] = max(0, -get_robust_z(target['comment_rate'], [p['comment_rate'] for p in peers]))
        anomalies['country_entropy'] = max(0, -get_robust_z(target['country_entropy'], [p['country_entropy'] for p in peers]))
        anomalies['age_entropy'] = max(0, -get_robust_z(target['age_entropy'], [p['age_entropy'] for p in peers]))
        anomalies['subs_per_video'] = max(0, get_robust_z(target['subs_per_video'], [p['subs_per_video'] for p in peers]))

        return anomalies

    def _calculate_final_score(self, anomalies: Dict[str, float]) -> float:
        score = 0.0
        for feat, weight in self.base_weights.items():
            z = anomalies.get(feat, 0.0)
            norm_z = 2 / (1 + math.exp(-0.366 * z)) - 1
            score += norm_z * weight
        return round(score, 4)

    def score_single(self, target_record: ChannelRecord, baseline_records: List[ChannelRecord], baseline_features: List[Dict], global_countries: set) -> Dict:
        """Scores a single incoming influencer using the existing population as the peer baseline."""
        # Extract features using the fixed global country count
        target_feat = FeatureExtractor.extract(target_record, global_countries)

        # Continuous window selection from baseline population
        target_log = math.log10(max(target_record.subscriber_count, 1))

        niche_indices = [idx for idx, rec in enumerate(baseline_records) if rec.niche == target_record.niche]
        pool_indices = niche_indices if len(niche_indices) >= self.min_cluster_size else list(range(len(baseline_records)))

        # Find closest peers in the baseline list based on log subscriber count
        peer_indices = sorted(pool_indices, key=lambda idx: abs(math.log10(max(baseline_records[idx].subscriber_count, 1)) - target_log))[:max(self.min_cluster_size, 30)]
        peer_features = [baseline_features[idx] for idx in peer_indices]

        # Compute peer-relative scores
        score_data = self._compute_z_scores(target_feat, peer_features)
        suspicion_score = self._calculate_final_score(score_data)

        return {
            "channel_id": target_record.channel_id,
            "suspicion_score": suspicion_score,
            "features": target_feat,
            "anomalies": score_data
        }