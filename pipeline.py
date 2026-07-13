import logging
from typing import List, Dict, Optional

from .channel_record import ChannelRecord
from .record_builder import ChannelRecordBuilder
from .scorer import StatisticalScorer, FeatureExtractor
from .refiner import WeightRefiner

logger = logging.getLogger("father.bot-detection-pipeline")


class BotDetectorPipeline:
    """Two-stage bot-detection pipeline, sourced live from Brandconnect
    (via ChannelExtractor + the Mongo CRUD layer) instead of a CSV file.

    Usage:
        pipeline = BotDetectorPipeline()

        # Bootstrap / retrain the baseline on a known population.
        # Run this once up front, and again periodically to keep the
        # baseline + refiner current.
        pipeline.run_population(channel_ids, refresh=True)

        # From then on, score new influencers one at a time against the
        # cached baseline:
        result = pipeline.score_new_influencer("UCxxxxxxxxxxxxxxxxxxxxxx")
    """

    def __init__(self):
        self.builder = ChannelRecordBuilder()
        self.scorer = StatisticalScorer()
        self.refiner = WeightRefiner()

        self._baseline_records: List[ChannelRecord] = []
        self._baseline_features: List[Dict] = []
        self._global_countries: set = set()
        self._trained = False

    def run_population(self, channel_ids: List[str], refresh: bool = False) -> List[Dict]:
        """Extracts (optionally) + scores a full population, and trains the
        refiner + caches the baseline for later single-influencer scoring.
        """
        logger.info(f"[*] Building records for {len(channel_ids)} channels...")
        records = self.builder.build_population(channel_ids, refresh=refresh)
        logger.info(f"[*] Built {len(records)} valid records "
                    f"({len(channel_ids) - len(records)} skipped due to missing data).")

        logger.info("[*] Stage 1: Running statistical scoring...")
        stage1_results = self.scorer.score_population(records)

        logger.info("[*] Stage 2: Refining weights with PCA + SVM...")
        final_results = self.refiner.refine(stage1_results)

        # Cache baseline state for score_new_influencer()
        self._baseline_records = records
        self._global_countries = {c for r in records for c in r.country_dist.keys()}
        self._baseline_features = [
            FeatureExtractor.extract(r, self._global_countries) for r in records
        ]
        self._trained = True

        logger.info("[+] Baseline population trained and cached.")
        return final_results

    def score_new_influencer(self, channel_id: str, refresh: bool = True) -> Optional[Dict]:
        """Pulls one influencer live (via ChannelExtractor, if refresh=True)
        and scores them against the cached baseline population.
        """
        if not self._trained:
            raise ValueError("Call run_population() at least once before scoring single influencers.")

        record = self.builder.build_single(channel_id, refresh=refresh)
        if record is None:
            logger.warning(f"Could not build a record for channel {channel_id}; skipping.")
            return None

        stage1_result = self.scorer.score_single(
            record, self._baseline_records, self._baseline_features, self._global_countries
        )
        return self.refiner.refine_single(stage1_result)