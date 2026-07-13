import logging
from typing import List, Optional

from youtube.extract.brandconnect.extractors.channel_extractor import ChannelExtractor
from youtube.extract.brandconnect.crud.channelpublicdata_crud import ChannelPublicDataCRUD
from youtube.extract.brandconnect.crud.channelinsights_crud import ChannelInsightCRUD

from .channel_record import ChannelRecord

logger = logging.getLogger("father.bot-detection-record-builder")

# snake_case to match StatisticalScorer.GLOBAL_AGE_BUCKETS.
AGE_KEY_MAP = {
    "age18To24": "age18_24",
    "age25To34": "age25_34",
    "age35To44": "age35_44",
    "age45To54": "age45_54",
    "age55To64": "age55_64",
    "age65ToInf": "age65_plus",
}


class ChannelRecordBuilder:
    """Bridges the Brandconnect extractor/CRUD layer with the bot-detection
    pipeline's ChannelRecord format.

    Flow for a single channel:
      1. (optional) Trigger extraction: Brandconnect API -> Mongo, via ChannelExtractor
      2. Read the persisted public-data + insight docs back out of Mongo
      3. Map those docs -> ChannelRecord
    """

    def __init__(self):
        self.public_crud = ChannelPublicDataCRUD()
        self.insight_crud = ChannelInsightCRUD()

    def build_single(self, channel_id: str, refresh: bool = True) -> Optional[ChannelRecord]:
        if refresh:
            ChannelExtractor.extract_data([channel_id])

        public_doc = self.public_crud.get_collection().find_one(
            {"channel_id": channel_id},
            sort=[("creation_date", -1)],  # matches the field name extract_channel_public_data actually sets
        )
        insight_doc = self.insight_crud.get_collection().find_one(
            {"channel_id": channel_id},
            sort=[("collected_at", -1)],  # insights are historical/append-only; take the latest
        )

        if public_doc is None:
            logger.warning(f"No public data found for channel {channel_id}")
            return None
        if insight_doc is None:
            logger.warning(f"No insight data found for channel {channel_id}")
            return None

        return self._to_channel_record(public_doc, insight_doc)

    def build_population(self, channel_ids: List[str], refresh: bool = False) -> List[ChannelRecord]:
        if refresh:
            ChannelExtractor.extract_data(channel_ids)

        records = []
        for cid in channel_ids:
            # Extraction already done above (or skipped) for the whole batch,
            # so no need to refresh per-channel here.
            rec = self.build_single(cid, refresh=False)
            if rec is not None:
                records.append(rec)
        return records

    def _to_channel_record(self, public_doc: dict, insight_doc: dict) -> ChannelRecord:
        overall = insight_doc.get("OVERALL") or {}

        record = ChannelRecord(
            channel_id=public_doc.get("channel_id") or overall.get("external_channel_id"),
            subscriber_count=int(public_doc.get("subscriber_count") or overall.get("subscriber_count") or 0),
            video_count=int(public_doc.get("video_count") or overall.get("video_count") or 0),
            views=float(overall.get("views", 0)),
            likes=float(overall.get("likes", 0)),
            comments=float(overall.get("comments", 0)),
            engagement_rate=float(overall.get("engagement_rate", 0)),
        )

        # Age distribution: rename camelCase API keys -> internal snake_case buckets
        raw_age = overall.get("audience_age_distribution") or {}
        record.age_dist = {
            AGE_KEY_MAP[k]: float(v)
            for k, v in raw_age.items()
            if k in AGE_KEY_MAP
        }

        # Country distribution: top-level region_code/percentage only
        # (nested per-region breakdown under each entry is ignored — finer
        # grain than the entropy calculation needs)
        raw_countries = overall.get("audience_country_distribution") or []
        record.country_dist = {
            entry["region_code"]: float(entry["percentage"])
            for entry in raw_countries
            if entry.get("region_code")
        }

        # Not used by scorer.py / refiner.py — left empty on purpose.
        record.interests = []

        topic_categories = public_doc.get("topic_categories") or overall.get("topic_categories") or []
        if topic_categories:
            record.niche = topic_categories[0].split('/')[-1].replace('_', ' ')

        return record