import logging
from typing import List, Optional

from youtube.extract.brandconnect.extractors.channel_extractor import ChannelExtractor
from youtube.extract.brandconnect.crud.channelpublicdata_crud import ChannelPublicDataCRUD
from youtube.extract.brandconnect.crud.channelinsights_crud import ChannelInsightCRUD

from .channel_record import ChannelRecord

logger = logging.getLogger("father.bot-detection-record-builder")

AGE_KEY_MAP = {
    "age18To24": "age18_24",
    "age25To34": "age25_34",
    "age35To44": "age35_44",
    "age45To54": "age45_54",
    "age55To64": "age55_64",
    "age65ToInf": "age65_plus",
}

_IN_CLAUSE_CHUNK_SIZE = 5000


class ChannelRecordBuilder:
    def __init__(self):
        self.public_crud = ChannelPublicDataCRUD()
        self.insight_crud = ChannelInsightCRUD()

    def build_single(self, channel_id: str, refresh: bool = True) -> Optional[ChannelRecord]:
        if refresh:
            ChannelExtractor.extract_data([channel_id])

        public_doc = self.public_crud.get_collection().find_one(
            {"channel_id": channel_id},
            sort=[("creation_date", -1)],
        )
        insight_doc = self.insight_crud.get_collection().find_one(
            {"channel_id": channel_id},
            sort=[("collected_at", -1)],
        )

        if public_doc is None:
            logger.warning(f"No public data found for channel {channel_id}")
            return None
        if insight_doc is None:
            logger.warning(f"No insight data found for channel {channel_id}")
            return None

        return self._to_channel_record(public_doc, insight_doc)

    def build_population(self, channel_ids: Optional[List[str]] = None, refresh: bool = False) -> List[ChannelRecord]:
        """
        Builds records for many channels in O(1) round trips per collection
        instead of O(N) find_one calls.

        channel_ids=None means "every channel currently in the DB" (the
        normal case for building/refreshing the baseline population).
        Pass an explicit list to build a subset.
        """
        if refresh:
            if not channel_ids:
                raise ValueError("refresh=True requires an explicit channel_ids list to extract.")
            ChannelExtractor.extract_data(channel_ids)

        public_docs = self._latest_docs_by_channel(
            self.public_crud, channel_ids, sort_field="creation_date",
            projection=["channel_id", "subscriber_count", "video_count", "topic_categories"],
        )
        insight_docs = self._latest_docs_by_channel(
            self.insight_crud, channel_ids, sort_field="collected_at",
            projection=["channel_id", "OVERALL"],
        )

        ids_to_build = channel_ids if channel_ids is not None else public_docs.keys()

        records = []
        skipped = 0
        for cid in ids_to_build:
            public_doc = public_docs.get(cid)
            insight_doc = insight_docs.get(cid)
            if public_doc is None or insight_doc is None:
                skipped += 1
                continue
            records.append(self._to_channel_record(public_doc, insight_doc))

        if skipped:
            logger.warning(f"{skipped} channels skipped (missing public data or insights)")

        return records

    @staticmethod
    def _latest_docs_by_channel(crud, channel_ids: Optional[List[str]], sort_field: str, projection: List[str]) -> dict:
        """
        Returns {channel_id: latest_doc} for either the whole collection
        (channel_ids=None) or a specific set of ids, using one aggregation
        per chunk instead of one query per channel.

        Requires a compound index on (channel_id, sort_field) to avoid a
        full collection scan on the $sort stage -- e.g.:
            db.channel_public_data.create_index([("channel_id", 1), ("creation_date", -1)])
            db.channel_insights.create_index([("channel_id", 1), ("collected_at", -1)])
        """
        results = {}

        def run_pipeline(match_stage: Optional[dict]):
            pipeline = []
            if match_stage:
                pipeline.append({"$match": match_stage})
            pipeline += [
                {"$sort": {sort_field: -1}},
                {"$group": {"_id": "$channel_id", "doc": {"$first": "$$ROOT"}}},
                {"$project": {"_id": 0, "channel_id": "$_id", **{f: f"$doc.{f}" for f in projection}}},
            ]
            for row in crud.get_collection().aggregate(pipeline):
                results[row["channel_id"]] = row

        if channel_ids is None:
            run_pipeline(match_stage=None)
        else:
            for i in range(0, len(channel_ids), _IN_CLAUSE_CHUNK_SIZE):
                chunk = channel_ids[i:i + _IN_CLAUSE_CHUNK_SIZE]
                run_pipeline(match_stage={"channel_id": {"$in": chunk}})

        return results

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

        raw_age = overall.get("audience_age_distribution") or {}
        record.age_dist = {
            AGE_KEY_MAP[k]: float(v)
            for k, v in raw_age.items()
            if k in AGE_KEY_MAP
        }

        raw_countries = overall.get("audience_country_distribution") or []
        record.country_dist = {
            entry["region_code"]: float(entry["percentage"])
            for entry in raw_countries
            if entry.get("region_code")
        }

        record.interests = []

        topic_categories = public_doc.get("topic_categories") or overall.get("topic_categories") or []
        if topic_categories:
            record.niche = topic_categories[0].split('/')[-1].replace('_', ' ')

        return record