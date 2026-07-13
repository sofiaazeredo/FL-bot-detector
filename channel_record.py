from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class ChannelRecord:
    channel_id: str
    subscriber_count: int
    video_count: int
    views: float
    likes: float
    comments: float
    engagement_rate: float

    # Audience Age Distribution (keys: age18_24, age25_34, ..., age65_plus)
    age_dist: Dict[str, float] = field(default_factory=dict)

    # Audience Country Distribution (keys: ISO region codes, e.g. "US", "IN")
    country_dist: Dict[str, float] = field(default_factory=dict)

    # Audience Interests (no use currently)
    interests: List[Dict[str, float]] = field(default_factory=list)

    # Niche derived from topic_categories
    niche: str = "Unknown"