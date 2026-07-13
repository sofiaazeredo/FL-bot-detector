from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import csv

@dataclass
class ChannelRecord:
    channel_id: str
    subscriber_count: int
    video_count: int
    views: float
    likes: float
    comments: float
    engagement_rate: float
    
    # Audience Age Distribution
    age_dist: Dict[str, float] = field(default_factory=dict)
    
    # Audience Country Distribution
    country_dist: Dict[str, float] = field(default_factory=dict)
    
    # Audience Interests
    interests: List[Dict[str, float]] = field(default_factory=list)
    
    # Niche derived from topic_categories
    niche: str = "Unknown"

class CSVLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load(self) -> List[ChannelRecord]:
        records = []
        with open(self.file_path, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(self._parse_row(row))
        return records

    def _parse_row(self, row: Dict[str, str]) -> ChannelRecord:
        # Basic metrics
        record = ChannelRecord(
            channel_id=row['external_channel_id'],
            subscriber_count=int(float(row['subscriber_count'] or 0)),
            video_count=int(float(row['video_count'] or 0)),
            views=float(row['views'] or 0),
            likes=float(row['likes'] or 0),
            comments=float(row['comments'] or 0),
            engagement_rate=float(row['engagement_rate'] or 0)
        )

        # Age distribution
        age_keys = ['age18_24', 'age25_34', 'age35_44', 'age45_54', 'age55_64', 'age65_plus']
        record.age_dist = {k: float(row[k] or 0) for k in age_keys if k in row}

        # Country distribution
        for i in range(1, 4):
            c_key = f'country{i}'
            s_key = f'country{i}_share'
            if row.get(c_key) and row.get(s_key):
                record.country_dist[row[c_key]] = float(row[s_key])

        # Interests
        for i in range(1, 4):
            name_key = f'interest{i}'
            share_key = f'interest{i}_share'
            rel_key = f'interest{i}_relevance'
            if row.get(name_key):
                record.interests.append({
                    "name": row[name_key],
                    "share": float(row[share_key] or 0),
                    "relevance": float(row[rel_key] or 0)
                })

        # Niche extraction
        if row.get('topic_categories'):
            categories = row['topic_categories'].split(';')
            if categories:
                # Get the last segment of the first category URL
                last_segment = categories[0].split('/')[-1]
                record.niche = last_segment.replace('_', ' ')

        return record
