from __future__ import annotations

# OpenSearch/Elasticsearch index mappings live here (suggested).
# Keep versioned mappings for painless upgrades.

PAPER_INDEX = "aks-papers-v1"

PAPER_MAPPING = {
    "settings": {
        "number_of_shards": 6,
        "number_of_replicas": 1,
        "analysis": {
            "analyzer": {
                "folded": {"tokenizer": "standard", "filter": ["lowercase", "asciifolding"]}
            }
        },
    },
    "mappings": {
        "dynamic": "strict",
        "properties": {
            "paper_key": {"type": "keyword"},
            "doi": {"type": "keyword"},
            "arxiv_id": {"type": "keyword"},
            "title": {"type": "text", "analyzer": "folded"},
            "abstract": {"type": "text", "analyzer": "folded"},
            "authors": {"type": "text", "analyzer": "folded"},
            "year": {"type": "integer"},
            "venue": {"type": "keyword"},
            "domains": {"type": "keyword"},
            "topics": {"type": "keyword"},
            "citation_count": {"type": "integer"},
        },
    },
}
