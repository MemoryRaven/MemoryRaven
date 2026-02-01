"""Memory Empire Memory Service (distributed ingest + retrieve).

This is the production service layer that sits in front of Postgres(pgvector)
+ Qdrant + Redis.

It is intentionally small and dependency-light.
"""
