"""
ingestion/__init__.py
---------------------
Exposes the public API of the ingestion package.
"""

from ingestion.pipeline import run_ingestion_pipeline

__all__ = ["run_ingestion_pipeline"]
