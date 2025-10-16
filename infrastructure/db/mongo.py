from __future__ import annotations

from loguru import logger
from pymongo import MongoClient
from pymongo.errors import ConfigurationError, ConnectionFailure, ServerSelectionTimeoutError

from core import settings


class MongoDatabaseConnector:
    """
    Lazy singleton wrapper around MongoClient.

    Ensures only one client is created per process.
    Usage:
        from application.dataset.io.mongo_connector import connection
        db = connection[settings.DATABASE_NAME]
    """

    _instance: MongoClient | None = None

    def __new__(cls) -> MongoClient:
        if cls._instance is None:
            try:
                logger.info("Connecting to MongoDB...")
                cls._instance = MongoClient(
                    settings.DATABASE_HOST,
                    serverSelectionTimeoutMS=5000,  # 5s timeout to fail fast
                )
                # Trigger an immediate ping to confirm connectivity
                cls._instance.admin.command("ping")
                logger.success("MongoDB connection established successfully.")
            except (ConnectionFailure, ConfigurationError, ServerSelectionTimeoutError) as e:
                logger.error(f"MongoDB connection failed: {e}")
                raise

        return cls._instance


def get_client() -> MongoClient:
    """Get the singleton MongoClient instance."""
    return MongoDatabaseConnector()
