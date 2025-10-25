# This pipeline writes items to MongoDB efficiently with batched upserts.
# It is defensive against missing 'url' fields and resilient to index/bulk errors.

from itemadapter import ItemAdapter
from pymongo import ASCENDING, MongoClient, UpdateOne
from pymongo.errors import BulkWriteError, DuplicateKeyError, OperationFailure
from scrapy.exceptions import DropItem

from .settings import MONGO_COLLECTION


class BotPipeline:
    collection_name = MONGO_COLLECTION  # Or a dynamic name based on your item class

    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db
        self._buffer = []
        self._bufsize = 500

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get("MONGO_URI"),
            mongo_db=crawler.settings.get("MONGO_DATABASE", "items"),
        )

    def open_spider(self, spider):
        self.client = MongoClient(self.mongo_uri, retryWrites=True)
        self.db = self.client[self.mongo_db]
        try:
            self.db[self.collection_name].create_index(
                [("url", ASCENDING)],
                name="url_unique",
                unique=True,
                background=True,
            )
        except (DuplicateKeyError, OperationFailure) as exc:
            spider.logger.warning("Unable to ensure unique index on 'url': %s", exc)

    def _flush(self, spider):
        if not self._buffer:
            return
        try:
            self.db[self.collection_name].bulk_write(self._buffer, ordered=False)
        except BulkWriteError as exc:
            write_errors = exc.details.get("writeErrors", []) if exc.details else []
            dup_errors = [e for e in write_errors if e.get("code") == 11000]
            other_errors = [e for e in write_errors if e.get("code") != 11000]
            if dup_errors:
                spider.logger.info("BulkWrite: %d duplicate key errors ignored.", len(dup_errors))
            if other_errors:
                spider.logger.warning("BulkWrite had %d non-duplicate errors: %s", len(other_errors), other_errors[:3])
        finally:
            self._buffer.clear()

    def close_spider(self, spider):
        self._flush(spider)
        self.client.close()

    def process_item(self, item, spider):
        doc = ItemAdapter(item).asdict()

        url = doc.get("url")
        if not url:
            raise DropItem("Item missing required 'url' field")

        key = {"url": url}
        self._buffer.append(UpdateOne(key, {"$set": doc}, upsert=True))

        if len(self._buffer) >= self._bufsize:
            self._flush(spider)

        return item
