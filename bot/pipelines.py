# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html


# useful for handling different item types with a single interface
from itemadapter import ItemAdapter
from pymongo import ASCENDING, MongoClient, UpdateOne

from .settings import MONGO_COLLECTION


class BotPipeline:
    collection_name = MONGO_COLLECTION  # Or a dynamic name based on your item class

    def __init__(self, mongo_uri, mongo_db):
        self.mongo_uri = mongo_uri
        self.mongo_db = mongo_db

    @classmethod
    def from_crawler(cls, crawler):
        return cls(
            mongo_uri=crawler.settings.get("MONGO_URI"), mongo_db=crawler.settings.get("MONGO_DATABASE", "items")
        )

    def open_spider(self, spider):
        self.client = MongoClient(self.mongo_uri, retryWrites=True)
        self.db = self.client[self.mongo_db]
        self.db[self.collection_name].create_index([("url", ASCENDING)], unique=True)

        self._buffer = []
        self._bufsize = 500

    def close_spider(self, spider):
        if self._buffer:
            self.db[self.collection_name].bulk_write(self._buffer, ordered=False)
        self.client.close()

    def process_item(self, item, spider):
        doc = ItemAdapter(item).asdict()
        key = {"url": doc["url"]}  # or another stable key
        self._buffer.append(UpdateOne(key, {"$set": doc}, upsert=True))
        if len(self._buffer) >= self._bufsize:
            self.db[self.collection_name].bulk_write(self._buffer, ordered=False)
            self._buffer.clear()
        return item
