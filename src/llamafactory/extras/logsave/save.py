import os
from typing import Optional

from pymongo import MongoClient

from .mongodb import default_client


db_key = 'MONGODB_DB'
default_db = 'training'
collection_key = 'MONGODB_COLLECTION'
default_collection = 'metrics'

class LogSaver:
    mongo_client:  Optional[MongoClient] = None
    def __init__(self):
        self.mongo_client = default_client()
        db_name = os.environ.get(db_key, default_db)
        self.db = self.mongo_client[db_name]
        collection_name = os.environ.get(collection_key, default_collection)
        self.collection = self.db[collection_name]

    def save(self, log_entries: dict):
        task_id = os.environ.get('TASK_ID', 'unknown')
        log_entries['task_id'] = task_id
        self.collection.insert_one(log_entries)

saver: Optional[LogSaver] = None

def save_logs(log_entries: dict):
    global saver
    try:
        if saver is None:
            saver = LogSaver()
        saver.save(log_entries)
    except Exception as e:
        print(f"Failed to save logs: {e}")
