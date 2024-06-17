import os
from typing import Optional

from pymongo import MongoClient

from .mongodb import default_client

# key of the environment variable that contains the MongoDB URI
db_key = 'MONGODB_DB'
# default MongoDB database name
default_db = 'training'
# key of the environment variable that contains the MongoDB collection name
collection_key = 'MONGODB_COLLECTION'
# default MongoDB collection name
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
        """Save task logs to MongoDB

        Args:
            log_entries (dict): task logs, e.g. {'epoch': 1, 'loss': 0.1}
        """
        # add task_id to log entries
        task_id = os.environ.get('TASK_ID', 'unknown')
        log_entries['task_id'] = task_id
        self.collection.insert_one(log_entries)

saver: Optional[LogSaver] = None

def save_logs(log_entries: dict):
    """Save logs to MongoDB

    Args:
        log_entries (dict): training logs, e.g. {'epoch': 1, 'loss': 0.1}
    """
    global saver
    task_id = os.environ.get('TASK_ID')
    if not task_id:
        return
    try:
        if saver is None:
            saver = LogSaver()
        saver.save(log_entries)
    except Exception as e:
        print(f"Failed to save logs: {e}")
