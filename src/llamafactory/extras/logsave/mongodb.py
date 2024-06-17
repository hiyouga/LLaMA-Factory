import os

from pymongo import MongoClient


def new_client(uri='', from_env=''):
    if from_env:
        uri = os.environ[from_env]
    if not uri:
        raise ValueError("mongodb uri is required, either pass it as an argument or set it in the environment variable MONGODB_URI")
    return MongoClient(uri)


def default_client():
    return new_client(from_env="MONGODB_URI")
