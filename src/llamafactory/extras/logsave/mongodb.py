import os

from pymongo import MongoClient


def new_client(uri='', from_env=''):
    """Creates a new MongoClient instance using the provided URI or the value of the environment variable MONGODB_URI.

    Args:
        uri (str, optional): Mongodb uri. Defaults to ''.
        from_env (str, optional): environment that contains Mongodb uri. Defaults to ''.

    Raises:
        ValueError: mongodb uri is required, either pass it as an argument or set it in the environment variable MONGODB_URI

    Returns:
        MongoClient: A new MongoClient instance
    """
    if from_env:
        uri = os.environ[from_env]
    if not uri:
        raise ValueError("mongodb uri is required, either pass it as an argument or set it in the environment variable MONGODB_URI")
    return MongoClient(uri)


def default_client():
    """Reads the MongoDB URI from the environment variable MONGODB_URI and returns a new MongoClient instance.

    Returns:
        MongoClient: A new MongoClient instance
    """
    return new_client(from_env="MONGODB_URI")
