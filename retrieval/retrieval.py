import pickle

from icecream import ic


class Retrieval:
    def __init__(self, query: str, k:int) -> None:
        self.query = query
        self.k = k

    