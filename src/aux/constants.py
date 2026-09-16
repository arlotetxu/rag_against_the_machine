from enum import Enum


class PathsAndNames(Enum):
    corpus_path = "data/raw"
    save_index_path = "data/processed"
    index_name = "bm25_index.pkl"
    save_chunks = "data/processed"
    chunks_json = "/chunks.json"
