from enum import Enum


class PathsAndNames(Enum):
    corpus_path = "data/raw"
    save_index_path = "data/processed"
    index_name = "bm25_index.pkl"
    save_chunks = "data/processed"
    chunks_json = "/chunks.json"


TQDM_FMT = (
    "{desc:<32}{percentage:3.0f}%|{bar:30}| "
    "{n_fmt:>6}/{total_fmt:<6} [{elapsed}<{remaining}]"
)
