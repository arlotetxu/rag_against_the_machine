"""Project-wide paths, tuning parameters and display settings.

Attributes:
    MIN_RATIO (float): Share of chunks below which a file type (Python code
        or documentation) counts as under-represented in the corpus.
    BOOST (float): Multiplier applied to the BM25 scores of chunks from the
        under-represented file type, so they are not drowned out at
        retrieval time.
    MAX_OUT_TOKENS (int): Maximum number of new tokens the LLM may generate
        per answer.
    K_FOR_ANSWER (int): Maximum number of retrieved chunks used as context
        for each question in batch answer generation.
    TQDM_FMT (str): ``bar_format`` string shared by all tqdm progress bars.
"""
from enum import Enum


class PathsAndNames(Enum):
    """Locations of the corpus and of the files produced by the indexer.

    Paths are relative to the project root, where the commands are run.

    Attributes:
        corpus_path: Folder with the raw documents to index.
        save_index_path: Folder where the pickled BM25 index is stored.
        index_name: File name of the pickled BM25 index.
        save_chunks: Folder where the chunks JSON file is stored.
        chunks_json: File name of the chunks JSON file. It keeps its
            leading ``/`` because it is joined to ``save_chunks`` by string
            concatenation.
    """

    corpus_path = "data/raw"
    save_index_path = "data/processed"
    index_name = "bm25_index.pkl"
    save_chunks = "data/processed"
    chunks_json = "/chunks.json"
    embeddings_name = "embeddings.npy"
    embeddings_info_name = "embeddings_info.json"
    cache_path = "data/cache"
    cache_sources_name = "sources.json"
    cache_answers_name = "answers.json"


MIN_RATIO = 0.3
BOOST = 1.2

MAX_OUT_TOKENS = 200
K_FOR_ANSWER = 3

EMBEDDINGS_BATCH_SIZE = 64

TQDM_FMT = (
    "{desc:<32}{percentage:3.0f}%|{bar:30}| "
    "{n_fmt:>6}/{total_fmt:<6} [{elapsed}<{remaining}]"
)
