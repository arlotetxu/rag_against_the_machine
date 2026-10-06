"""Project-wide paths, tuning parameters and display settings.

Attributes:
    MIN_RATIO (float): Share of chunks below which a file type (Python code
        or documentation) counts as under-represented in the corpus.
    BOOST (float): Multiplier applied to the BM25 scores of chunks from the
        under-represented file type, so they are not drowned out at
        retrieval time.
    MAX_OUT_TOKENS (int): Maximum number of new tokens the LLM may generate
        per answer. Part of the answer cache key.
    K_FOR_ANSWER (int): Maximum number of retrieved chunks used as context
        for each question in batch answer generation.
    EMBEDDINGS_BATCH_SIZE (int): Default number of chunks encoded per batch
        when building the embeddings matrix.
    SYSTEM_PROMPT (str): Instructions placed before the retrieved context in
        every prompt. Its hash is part of the answer cache key, so editing
        it makes the cached answers miss.
    MODEL_NAME (str): Hugging Face id of the LLM that generates the
        answers. Part of the answer cache key.
    TQDM_FMT (str): ``bar_format`` string shared by all tqdm progress bars.
"""
from enum import Enum


class PathsAndNames(Enum):
    """Locations of the corpus, the index files and the caches.

    Paths are relative to the project root, where the commands are run.

    Attributes:
        corpus_path: Folder with the raw documents to index.
        save_index_path: Folder where the pickled BM25 index and the
            embeddings files are stored.
        index_name: File name of the pickled BM25 index.
        save_chunks: Folder where the chunks JSON file is stored.
        chunks_json: File name of the chunks JSON file. It keeps its
            leading ``/`` because it is joined to ``save_chunks`` by string
            concatenation.
        embeddings_name: File name of the NumPy matrix with one embedding
            per chunk.
        embeddings_info_name: File name of the JSON file describing how the
            embeddings were built (model, pooling, dimension...).
        cache_path: Folder holding the search and answer caches.
        cache_sources_name: File name of the cache of retrieved sources.
        cache_answers_name: File name of the cache of generated answers.
            Both cache files are deleted whenever the index is rebuilt.
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

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about the vLLM "
    "source code and documentation, using ONLY the given portions "
    "of context. If the context does not contain the answer, you "
    "MUST say that the question cannot be answered with the "
    "available information. Do not use external knowledge or make "
    "up information.\n"
)

MODEL_NAME = "Qwen/Qwen3-0.6B"
# MODEL_NAME2 = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
# MODEL_NAME3 = "Qwen/Qwen2.5-0.5B-Instruct"
# MODEL_NAME4 = "HuggingFaceTB/SmolLM2-360M-Instruct"

TQDM_FMT = (
    "{desc:<32}{percentage:3.0f}%|{bar:30}| "
    "{n_fmt:>6}/{total_fmt:<6} [{elapsed}<{remaining}]"
)
