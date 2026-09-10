# import os
import sys
import fire
from aux.colors import Colors
from aux.error_desc import ErrorCodes
from indexing.indexer import Indexer
from retrieval.retrieval import Retrieval
import traceback
from icecream import ic

ic.configureOutput(includeContext=True)


def index(max_chunk_size: int = 2000) -> None:
    """
    Create the index with file chunks
    Uses MinimalSource
    """
    if max_chunk_size > 2000:
        max_chunk_size = 2000
        print(f"{Colors.YELLOW.value}[WARNING] - "
              f"{ErrorCodes.MAX_SIZE_CHUNK.value}"
              f"{Colors.RESET.value}")
    try:
        indexer = Indexer(max_chunk_size)
        indexer.run()
    except Exception as e:
        print(
            f"{Colors.RED.value}[ERROR] - "
            f"Error during the process...\n"
            f"Details: {e} (occurred in "
            f"{traceback.extract_tb(sys.exc_info()[2])[-1].filename} at line "
            f"{traceback.extract_tb(sys.exc_info()[2])[-1].lineno})"
            f"{Colors.RESET.value}\n"
        )


def search(query: str, k: int) -> None:
    """
    One single query
    Uses StudentSearchResults to generate JSON
    """
    retrieval = Retrieval()
    # Retrieval().get_bm25_index()
    # Retrieval().get_chunks()
    retrieval.get_query_chunks(query=query, k=k)


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    """
    Batch queries
    Uses StudentSearchResults to generate JSON
    """
    ic(dataset_path)
    ic(k)
    ic(save_directory)
    pass


def answer() -> None:
    """
    Generate a single answer to a single query using LLM Qwen3-0.6B
    Uses StudentSearchResultsAndAnswer to generate JSON
    """
    ic("From answer")
    pass


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> None:
    """
    Generates answers to a batch of queries using LLM Qwen3-0.6B
    Uses StudentSearchResultsAndAnswer to generate JSON
    """
    ic("From answer_dataset")
    pass


def evaluate() -> None:
    """
    The test function to check the results
    """
    ic("From evaluate")
    pass


if __name__ == '__main__':
    fire.Fire()  # type: ignore[no-untyped-call]
