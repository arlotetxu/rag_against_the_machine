from pathlib import Path
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.indexer.indexer import Indexer
from src.retrieval.retrieval import Retrieval
from src.evaluate.evaluate import Evaluate
from icecream import ic

ic.configureOutput(includeContext=True)

'''
# ========PENDING TASKS========

[X] - Modify the evaluate method in retrieval.py (refactor?)
[X] - Improve ranking performance
[] - Modify error messages under constants.py/PathsAndNames
[] - Last checks to open file exceptions

'''


def index(max_chunk_size: int = 2000) -> None:
    """
    Create the index with file chunks
    Uses MinimalSource
    """
    if max_chunk_size > 2000:
        max_chunk_size = 800
        print(f"{Colors.YELLOW.value}[WARNING] - "
              f"{ErrorCodes.MAX_SIZE_CHUNK.value}"
              f"{Colors.RESET.value}")
    indexer = Indexer(max_chunk_size)
    indexer.run()


def search(query: str, k: int) -> None:
    """
    One single query
    Uses StudentSearchResults to generate JSON
    """
    ic("From search")
    retrieval = Retrieval()
    retrieval.get_query_chunks(query=query, k=k, print_=True)


def search_dataset(
        dataset_path: str,
        k: int,
        save_directory: str = "data/output/search_results") -> None:
    """
    Batch queries
    Uses StudentSearchResults to generate JSON
    """
    dataset_file = Path(dataset_path)
    save_folder = Path(save_directory)
    save_folder.mkdir(parents=True, exist_ok=True)
    output_path = save_folder / dataset_file.name

    ic("From search_dataset")
    retrieval = Retrieval()
    retrieval.get_batch_query_chunks(dataset_path, k, str(output_path))


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


def evaluate(student_search_results_path: str,
             dataset_path: str,
             k: int) -> None:
    """
    The test function to check the results
    """
    ic("From evaluate")
    evaluate = Evaluate()
    evaluate.get_recall(student_search_results_path, dataset_path, k)
