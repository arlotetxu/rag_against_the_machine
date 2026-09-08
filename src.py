# import os
import sys
import fire
from aux.colors import Colors
from aux.error_desc import ErrorCodes
from indexing.indexer import Indexer
import traceback
from icecream import ic

ic.configureOutput(includeContext=True)


def index(max_chunk_size: int = 2000) -> None:
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
    print(query)
    print(k)


def search_dataset(dataset_path: str, k: int, save_directory: str) -> None:
    pass


def answer() -> None:
    pass


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> None:
    pass


def evaluate() -> None:
    pass


if __name__ == '__main__':
    fire.Fire()  # type: ignore[no-untyped-call]
