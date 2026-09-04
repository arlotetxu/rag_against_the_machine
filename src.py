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
    # print("Desde funcion index con python fire")
    # print(f"Valor max_chunk_size: {max_chunk_size}")
    if max_chunk_size > 2000:
        max_chunk_size = 2000
        print(f"{Colors.YELLOW.value}[WARNING] - "
              f"{ErrorCodes.MAX_SIZE_CHUNK.value}"
              f"{Colors.RESET.value}")
    try:
        indexer = Indexer(max_chunk_size)
        indexer.get_input_files()
        indexer.chunk_py()
        indexer.chunk_others()
        # indexer.chunk_checker("py_1")
    except Exception as e:
        print(
            f"{Colors.RED.value}[ERROR] - "
            f"Error during game generation process...\n"
            f"Details: {e} (occurred in "
            f"{traceback.extract_tb(sys.exc_info()[2])[-1].filename} at line "
            f"{traceback.extract_tb(sys.exc_info()[2])[-1].lineno})"
            f"{Colors.RESET.value}\n"
        )





if __name__ == '__main__':
    fire.Fire()
