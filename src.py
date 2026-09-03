import os
import fire
from aux.colors import Colors
from indexing.indexer import Indexer
from icecream import ic

ic.configureOutput(includeContext=True)


def index(max_chunk_size: int = 2000) -> None:
    print("Desde funcion index con python fire")
    print(f"Valor max_chunk_size: {max_chunk_size}")
    if max_chunk_size > 2000:
        max_chunk_size = 2000
        print(f"{Colors.YELLOW.value}[WARNING] - "
              "max_chunk_size needs to be less than 2001. "
              "Applying default max value: 2000."
              f"{Colors.RESET.value}")
    indexer = Indexer(max_chunk_size)
    indexer.get_input_files()
    indexer.chunk_py()
    indexer.chunk_others()



if __name__ == '__main__':
    fire.Fire()
