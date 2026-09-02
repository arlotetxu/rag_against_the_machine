import os
import fire
from indexing.indexer import Indexer
from icecream import ic

ic.configureOutput(includeContext=True)


def index(max_chunk_size: int = 2000) -> None:
    print("Desde funcion index con python fire")
    print(f"Valor max_chunk_size: {max_chunk_size}")
    indexer = Indexer(max_chunk_size)
    indexer.get_input_files()
    indexer.chunk_py()
    indexer.chunk_others()
    


if __name__ == '__main__':
    fire.Fire()