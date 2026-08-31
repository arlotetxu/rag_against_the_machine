import os
import fire
from indexing.create_index import Indexing
from icecream import ic

ic.configureOutput(includeContext=True)


def index(max_chunk_size: int = 2000) -> None:
    print("Desde funcion index con python fire")
    print(f"Valor max_chunk_size: {max_chunk_size}")
    Indexing(max_chunk_size).get_input_files()


if __name__ == '__main__':
    fire.Fire()