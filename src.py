import os
import fire
from indexing.create_chunks import Indexing
from icecream import ic

ic.configureOutput(includeContext=True)


def index(max_chunk_size: int = 2000) -> None:
    print("Desde funcion index con python fire")
    print(f"Valor max_chunk_size: {max_chunk_size}")
    index = Indexing(max_chunk_size)
    index.get_input_files()
    index.chunk_py()
    index.chunk_md()
    index.chunk_json()
    


if __name__ == '__main__':
    fire.Fire()