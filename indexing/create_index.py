import os
from icecream import ic


class Indexing:

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.files_lst = {}

    def get_input_files(self) -> None:
        path = 'data/raw'
        index = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                self.files_lst[f"id{index}"] = (os.path.join(root, file))
                index += 1
        # ic(self.files_lst)
        # ic(self.files_lst.get("id999"))
        

    def chunk_py(self):
        pass

    def chunk_json_txt(self):
        pass