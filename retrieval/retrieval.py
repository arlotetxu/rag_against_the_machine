import pickle
import pydantic
from aux.constants import PathsAndNames
from aux.colors import Colors
from aux.error_desc import ErrorCodes
from entities.data_model import IndexedChunk, RagIndex, MinimalSource


from icecream import ic


class Retrieval:

    def get_bm25_index(self) -> None:
        index_parents = PathsAndNames.save_index_path.value
        index_name = PathsAndNames.index_name.value
        path = index_parents + '/' + index_name

        try:
            with open(path, mode='rb') as fd:
                self.bm25_index = pickle.load(fd)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Colors.RED.value}[ERROR] - "
                f"The index file '{path}'{ErrorCodes.FILE_NOT_FOUND.value}"
                )
        except PermissionError:
            raise PermissionError(
                f"{Colors.RED.value}[ERROR] - "
                f"The index file '{path}'{ErrorCodes.PERMISSION.value}"
                )            

    def get_chunks(self) -> None:
        chunks_parents = PathsAndNames.save_chunks.value
        file_name = PathsAndNames.chunks_json.value
        path = chunks_parents + file_name

        with open(path, mode='r') as fd:
            self.chunks = RagIndex.model_validate_json(fd)

        ic(self.chunks)