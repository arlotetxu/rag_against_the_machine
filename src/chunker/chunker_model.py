from pathlib import Path
from src.entities.data_model import IndexedChunk


class Chunk:

    def __init__(
                self,
                files_lst: dict[str, str],
                chunks: dict[str, IndexedChunk],
                max_chunk_size: int = 2000,
                ) -> None:
        self.max_chunk = max_chunk_size
        self.chunks: dict[str, IndexedChunk] = chunks
        self.files_lst: dict[str, str] = files_lst

    def get_extension(self, doc_path: str) -> str:
        return (Path(doc_path).suffix)

    def chunk_checker(self, id: str) -> None:
        for _id, chunk in self.chunks.items():
            if _id == id:
                print(f"Text:\n{chunk.text}")
                print("###" * 30)
                print(f"chunk_id: {_id}")
                print(f"File Path: {chunk.metadata.file_path}")
                print(f"Start char: {chunk.metadata.first_character_index}")
                print(f"Last char: {chunk.metadata.last_character_index}")
                print("===" * 30)
