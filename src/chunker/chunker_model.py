"""Base class shared by the corpus chunkers."""
from pathlib import Path
from src.entities.data_model import IndexedChunk


class Chunk:
    """Shared state and helpers for the chunkers.

    Subclasses (``ChunkerCode`` for Python files, ``ChunkOther`` for the
    rest) split the files in ``files_lst`` and add the resulting chunks to
    ``chunks``. The indexer passes the same dictionary through each
    chunker in turn, so it accumulates the chunks of the whole corpus.
    """

    def __init__(
                self,
                files_lst: dict[str, str],
                chunks: dict[str, IndexedChunk],
                max_chunk_size: int = 2000,
                ) -> None:
        """Store the files to chunk and the dictionary to fill.

        Args:
            files_lst (dict[str, str]): Paths of the corpus files, keyed by
                file id.
            chunks (dict[str, IndexedChunk]): Chunks produced so far, keyed
                by chunk id. It is modified in place, not copied.
            max_chunk_size (int, optional): Maximum number of characters per
                chunk. Defaults to 2000.
        """
        self.max_chunk = max_chunk_size
        self.chunks: dict[str, IndexedChunk] = chunks
        self.files_lst: dict[str, str] = files_lst

    def get_extension(self, doc_path: str) -> str:
        """Return the file extension of ``doc_path``, including the dot.

        Args:
            doc_path (str): Path of the file.

        Returns:
            str: The extension (e.g. ``".py"``), or an empty string if the
                file has none.
        """
        return (Path(doc_path).suffix)

    def chunk_checker(self, id: str) -> None:
        """Print a chunk's text and location, for debugging.

        Nothing is printed if no chunk has the given id.

        Args:
            id (str): Id of the chunk to show.
        """
        for _id, chunk in self.chunks.items():
            if _id == id:
                print(f"Text:\n{chunk.text}")
                print("###" * 30)
                print(f"chunk_id: {_id}")
                print(f"File Path: {chunk.metadata.file_path}")
                print(f"Start char: {chunk.metadata.first_character_index}")
                print(f"Last char: {chunk.metadata.last_character_index}")
                print("===" * 30)
