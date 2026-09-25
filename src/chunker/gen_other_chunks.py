"""Chunker for every non-Python text file in the corpus."""
from src.entities.data_model import IndexedChunk, MinimalSource
from src.chunker.chunker_model import Chunk
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT
from tqdm import tqdm
# from icecream import ic


class ChunkOther(Chunk):
    """Split text files (Markdown, plain text, etc.) into overlapping chunks.

    Python files are left to ``ChunkerCode``. Chunk ids have the form
    ``id_<n>``, and the stored start/end indexes are character offsets
    into the decoded text.
    """

    def __init__(
                self,
                files_lst: dict[str, str],
                chunks: dict[str, IndexedChunk],
                max_chunk_size: int = 2000,
                overlap: int = 100,
                ) -> None:
        """Initialise the chunker and its chunk-id counter.

        Args:
            files_lst (dict[str, str]): Paths of the corpus files, keyed by
                file id.
            chunks (dict[str, IndexedChunk]): Chunks produced so far, keyed
                by chunk id. New chunks are added to it in place.
            max_chunk_size (int, optional): Maximum number of characters per
                chunk. Defaults to 2000.
            overlap (int, optional): Number of characters shared by
                consecutive chunks of the same file. Capped at half of
                ``max_chunk_size``. Defaults to 100.
        """
        self.chunk_id = 0
        self.prefix = "id_"
        super().__init__(files_lst, chunks, max_chunk_size)
        # The overlap must be smaller than the chunk or start never advances
        self.overlap = min(overlap, self.max_chunk // 2)

    def chunk_others(self) -> dict[str, IndexedChunk]:
        """Chunk every text file in ``files_lst`` that is not Python code.

        Files with a binary extension (images, archives, libraries, PDFs),
        ``.py`` files and ``.DS_Store`` are ignored. Files that are not
        valid UTF-8 are skipped with a warning.

        Each chunk ends at the last line break before ``max_chunk``
        characters, as long as that line break comes after the overlap
        zone; otherwise it ends exactly at the limit. The next chunk starts
        ``overlap`` characters before the previous one ended.

        Returns:
            dict[str, IndexedChunk]: The ``chunks`` dictionary, now also
                holding the chunks of the text files.

        Raises:
            OSError: If a file does not exist or cannot be read
        """
        bin_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp',
            '.pdf', '.zip', '.tar', '.gz', '.whl', '.so', '.dylib', '.dll',
            '.py'
        }
        generic_docs = {id: doc_path for id, doc_path in self.files_lst.items()
                        if self.get_extension(doc_path) not in bin_extensions}
        self.chunk_id = 0

        for _, path in tqdm(generic_docs.items(),
                            desc="Chunking other files",
                            bar_format=TQDM_FMT):
            if path.endswith(".DS_Store"):
                continue
            try:
                with open(path, mode='rb') as fd:
                    data_bytes = fd.read()
                try:
                    data = data_bytes.decode('utf8')
                except UnicodeDecodeError:
                    print(
                        f"{Colors.YELLOW.value}[WARNING] - "
                        f"Skipping {path}: not valid UTF-8 text"
                        f"{Colors.RESET.value}")
                    continue

                file_len = len(data)
                start = 0

                while start < file_len:
                    end = min(start + self.max_chunk, file_len)
                    if end < file_len:
                        # Cutting the chunk in the previous \n, but only
                        # beyond the overlap zone so start always advances
                        cut = data.rfind('\n', start + self.overlap + 1, end)
                        if cut != -1:
                            end = cut
                    self.chunks[f"{self.prefix}{self.chunk_id}"] = \
                        IndexedChunk(text=data[start:end].strip(),
                                     metadata=MinimalSource(
                                         file_path=path,
                                         first_character_index=start,
                                         last_character_index=end))
                    self.chunk_id += 1
                    if end >= file_len:
                        break
                    start = end - self.overlap

            except OSError as e:
                raise FileNotFoundError(
                    f"{Colors.RED.value}[ERROR] -  "
                    f"The file {path}{ErrorCodes.OS_ERROR.value}"
                    f"{Colors.RESET.value}") from e

        return self.chunks
