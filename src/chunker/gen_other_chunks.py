from src.entities.data_model import IndexedChunk, MinimalSource
from src.chunker.chunker_model import Chunk
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT
from tqdm import tqdm
# from icecream import ic


class ChunkOther(Chunk):

    def __init__(
                self,
                files_lst: dict[str, str],
                chunks: dict[str, IndexedChunk],
                max_chunk_size: int = 2000,
                overlap: int = 150,
                ) -> None:

        self.chunk_id = 0
        self.prefix = "id_"
        super().__init__(files_lst, chunks, max_chunk_size)
        # The overlap must be smaller than the chunk or start never advances
        self.overlap = min(overlap, self.max_chunk // 2)

    def chunk_others(self) -> dict[str, IndexedChunk]:

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

            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"{Colors.YELLOW.value}[WARNING] -  "
                    f"The file {path}{ErrorCodes.FILE_NOT_FOUND.value}"
                    f"{Colors.RESET.value}") from e
            except PermissionError as e:
                raise PermissionError(
                    f"{Colors.YELLOW.value}[WARNING] -  "
                    f"The file {path}{ErrorCodes.PERMISSION.value}"
                    f"{Colors.RESET.value}") from e

        return self.chunks
