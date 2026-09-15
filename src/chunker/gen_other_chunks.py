from src.entities.data_model import IndexedChunk, MinimalSource
from src.chunker.chunker_model import Chunk
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from tqdm import tqdm


class ChunkOther(Chunk):

    def __init__(
                self,
                files_lst: dict[str, str],
                chunks: dict[str, IndexedChunk],
                max_chunk_size: int = 2000,
                ) -> None:

        self.chunk_id = 0
        self.prefix = "id_"
        super().__init__(files_lst, chunks, max_chunk_size)

    def chunk_others(self) -> dict[str, IndexedChunk]:

        bin_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp',
            '.pdf', '.zip', '.tar', '.gz', '.whl', '.so', '.dylib', '.dll',
            '.py'
        }
        generic_docs = {id: doc_path for id, doc_path in self.files_lst.items()
                        if self.get_extension(doc_path) not in bin_extensions}
        self.chunk_id = 0

        for _, path in tqdm(generic_docs.items(), desc="Chunking other files"):
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
                end = file_len if file_len <= self.max_chunk else \
                    start + self.max_chunk
                diff = end - start

                while diff >= self.max_chunk:
                    # Cutting the chunk in the previous /n
                    end_prov = end
                    while data[end_prov:end_prov + 1] != '\n':
                        end_prov -= 1
                        if end_prov <= start:
                            end = start + self.max_chunk
                            break
                        end = end_prov
                    self.chunks[f"{self.prefix}{self.chunk_id}"] = \
                        IndexedChunk(text=data[start:end].strip(),
                                     metadata=MinimalSource(
                                         file_path=path,
                                         first_character_index=start,
                                         last_character_index=end))
                    self.chunk_id += 1
                    start = end + 1
                    to = len(data[start:])
                    end = start + self.max_chunk if to > self.max_chunk else \
                        start + to
                    diff = end - start

                self.chunks[f"{self.prefix}{self.chunk_id}"] = \
                    IndexedChunk(text=data[start:end].strip(),
                                 metadata=MinimalSource(
                                     file_path=path,
                                     first_character_index=start,
                                     last_character_index=end))
                self.chunk_id += 1

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
