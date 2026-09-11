import os
from pathlib import Path
from tqdm import tqdm
from entities.data_model import MinimalSource
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from aux.colors import Colors
from aux.error_desc import ErrorCodes
from aux.constants import PathsAndNames
from indexing.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
from entities.data_model import IndexedChunk, RagIndex
import pickle
# from icecream import ic


class Indexer:

    def __init__(
            self, max_chunk_size: int = 2000, min_chunk_tokens: int = 10
            ) -> None:
        self.max_chunk = max_chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.files_lst: dict[str, str] = {}
        self.chunks: dict[str, IndexedChunk] = {}
        self.chunk_id = 0
        self.prefix_py = "py_"
        self.prefix = "id_"

    def get_input_files(self) -> None:
        path = PathsAndNames.corpus_path.value
        index = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                self.files_lst[f"id{index}"] = (os.path.join(root, file))
                index += 1
        print(f"Documents read: {len(self.files_lst)}")

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

    def add_imports_py_chunks(
            self, py_path: str, data_bytes: bytes, childrens: list[Node]
            ) -> None:
        children_imports = [
            children for children in childrens if children.type in [
                'import_statement', 'import_from_statement']]
        if children_imports:
            start = min(
                children.start_byte for children in children_imports)
            end = max(
                children.end_byte for children in children_imports)
            diff = end - start
            if diff:
                while diff > self.max_chunk:
                    self.chunks[f"{self.prefix_py}{self.chunk_id}"] = \
                        IndexedChunk(
                            text=data_bytes[
                                start: start + self.max_chunk].decode('utf8'),
                            metadata=MinimalSource(
                                file_path=py_path,
                                first_character_index=start,
                                last_character_index=start + self.max_chunk)
                        )
                    self.chunk_id += 1
                    start = start + self.max_chunk
                    diff -= self.max_chunk

            self.chunks[f"{self.prefix_py}{self.chunk_id}"] = IndexedChunk(
                        text=data_bytes[start: end].decode('utf8'),
                        metadata=MinimalSource(
                            file_path=py_path,
                            first_character_index=start,
                            last_character_index=end)
            )
        self.chunk_id += 1

    def chunk_py(self) -> None:

        py_docs = {id: doc_path for id, doc_path in self.files_lst.items() if
                   self.get_extension(doc_path) == '.py'}

        # Setting up tree-sitter with python grammar
        py_language = Language(tspython.language())
        parser = Parser(py_language)

        try:
            for _, py_path in tqdm(
                    py_docs.items(), desc="Chunking .py files"):
                with open(py_path, mode='r', encoding='utf8') as fd:
                    data = fd.read()
                data_bytes = data.encode('utf8')

                tree = parser.parse(data_bytes)  # Getting the file tree
                root_node = tree.root_node  # Getting the root node
                childrens = root_node.children  # Getting the childrens
                # Getting the import block unified
                self.add_imports_py_chunks(py_path, data_bytes, childrens)

                # Getting the file body
                for children in childrens:
                    if children.type in [
                            'import_statement', 'import_from_statement']:
                        continue
                    start = children.start_byte
                    end = children.end_byte
                    diff = end - start
                    while diff > self.max_chunk:
                        # Cutting the chunk in the previous /n
                        to = start + self.max_chunk
                        while data_bytes[to:to+1] != b'\n':
                            to -= 1
                            if to <= start:
                                to = start + self.max_chunk
                                break
                        self.chunks[f"{self.prefix_py}{self.chunk_id}"] = \
                            IndexedChunk(
                                text=data_bytes[start: to].decode('utf8'),
                                metadata=MinimalSource(
                                    file_path=py_path,
                                    first_character_index=start,
                                    last_character_index=to)
                                    )
                        self.chunk_id += 1
                        diff -= (to - start)
                        start = to + 1

                    self.chunks[f"{self.prefix_py}{self.chunk_id}"] = \
                        IndexedChunk(
                            text=data_bytes[start:end].decode('utf8'),
                            metadata=MinimalSource(
                                file_path=py_path,
                                first_character_index=start,
                                last_character_index=end)
                                )
                    self.chunk_id += 1
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"{Colors.YELLOW.value}[WARNING] -  "
                f"The file {py_path}{ErrorCodes.FILE_NOT_FOUND.value}"
                f"{Colors.RESET.value}") from e
        except PermissionError as e:
            raise PermissionError(
                f"{Colors.YELLOW.value}[WARNING] -  "
                f"The file {py_path}{ErrorCodes.PERMISSION.value}"
                f"{Colors.RESET.value}") from e

    def chunk_others(self) -> None:
        bin_extensions = {
            '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.ico', '.webp',
            '.pdf', '.zip', '.tar', '.gz', '.whl', '.so', '.dylib', '.dll',
        }
        generic_docs = {id: doc_path for id, doc_path
                        in self.files_lst.items()
                        if self.get_extension(doc_path) != '.py'
                        and self.get_extension(doc_path) not in bin_extensions}
        self.chunk_id = 0

        for _, path in tqdm(generic_docs.items(), desc="Chunking other files"):
            # ic(path)
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
                        IndexedChunk(text=data[start:end],
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
                    IndexedChunk(text=data[start:end],
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

    def tokenize_chunks(self) -> list[list[str]]:
        corpus_tokens: list[list[str]] = []
        discarded: list[str] = []
        tokenizer = Tokenizer()
        for id, meta in tqdm(self.chunks.items(), desc="Tokenizing..."):
            path = meta.metadata.file_path
            is_py = Path(path).suffix == '.py'
            if is_py:
                tokens = tokenizer.tokenize_code(meta.text)
            else:
                tokens = tokenizer.tokenize_other(meta.text)
            if len(tokens) < self.min_chunk_tokens:
                discarded.append(id)
                continue
            if is_py:
                tokens.extend(tokenizer.tokenize_code(path))
            else:
                tokens.extend(tokenizer.tokenize_other(path))
            corpus_tokens.append(tokens)

        for id in discarded:
            del self.chunks[id]
        return corpus_tokens

    def bm25_index(self) -> BM25Okapi:
        corpus_tokens = self.tokenize_chunks()
        bm25_index = BM25Okapi(corpus_tokens)  # type: ignore[no-untyped-call]
        return bm25_index

    def save_index(self, bm25_index: BM25Okapi) -> None:
        file_2_save = PathsAndNames.index_name.value
        path_2_save = Path(PathsAndNames.save_index_path.value)

        try:
            path_2_save.mkdir(parents=True, exist_ok=True)

            with open(path_2_save / file_2_save, mode='wb') as fd:
                pickle.dump(bm25_index, fd)
        except PermissionError as e:
            raise PermissionError(
                f"{Colors.YELLOW.value}[WARNING] -  "
                f"The file {path_2_save / file_2_save}"
                f"{ErrorCodes.PERMISSION.value}"
                f"{Colors.RESET.value}") from e

        file_2_save = PathsAndNames.chunks_json.value
        path = PathsAndNames.save_chunks.value
        try:
            chunk_list = list(self.chunks.values())
            with open(path + file_2_save, mode='w', encoding='utf') as fd:
                fd.write(RagIndex(chunks=chunk_list).model_dump_json(indent=2))
        except PermissionError as e:
            raise PermissionError(
                f"{Colors.YELLOW.value}[WARNING] -  "
                f"The file {path}"
                f"{ErrorCodes.PERMISSION.value}"
                f"{Colors.RESET.value}") from e

    def run(self) -> None:
        try:
            self.get_input_files()
            self.chunk_py()
            self.chunk_others()
            bm25_index = self.bm25_index()
            self.save_index(bm25_index)
        except (FileNotFoundError, PermissionError) as e:
            raise Exception(e)


if __name__ == '__main__':
    indexer = Indexer(1800)
    indexer.get_input_files()
    indexer.chunk_others()
