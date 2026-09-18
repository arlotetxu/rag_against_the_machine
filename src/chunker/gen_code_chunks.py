from src.entities.data_model import IndexedChunk, MinimalSource
from src.chunker.chunker_model import Chunk
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
from tqdm import tqdm


class ChunkerCode(Chunk):

    def __init__(
            self,
            files_lst: dict[str, str],
            chunks: dict[str, IndexedChunk],
            max_chunk_size: int = 2000,
            ) -> None:

        self.chunk_id = 0
        self.prefix_py = "py_"
        super().__init__(files_lst, chunks, max_chunk_size)

    def get_children(
            self,
            py_path: str,
            parser: Parser) -> tuple[bytes, list[Node]]:

        try:
            with open(py_path, mode='r', encoding='utf8') as fd:
                data = fd.read()
            data_b = data.encode('utf8')

            tree = parser.parse(data_b)  # Getting the file tree
            root_node = tree.root_node  # Getting the root node
            childrens = root_node.children  # Getting the childrens

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

        return data_b, childrens

    def add_imports_py_chunks(
            self, py_path: str, data_b: bytes, childrens: list[Node]
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
                    gen_text = data_b[
                        start: start + self.max_chunk].decode('utf8').strip()
                    self.chunks[f"{self.prefix_py}{self.chunk_id}"] = \
                        IndexedChunk(
                            text=gen_text,
                            metadata=MinimalSource(
                                file_path=py_path,
                                first_character_index=start,
                                last_character_index=start + self.max_chunk)
                        )
                    self.chunk_id += 1
                    start = start + self.max_chunk
                    diff -= self.max_chunk

            gen_text = data_b[start: end].decode('utf8').strip()
            self.chunks[f"{self.prefix_py}{self.chunk_id}"] = IndexedChunk(
                        text=gen_text,
                        metadata=MinimalSource(
                            file_path=py_path,
                            first_character_index=start,
                            last_character_index=end)
            )
        self.chunk_id += 1

    def chunk_py(self) -> dict[str, IndexedChunk]:

        py_docs = {id: doc_path for id, doc_path in self.files_lst.items() if
                   self.get_extension(doc_path) == '.py'}

        # Setting up tree-sitter with python grammar
        py_language = Language(tspython.language())
        parser = Parser(py_language)

        try:
            for _, py_path in tqdm(py_docs.items(),
                                   desc="Chunking .py files",
                                   bar_format=TQDM_FMT):

                data_b, childrens = self.get_children(py_path, parser)
                # Getting the import block unified
                self.add_imports_py_chunks(py_path, data_b, childrens)

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
                        while data_b[to:to+1] != b'\n':
                            to -= 1
                            if to <= start:
                                to = start + self.max_chunk
                                break
                        gen_text = data_b[start: to].decode('utf8').strip()
                        self.chunks[f"{self.prefix_py}{self.chunk_id}"] = \
                            IndexedChunk(
                                text=gen_text,
                                metadata=MinimalSource(
                                    file_path=py_path,
                                    first_character_index=start,
                                    last_character_index=to)
                                    )
                        self.chunk_id += 1
                        diff -= (to - start)
                        start = to + 1

                    gen_text = data_b[start:end].decode('utf8').strip()
                    self.chunks[f"{self.prefix_py}{self.chunk_id}"] = \
                        IndexedChunk(
                            text=gen_text,
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

        return self.chunks
