"""Chunker for Python source files, based on the tree-sitter syntax tree."""
from src.entities.data_model import IndexedChunk, MinimalSource
from src.chunker.chunker_model import Chunk
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT
from tree_sitter import Language, Parser, Node
import tree_sitter_python as tspython
from tqdm import tqdm


class ChunkerCode(Chunk):
    """Split ``.py`` files into chunks along top-level syntax nodes.

    Each file gets one chunk for its import block plus one chunk per
    remaining top-level node (function, class, statement...). Nodes longer
    than ``max_chunk`` are cut into several chunks. Chunk ids have the form
    ``py_<n>``.

    The ``first_character_index``/``last_character_index`` stored in each
    chunk are byte offsets into the UTF-8 encoded file, so they only match
    character offsets for ASCII-only files.
    """

    def __init__(
            self,
            files_lst: dict[str, str],
            chunks: dict[str, IndexedChunk],
            max_chunk_size: int = 2000,
            ) -> None:
        """Initialise the chunker and its chunk-id counter.

        Args:
            files_lst (dict[str, str]): Paths of the corpus files, keyed by
                file id. Only the ``.py`` ones are chunked.
            chunks (dict[str, IndexedChunk]): Chunks produced so far, keyed
                by chunk id. New chunks are added to it in place.
            max_chunk_size (int, optional): Maximum chunk length in bytes.
                Defaults to 2000.
        """
        self.chunk_id = 0
        self.prefix_py = "py_"
        super().__init__(files_lst, chunks, max_chunk_size)

    def get_children(
            self,
            py_path: str,
            parser: Parser) -> tuple[bytes, list[Node]]:
        """Read a Python file and parse it into its top-level nodes.

        Args:
            py_path (str): Path of the Python file.
            parser (Parser): tree-sitter parser set up with the Python
                grammar.

        Returns:
            tuple[bytes, list[Node]]: The file content encoded as UTF-8,
                and the direct children of the syntax tree's root node.

        Raises:
            FileNotFoundError: If the file does not exist.
            PermissionError: If the file cannot be read.
        """
        try:
            with open(py_path, mode='r', encoding='utf8') as fd:
                data = fd.read()
            data_b = data.encode('utf8')

            tree = parser.parse(data_b)  # Getting the file tree
            root_node = tree.root_node  # Getting the root node
            childrens = root_node.children  # Getting the childrens

        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] -  "
                f"The file {py_path}{ErrorCodes.OS_ERROR.value}"
                f"{Colors.RESET.value}") from e

        return data_b, childrens

    def add_imports_py_chunks(
            self, py_path: str, data_b: bytes, childrens: list[Node]
            ) -> None:
        """Add the file's import statements to ``chunks`` as one block.

        The block spans from the first top-level import to the last one, so
        any code between them is included too. Blocks longer than
        ``max_chunk`` are cut into fixed-size pieces, not on line breaks.
        Nothing is added if the file has no top-level imports.

        Args:
            py_path (str): Path of the Python file, stored in the chunk
                metadata.
            data_b (bytes): UTF-8 encoded content of the file.
            childrens (list[Node]): Top-level nodes of the file's syntax
                tree.
        """
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
        """Chunk every ``.py`` file in ``files_lst``.

        Top-level nodes longer than ``max_chunk`` are split at the last line
        break before the limit, or exactly at the limit if the piece has no
        line break.

        Returns:
            dict[str, IndexedChunk]: The ``chunks`` dictionary, now also
                holding the chunks of the Python files.

        Raises:
            OSError: If a Python file does not exist or cannot be read.
        """
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
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] -  "
                f"The file {py_path}{ErrorCodes.OS_ERROR.value}"
                f"{Colors.RESET.value}") from e

        return self.chunks
