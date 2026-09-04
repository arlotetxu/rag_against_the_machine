import os
import re
from pathlib import Path
from pydantic import BaseModel
from tqdm import tqdm
from entities.minimal_source import MinimalSource
import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node
from aux.colors import Colors
from aux.error_desc import ErrorCodes
from icecream import ic


class IndexedChunk(BaseModel):
    text: str
    metadata: MinimalSource

class Indexer:

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.files_lst: dict[str, str] = {}
        self.chunks: dict[str, IndexedChunk] = {}
        self.chunk_id = 0

    def get_input_files(self) -> None:
        path = 'data/raw'
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
                print(f"chunk_id: {_id}")
                print(f"Text: {chunk.text}")
                print(f"File Path: {chunk.metadata.file_path}")
                print(f"Start char: {chunk.metadata.first_character_index}")
                print(f"Last char: {chunk.metadata.last_character_index}")
                print("===" * 30)

    def norm_code(self, text: str) -> str:
        # Splitting camelCase/PascalCase
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        # Splitting snake_case
        text = re.sub(r'([_\-])', ' ', text)
        return text

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
                while diff > self.max_chunk_size:
                    self.chunks[f"py_{self.chunk_id}"] = IndexedChunk(
                        text=data_bytes[
                            start: start + self.max_chunk_size].decode('utf8'),
                        metadata= MinimalSource(
                            file_path=py_path,
                            first_character_index=start,
                            last_character_index=start + self.max_chunk_size)
                    )
                    self.chunk_id += 1
                    start = start + self.max_chunk_size
                    diff -= self.max_chunk_size

            self.chunks[f"py_{self.chunk_id}"] = IndexedChunk(
                        text=data_bytes[start: end].decode('utf8'),
                        metadata= MinimalSource(
                            file_path=py_path,
                            first_character_index=start,
                            last_character_index=end)
            )
        self.chunk_id += 1

    def chunk_py(self) -> None:
        """
        1- Crear chunks y guardar en self.chunks
        2- Generar estadisticas del corpus con BM25

            children_types: [
            'import_statement', 'import_from_statement', 'class_definition',
            'function_definition', 'if_statement']
        """
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
                # Getting the file tree
                tree = parser.parse(data_bytes)
                # Getting the root node
                root_node = tree.root_node
                # Getting the childrens from the root node
                childrens = root_node.children

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
                    if diff:
                        while diff > self.max_chunk_size:
                            self.chunks[f"py_{self.chunk_id}"] = IndexedChunk(
                                text=data_bytes[
                                    start: start + self.max_chunk_size].decode(
                                        'utf8'),
                                metadata=MinimalSource(
                                    file_path=py_path,
                                    first_character_index=start,
                                    last_character_index=start +
                                    self.max_chunk_size
                                )
                            )
                            self.chunk_id += 1
                            start = start + self.max_chunk_size
                            diff -= self.max_chunk_size
                    self.chunks[f"py_{self.chunk_id}"] = IndexedChunk(
                        text=data_bytes[start:end].decode('utf8'),
                        metadata=MinimalSource(
                            file_path=py_path,
                            first_character_index=start,
                            last_character_index=end
                            )
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


    # def chunk_md(self):
    #     md_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) == '.md'}
    #     ic(md_docs)

    # def chunk_json(self):
    #     json_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) == '.json'}
    #     # ic(json_docs)

    def chunk_others(self):
        generic_docs = {id: doc_path for id, doc_path in self.files_lst.items()
                        if self.get_extension(doc_path) != '.py'}
        print("From chunk_others")
        # ic(generic_docs)

if __name__ == '__main__':
    indexer = Indexer(1800)
    indexer.get_input_files()
    indexer.chunk_others()

