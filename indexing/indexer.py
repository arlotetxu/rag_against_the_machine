import os
import re
from pathlib import Path
from pydantic import BaseModel
from entities.minimal_source import MinimalSource
import tree_sitter_python as tspython
from tree_sitter import Language, Parser
from icecream import ic


class IndexedChunk(BaseModel):
    text: str
    metadata: MinimalSource

class Indexer:

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size
        self.files_lst = {}
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
        # ic(self.files_lst)
        # ic(self.files_lst.get("id999", None))

    def get_extension(self, doc_path: str) -> str:
        # ic(Path(doc_path).suffix)
        return (Path(doc_path).suffix)

    def norm_code(self, text: str) -> str:
        # Fixing camelCase/PascalCase
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        # Fixing snake_case
        text = re.sub(r'([_\-])', ' ', text)
        return text


    def chunk_py(self):
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
            for _, py_path in py_docs.items():
                with open(py_path, mode='r', encoding='utf8') as fd:
                    data = fd.read()
                data_bytes = data.encode('utf8')
                # Getting the file tree
                tree = parser.parse(data_bytes)
                # Getting the root node
                root_node = tree.root_node
                # Getting the childrens from the root node
                childrens = root_node.children
                for children in childrens:
                    start = children.start_byte
                    end = children.end_byte
                    print(data[start:end])
                    print("===" * 30)




        except (FileNotFoundError, PermissionError) as e:
            print(e)

        print(self.chunks)

    # def chunk_md(self):
    #     md_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) == '.md'}
    #     ic(md_docs)

    # def chunk_json(self):
    #     json_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) == '.json'}
    #     # ic(json_docs)

    def chunk_others(self):
        generic_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) != '.py'}
        ic("From chunk_others")
        # ic(generic_docs)

def test_function():
    pass

if __name__ == '__main__':
    indexer = Indexer(1800)
    indexer.get_input_files()
    indexer.chunk_others()

