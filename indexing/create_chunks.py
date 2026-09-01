import os
import re
from pathlib import Path
from entities.minimal_source import IndexedChunk
from icecream import ic


class Indexing:

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
        # ic(self.files_lst.get("id999"))
        
    def get_extension(self, doc_path: str) -> str:
        # ic(Path(doc_path).suffix)
        return (Path(doc_path).suffix)

    def norm_code_text(self, text: str) -> str:
        # Fixing camelCase/PascalCase
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        # Fixing snake_case
        text = re.sub(r'([_\-])', ' ', text)
        return text
    

    def chunk_py(self):
        """
        Tokenización de identificadores: 
        divide snake_case y camelCase en palabras sueltas 
        (get_user_by_id → get, user, by, id). Así una
        pregunta como "¿cómo obtengo un usuario por id?" puede hacer match con
        el término user e id aunque nunca aparezcan como palabras sueltas en
        el código.
        Incluir docstrings y comentarios como texto "natural" dentro del
        chunk — son la parte del código que más se parece al lenguaje de las
        preguntas, y ayudan mucho al matching léxico.
        Chunking por unidad semántica (función/clase completa) en vez de corte
        arbitrario por caracteres — esto es más una decisión de la fase de
        chunking (paso 2) que de tokenización, pero impacta directamente en
        cuánta señal útil tiene cada chunk para el matching.
        Considera si conviene filtrar palabras muy comunes en código (self,
        def, import, return...) para que no distorsionen el IDF — aunque BM25
        ya las penaliza algo por ser muy frecuentes, un stopword list
        específico de código puede ayudar.

        Modulo ast de python para sacar funciones
        """
        py_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) == '.py'}
        # ic(py_docs)
        # Normalizing
        for doc_path in py_docs:
            try:
                with open(doc_path, mode='+r') as fd:
                    lines = fd.readlines()
            except (PermissionError):
                raise PermissionError(
                    f"The file {doc_path} couldn't be open."
                    "Plese, check the permissions and try again"
                )

            
    def chunk_md(self):
        md_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) == '.md'}
        # ic(md_docs)

    def chunk_json(self):
        json_docs = {id: doc_path for id, doc_path in self.files_lst.items() if self.get_extension(doc_path) == '.json'}
        # ic(json_docs)

    def chunk_others(self):
        pass