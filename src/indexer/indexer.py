import os
from pathlib import Path
from tqdm import tqdm
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import PathsAndNames
from src.indexer.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
from src.entities.data_model import IndexedChunk, RagIndex
import pickle
from pydantic import ValidationError
from src.chunker.gen_code_chunks import ChunkerCode
from src.chunker.gen_other_chunks import ChunkOther

# from icecream import ic


class Indexer:

    def __init__(
            self, max_chunk_size: int = 2000, min_chunk_tokens: int = 15
            ) -> None:
        self.max_chunk = max_chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.files_lst: dict[str, str] = {}
        self.chunks: dict[str, IndexedChunk] = {}

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
            tokens = tokenizer.remove_stopwords(tokens)
            tokens = tokenizer.stem(tokens)
            corpus_tokens.append(tokens)

        for id in discarded:
            del self.chunks[id]
        return corpus_tokens

    def bm25_index(self) -> BM25Okapi:
        corpus_tokens = self.tokenize_chunks()
        # for k1 in (0.9, 1.2, 1.5, 2.0):
        #     for b in (0.3, 0.5, 0.75, 0.9):
        #         bm25_index = BM25Okapi(
        #           corpus_tokens, k1=k1, b=b)  # type: ignore[no-untyped-call]
        #         self.save_index_chunks(bm25_index)
        #         ic(k1, b)
        #         Retrieval().get_recall(
        #             "data/datasets/AnsweredQuestions/dataset_code_public.json",
        #             5)
        bm25_index = BM25Okapi(
            corpus_tokens, k1=1.5, b=0.3)  # type: ignore[no-untyped-call]
        return bm25_index

    def save_index_chunks(self, bm25_index: BM25Okapi) -> None:
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
        except ValidationError as e:
            raise ValueError(e)
        print(f"{Colors.GREEN.value}"
              f"Corpus ingestion complete! "
              f"Indexed {len(self.chunks)} chunks under {path}"
              f"{Colors.RESET.value}")

    def run(self) -> None:
        try:
            self.get_input_files()
            self.chunks = ChunkerCode(
                files_lst=self.files_lst,
                chunks=self.chunks,
                max_chunk_size=self.max_chunk).chunk_py()
            self.chunks = ChunkOther(
                files_lst=self.files_lst,
                chunks=self.chunks,
                max_chunk_size=self.max_chunk
            ).chunk_others()
            bm25_index = self.bm25_index()
            self.save_index_chunks(bm25_index)
        except (FileNotFoundError, PermissionError) as e:
            raise Exception(e)
