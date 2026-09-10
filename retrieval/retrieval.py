import pickle
import pydantic
from aux.constants import PathsAndNames
from aux.colors import Colors
from aux.error_desc import ErrorCodes
from entities.data_model import (IndexedChunk,
                                 RagIndex,
                                 MinimalSource,
                                 MinimalSearchResults,
                                 StudentSearchResults)
from indexing.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
import numpy as np
import uuid


from icecream import ic


class Retrieval:
    def __init__(self) -> None:
        self.tokenizer: Tokenizer = Tokenizer()
        self.bm25_index: BM25Okapi = self.get_bm25_index()
        self.chunks: RagIndex = self.get_chunks()

    def get_bm25_index(self) -> BM25Okapi:
        index_parents = PathsAndNames.save_index_path.value
        index_name = PathsAndNames.index_name.value
        path = index_parents + '/' + index_name

        try:
            with open(path, mode='rb') as fd:
                bm25_index = pickle.load(fd)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Colors.RED.value}[ERROR] - "
                f"The index file '{path}'{ErrorCodes.FILE_NOT_FOUND.value}"
                )
        except PermissionError:
            raise PermissionError(
                f"{Colors.RED.value}[ERROR] - "
                f"The index file '{path}'{ErrorCodes.PERMISSION.value}"
                )

        return bm25_index

    def get_chunks(self) -> RagIndex:
        chunks_parents = PathsAndNames.save_chunks.value
        file_name = PathsAndNames.chunks_json.value
        path = chunks_parents + file_name

        try:
            with open(path, mode='r') as fd:
                chunks = RagIndex.model_validate_json(fd.read())
        except FileNotFoundError:
            raise FileNotFoundError(
                f"{Colors.RED.value}[ERROR] - "
                f"The chunks file '{path}'{ErrorCodes.FILE_NOT_FOUND.value}"
                )
        except PermissionError:
            raise PermissionError(
                f"{Colors.RED.value}[ERROR] - "
                f"The chunks file '{path}'{ErrorCodes.PERMISSION.value}"
                )
        # ic(chunks.chunks[0].metadata)
        return chunks

    def tokenize_query(self, query: str) -> list[str]:
        query_tokens = set()
        query_tokens_code = self.tokenizer.tokenize_code(query)
        query_tokens_other = self.tokenizer.tokenize_other(query)
        query_tokens = set(query_tokens_other)
        for token in query_tokens_code:
            query_tokens.add(token)
        ic(list(query_tokens))
        return list(query_tokens)

    def get_indexes(self, query:str, k: int) -> list[int]:
        query_tokens = self.tokenize_query(query)
        scores = self.bm25_index.get_scores(query_tokens)
        # Returns the indices that would sort an array:
        scores = np.argsort(scores, descending=True)
        scores = scores.tolist()

        return scores[:k]

    def get_query_chunks(self, query: str, k:int) -> None:
        chunk_indexes = self.get_indexes(query, k)
        minimal_source_lst = [
            self.chunks.chunks[index].metadata for index in chunk_indexes
            ]

        minimal_search_result = MinimalSearchResults(
            question_id=str(uuid.uuid4()),
            question=query,
            retrieved_sources=minimal_source_lst
        )
        result = StudentSearchResults(
            search_results=[minimal_search_result],
            k=k
        )
        ic(result)


