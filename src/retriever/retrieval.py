import pickle
import pydantic
from src.aux.constants import PathsAndNames
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT, MIN_RATIO, BOOST
from src.entities.data_model import (
    RagIndex,
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    RagDataset)
from src.indexer.tokenizer import Tokenizer
from src.indexer.embeddings import Embeddings
from src.cacher.cacher import CacheHandler
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import tqdm
from typing import Any
import os
import torch
from pathlib import Path

# from icecream import ic


class Retrieval:
    def __init__(self, bonus: bool = False) -> None:

        self.tokenizer: Tokenizer = Tokenizer()
        self.bm25_index: BM25Okapi = self.get_bm25_index()
        self.chunks: RagIndex = self.get_chunks()
        self.booster = self.calculate_boosters()
        self.bonus = bonus
        self.embeddings: Embeddings | None = None
        if self.bonus:
            self.cacher = CacheHandler(Path(os.path.join(
                PathsAndNames.cache_path.value,
                PathsAndNames.cache_sources_name.value)))
            self.cache = self.cacher.load_cache()
            # self.embeddings = Embeddings()
            matrix_path = os.path.join(
                PathsAndNames.save_index_path.value,
                PathsAndNames.embeddings_name.value
            )
            if not os.path.exists(matrix_path):
                raise FileNotFoundError(
                    f"{Colors.RED.value}[ERROR] - "
                    f"The embeddings file '{matrix_path}'"
                    f"{ErrorCodes.OS_ERROR.value}"
                    f"{Colors.RESET.value}"
                )
            self.matrix = torch.from_numpy(np.load(matrix_path))

    def get_bm25_index(self) -> Any:

        index_parents = PathsAndNames.save_index_path.value
        index_name = PathsAndNames.index_name.value
        path = index_parents + '/' + index_name

        try:
            with open(path, mode='rb') as fd:
                bm25_index = pickle.load(fd)
        except OSError:
            raise FileNotFoundError(
                f"{Colors.RED.value}[ERROR] - "
                f"The index file '{path}'{ErrorCodes.OS_ERROR.value}"
                )

        return bm25_index

    def get_chunks(self) -> RagIndex:

        chunks_parents = PathsAndNames.save_chunks.value
        file_name = PathsAndNames.chunks_json.value
        path = chunks_parents + file_name

        try:
            with open(path, mode='r') as fd:
                ragindex_chunks = RagIndex.model_validate_json(fd.read())
        except OSError:
            raise FileNotFoundError(
                f"{Colors.RED.value}[ERROR] - "
                f"The chunks file '{path}'{ErrorCodes.OS_ERROR.value}"
                )
        except pydantic.ValidationError as e:
            raise ValueError(e)

        return ragindex_chunks

    def tokenize_query(self, query: str) -> list[str]:

        query_tokens = set()
        query_tokens_code = self.tokenizer.tokenize_code(query)
        query_tokens_other = self.tokenizer.tokenize_other(query)
        query_tokens = set(query_tokens_other)
        for token in query_tokens_code:
            query_tokens.add(token)
        query_tokens_lst = self.tokenizer.remove_stopwords(list(query_tokens))
        query_tokens_lst = self.tokenizer.stem(query_tokens_lst)
        return query_tokens_lst

    def calculate_boosters(self) -> np.ndarray:
        py_chunks = sum(1
                        for chunk in self.chunks.chunks
                        if chunk.metadata.file_path.endswith('.py')
                        )
        doc_chunks = len(self.chunks.chunks) - py_chunks
        total_chunks = len(self.chunks.chunks)
        if (py_chunks / total_chunks) < MIN_RATIO:
            calc_booster = np.array([
                BOOST if chunk.metadata.file_path.endswith('.py')
                else 1.0 for chunk in self.chunks.chunks])
        elif (doc_chunks / total_chunks) < MIN_RATIO:
            docs_extension = ('.md', '.txt', '.rst')
            calc_booster = np.array([
                BOOST if chunk.metadata.file_path.endswith(docs_extension)
                else 1.0 for chunk in self.chunks.chunks])
        else:
            calc_booster = np.array([1.0 for chunk in self.chunks.chunks])
        return calc_booster

    def get_query_scores(self, query: str, k: int) -> Any:

        def _ranks(scores: np.ndarray) -> np.ndarray:
            """Return the 1-based rank of each position (1 = best score)."""
            order = np.argsort(-scores)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, len(scores) + 1)
            return ranks

        query_tokens = self.tokenize_query(query)
        if self.bonus:
            if self.embeddings is not None:
                query_encoded = self.embeddings.encode([query])
            bm25_scores = self.bm25_index.get_scores(
                query_tokens) * self.booster   # type: ignore[no-untyped-call]
            cos_scores = (self.matrix @ query_encoded.numpy().T).ravel()
            # Both scores together using Reciprocal Rank Fusion (RRF)
            rrf_k = 60
            # w_bm25, w_cos = 1.0, 0.5
            fused = 1 / (rrf_k + _ranks(bm25_scores)) + \
                1 / (rrf_k + _ranks(cos_scores))
            scores = np.argsort(-fused)
            scores = scores.tolist()
            return scores[:k]
        else:
            scores = self.bm25_index.get_scores(
                query_tokens)  # type: ignore[no-untyped-call]
            scores = scores * self.booster
            # Returns the indices that would sort an array:
            scores = np.argsort(scores, descending=True)
            scores = scores.tolist()

            return scores[:k]

    def _retrieve(self, query: str, k: int) -> list[MinimalSource]:
        """Run the real search, without cache."""
        chunk_indexes = self.get_query_scores(query, k)

        return [
            self.chunks.chunks[index].metadata for index in chunk_indexes]

    def _retrieve_with_cache(
            self,
            query: str,
            k: int,
    ) -> tuple[list[MinimalSource], bool]:
        """Look the query up in an already loaded cache.

        On a miss, run the search and store the result in `cache`.
        Returns the sources and True if it was a cache hit.
        """
        key = self.cacher.make_key(query, k)
        if key in self.cache.keys():
            return [MinimalSource(**item) for item in self.cache[key]], True
        sources = self._retrieve(query, k)
        self.cache[key] = [item.model_dump() for item in sources]
        return sources, False

    def get_query_chunks(
            self,
            query: str,
            k: int,
            print_: bool = False) -> list[MinimalSource]:

        is_in_cache = False
        if self.bonus:
            # Checking if the query is already cached
            sources, is_in_cache = self._retrieve_with_cache(
                query, k)
            if not is_in_cache:
                self.cacher.save_cache(self.cache)
        else:
            sources = self._retrieve(query, k)

        if print_:
            if is_in_cache:
                print(
                    f"{Colors.GREEN.value}[INFO] - "
                    f"Cache hit for query '{query}' with k={k}."
                    f"{Colors.RESET.value}")

            for entry in sources:
                print(f"{entry.file_path} ["
                      f"{entry.first_character_index}:"
                      f"{entry.last_character_index}]")
        return sources

    def get_batch_query_chunks(self,
                               dataset_path: str,
                               k: int,
                               save_directory: str) -> None:

        try:
            with open(dataset_path, mode='r') as fdc:
                dataset = RagDataset.model_validate_json(fdc.read())
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{dataset_path}'{ErrorCodes.OS_ERROR.value}") from e
        except pydantic.ValidationError as e:
            raise ValueError(e)

        minimal_result_list = []
        questions_found_cache = 0

        if self.bonus and self.embeddings is None and any(
            self.cacher.make_key(q.question, k) not in self.cache
                for q in dataset.rag_questions):
            self.embeddings = Embeddings()

        for question in tqdm(dataset.rag_questions,
                             desc="Getting the dataset chunks...",
                             bar_format=TQDM_FMT):

            if self.bonus:
                # Checking if the query is already cached
                query_sources, is_in_cache = self._retrieve_with_cache(
                    question.question, k
                )
                if is_in_cache:
                    questions_found_cache += 1

            else:
                query_sources = self._retrieve(question.question, k)

            minimal_search_result = MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=query_sources
            )
            minimal_result_list.append(minimal_search_result)
        if self.bonus:
            self.cacher.save_cache(self.cache)
            print(
                f"{Colors.GREEN.value}[INFO] - "
                f"Found {questions_found_cache} questions in cache out of "
                f"{len(dataset.rag_questions)} total questions."
                f"{Colors.RESET.value}")

        result = StudentSearchResults(
            search_results=minimal_result_list,
            k=k)
        # Saving result
        self.save_json(save_directory, result)

    def save_json(
            self,
            save_directory: str,
            result: StudentSearchResults) -> None:
        try:
            with open(save_directory, mode='w') as fd:
                fd.write(result.model_dump_json(indent=2))
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{save_directory}'{ErrorCodes.OS_ERROR.value}") \
                    from e
        except pydantic.ValidationError as e:
            raise ValueError(e)
        print(f"{Colors.GREEN.value}"
              f"Saved student_search_results to "
              f"{save_directory}{Colors.RESET.value}")
