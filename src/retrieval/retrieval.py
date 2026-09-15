import pickle
import pydantic
from src.aux.constants import PathsAndNames
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.entities.data_model import (
    RagIndex,
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    RagDataset)
from src.indexer.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
import numpy as np
import uuid
from tqdm import tqdm
from typing import Any

from icecream import ic


class Retrieval:
    def __init__(self) -> None:

        self.tokenizer: Tokenizer = Tokenizer()
        self.bm25_index: BM25Okapi = self.get_bm25_index()
        self.chunks: RagIndex = self.get_chunks()

    def get_bm25_index(self) -> Any:

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
                ragindex_chunks = RagIndex.model_validate_json(fd.read())
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
        except pydantic.ValidationError as e:
            raise ValueError(e)

        # ic(chunks.chunks[0].metadata)
        return ragindex_chunks

    def tokenize_query(self, query: str) -> list[str]:

        query_tokens = set()
        query_tokens_code = self.tokenizer.tokenize_code(query)
        query_tokens_other = self.tokenizer.tokenize_other(query)
        query_tokens = set(query_tokens_other)
        for token in query_tokens_code:
            query_tokens.add(token)
        return list(query_tokens)

    def get_indexes(self, query: str, k: int) -> Any:

        query_tokens = self.tokenize_query(query)
        scores = self.bm25_index.get_scores(
            query_tokens)  # type: ignore[no-untyped-call]
        # Returns the indices that would sort an array:
        scores = np.argsort(scores, descending=True)
        scores = scores.tolist()

        return scores[:k]

    def get_query_chunks(
            self,
            query: str,
            k: int,
            print_: bool = False) -> StudentSearchResults:

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

        if print_:
            for entry in result.search_results:
                for source in entry.retrieved_sources:
                    print(f"{source.file_path} ["
                          f"{source.first_character_index}:"
                          f"{source.last_character_index}]")

        return result

    def get_batch_query_chunks(self,
                               dataset_path: str,
                               k: int,
                               save_directory: str) -> None:
        """
        Uses StudentSearchResults to generate JSON

        class StudentSearchResults(BaseModel):
            search_results: list[MinimalSearchResults]
            k: int

        class MinimalSearchResults(BaseModel):
            question_id: str
            question: str
            retrieved_sources: list[MinimalSource]

        class MinimalSource(BaseModel):
            file_path: str
            first_character_index: int
            last_character_index: int
        """

        try:
            with open(dataset_path, mode='r') as fdc:
                dataset = RagDataset.model_validate_json(fdc.read())
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{dataset_path}'{ErrorCodes.PERMISSION.value} or "
                f"{ErrorCodes.FILE_NOT_FOUND.value}"
                ) from e
        except pydantic.ValidationError as e:
            raise ValueError(e)
        ic(type(dataset))
        pass

    def get_iou(
            self,
            original: tuple[int, int],
            retrieved: tuple[int, int]) -> float:

        orig_start, orig_end = original
        retr_start, retr_end = retrieved

        num = max(0, min(orig_end, retr_end) - max(orig_start, retr_start) + 1)
        denom = (orig_end - orig_start + 1) + (retr_end - retr_start + 1) - num

        return (num / denom) if denom > 0 else 0.0

    def get_recall(self, dataset_path: str, k: int) -> None:

        try:
            with open(dataset_path, mode='r') as fdc:
                dataset = RagDataset.model_validate_json(fdc.read())
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{dataset_path}'{ErrorCodes.PERMISSION.value} or "
                f"{ErrorCodes.FILE_NOT_FOUND.value}"
                ) from e
        except pydantic.ValidationError as e:
            raise ValueError(e)

        k_values = list(range(1, k+1))
        score = {k_i: 0.0 for k_i in k_values}
        num_questions = 0

        for question in tqdm(
                dataset.rag_questions, desc=f"Calculating Recall@{k}..."):
            # Getting source info and saving into a dict[str, tuple(int, int)]
            source: list[MinimalSource] = question.sources
            if not source:
                continue
            num_questions += 1

            # Getting the retrieved info
            results: StudentSearchResults = \
                self.get_query_chunks(question.question, k)
            retrieved: list[MinimalSource] = [
                source
                for result in results.search_results
                for source in result.retrieved_sources]

            # Getting recall
            for k_i in k_values:
                top_k = retrieved[:k_i]
                found = 0
                for correct in source:
                    for candidate in top_k:
                        if candidate.file_path != correct.file_path:
                            continue
                        iou = self.get_iou(
                            (correct.first_character_index,
                             correct.last_character_index),
                            (candidate.first_character_index,
                             candidate.last_character_index))
                        # ic(iou)
                        if iou > 0.05:
                            found += 1
                            break
                score[k_i] += found / len(source)
        if num_questions > 0:
            final_result = {
                k_i: score[k_i] / num_questions for k_i in k_values}
        else:
            final_result = {k: 0.0 for k in k_values}
        self.get_print_recall(final_result, num_questions)

    def get_print_recall(
            self, final_result: dict[int, float],
            num_questions: int) -> None:
        print()
        print("Evaluation Results")
        print("==" * 15)
        print(f"Questions evaluated: {num_questions}")
        for k, result in final_result.items():
            print(f"{Colors.YELLOW.value}"
                  f"Recall@{k}: {result:.2f} ({result * 100:.2f}%)")
        print(f"{Colors.RESET.value}")
