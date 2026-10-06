"""Local HTTP API exposing search and answer generation.

The ``Retrieval`` and ``Generator`` are not created here: the ``server``
command loads them once and stores them in ``app.state`` before starting
uvicorn, so every request reuses the same index and model.

Attributes:
    generation_lock (threading.Lock): Serializes ``/answer`` requests, so
        only one runs the LLM at a time.
"""
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, ConfigDict
from src.retriever.retrieval import Retrieval
from src.generator.generator import Generator
from src.entities.data_model import (
    MinimalSource,
    StudentSearchResults,
    StudentSearchResultsAndAnswer
)
from src.commands import build_output_path
import threading


generation_lock = threading.Lock()


# ===========================INPUTS===========================
class QueryRequest(BaseModel):
    """Body of the ``/search`` and ``/answer`` requests.

    Attributes:
        query (str): Question. Leading and trailing whitespace is removed
            before validation, so a blank query is rejected.
        k (int): Number of chunks to retrieve, from 1 to 10. Defaults
            to 5.
    """

    # Removes whitespaces at the query's beginning and end
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=10)


# ===========================OUTPUTS===========================
class SearchResponse(BaseModel):
    """Response of ``/search``: the question and its sources, best first."""

    question: str
    sources: list[MinimalSource]


class AnswerResponse(BaseModel):
    """Response of ``/answer``: the question and the model's answer."""

    query: str
    answer: str


app = FastAPI(
    title="RAG against the machine",
    description="Local HTTP API to query the vLLM index and answer questions.",
    version="1.0.0",
)


@app.get("/")
def server_status() -> str:
    """Return a short message saying the server is running."""
    return "Server is up and running! :-)"


@app.get("/health")
def health() -> dict[str, str]:
    """Return ``{"status": "ok"}`` for health checks."""
    return {"status": "ok"}


@app.post("/search")
def search(body: QueryRequest, request: Request) -> SearchResponse:
    """Retrieve the k most relevant sources for a question.

    Args:
        body (QueryRequest): Question and ``k``.
        request (Request): Incoming request, used to reach the shared
            ``Retrieval`` in ``app.state``.

    Returns:
        SearchResponse: The question and its sources, best first.
    """
    retriver: Retrieval = request.app.state.retrieval
    retrieved_sources = retriver.get_query_chunks(
        query=body.query,
        k=body.k)
    return SearchResponse(question=body.query, sources=retrieved_sources)


@app.post("/answer")
def answer(body: QueryRequest, request: Request) -> AnswerResponse:
    """Answer a question with the LLM, using its top-k chunks as context.

    Requests are handled one at a time (see ``generation_lock``). The
    answer is also printed in the server console.

    Args:
        body (QueryRequest): Question and ``k``.
        request (Request): Incoming request, used to reach the shared
            ``Generator`` in ``app.state``.

    Returns:
        AnswerResponse: The question and the model's answer.
    """
    generator: Generator = request.app.state.generator
    with generation_lock:
        retrieved_answer = generator.get_single_answer(
            body.query,
            body.k,
            print_=True)
    return AnswerResponse(query=body.query, answer=retrieved_answer)


@app.post("/search_dataset")
def search_dataset(
        request: Request,
        dataset_path: str,
        k: int,
        save_directory: str) -> StudentSearchResults:
    """Retrieve the top-k chunks for every question in a dataset file.

    The parameters go in the query string. The paths are on the server's
    file system. The results are also saved in ``save_directory``, in a
    file named after the dataset file.

    Args:
        request (Request): Incoming request, used to reach the shared
            ``Retrieval`` in ``app.state``.
        dataset_path (str): Path to a ``RagDataset`` JSON file.
        k (int): Number of chunks to retrieve per question.
        save_directory (str): Folder for the results file. Created if
            needed; it cannot be the dataset's own folder.

    Returns:
        StudentSearchResults: The results, as saved.

    Raises:
        ValueError: If the dataset does not exist, ``save_directory`` is
            the dataset's folder, or the dataset is not valid.
        OSError: If a file cannot be read or written.
    """
    retriver: Retrieval = request.app.state.retrieval
    output_path = str(build_output_path(dataset_path, save_directory))
    student_search = retriver.get_batch_query_chunks(
        dataset_path=dataset_path,
        k=k,
        save_directory=output_path)
    return student_search


@app.post("/answer_dataset")
def answer_dataset(
        student_search_results_path: str,
        save_directory: str,
        request: Request) -> StudentSearchResultsAndAnswer:
    """Answer every question in a search-results file.

    The parameters go in the query string. The paths are on the server's
    file system. The answers are also saved in ``save_directory``, in a
    file named after the input file.

    Args:
        student_search_results_path (str): Path to a
            ``StudentSearchResults`` JSON file, as produced by
            ``/search_dataset``.
        save_directory (str): Folder for the answers file. Created if
            needed; it cannot be the input file's own folder.
        request (Request): Incoming request, used to reach the shared
            ``Generator`` in ``app.state``.

    Returns:
        StudentSearchResultsAndAnswer: The questions with their sources and
            answers, as saved.

    Raises:
        ValueError: If the input file does not exist, ``save_directory``
            is its folder, or the file is not valid.
        OSError: If a file cannot be read or written.
    """
    generator: Generator = request.app.state.generator
    output_path = str(build_output_path(
        student_search_results_path,
        save_directory))
    student_answer = generator.get_batch_query_answer(
        student_search_results_path=student_search_results_path,
        output_file_path=output_path)
    return student_answer
