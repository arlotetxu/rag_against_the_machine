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

    # Removes whitespaces at the query's beginning and end
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1, le=10)


# ===========================OUTPUTS===========================
class SearchResponse(BaseModel):

    question: str
    sources: list[MinimalSource]


class AnswerResponse(BaseModel):

    query: str
    answer: str


app = FastAPI(
    title="RAG against the machine",
    description="Local HTTP API to query the vLLM index and answer questions.",
    version="1.0.0",
)


@app.get("/")
def server_status() -> str:
    return "Server up and running! :-)"


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok"}


@app.post("/search")
def search(body: QueryRequest, request: Request) -> SearchResponse:
    """Retrieve the k most relevant sources for a question."""
    retriver: Retrieval = request.app.state.retrieval
    retrieved_sources = retriver.get_query_chunks(
        query=body.query,
        k=body.k)
    return SearchResponse(question=body.query, sources=retrieved_sources)


@app.post("/answer")
def answer(body: QueryRequest, request: Request) -> AnswerResponse:
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
    generator: Generator = request.app.state.generator
    output_path = str(build_output_path(
        student_search_results_path,
        save_directory))
    student_answer = generator.get_batch_query_answer(
        student_search_results_path=student_search_results_path,
        output_file_path=output_path)
    return student_answer
