from fastapi import FastAPI, Request
from pydantic import BaseModel, Field, ConfigDict
from src.retriever.retrieval import Retrieval
from src.entities.data_model import MinimalSource


# ===========================INPUTS===========================
class QueryRequest(BaseModel):

    # Removes whitespaces at the query's beginning and end
    model_config = ConfigDict(str_strip_whitespace=True)

    query: str = Field(min_length=1)
    k: int = Field(default=5, ge=1)


# ===========================OUTPUTS===========================
class SearchResponse(BaseModel):

    question: str
    sources: list[MinimalSource]


class AnswerResponse(BaseModel):

    response: SearchResponse
    answer: str


app = FastAPI(
    title="RAG against the machine",
    description="Local HTTP API to query the vLLM index and answer questions.",
    version="1.0.0",
)


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
