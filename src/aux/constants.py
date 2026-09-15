from enum import Enum


class PathsAndNames(Enum):
    corpus_path = "data/raw"
    save_index_path = "data/processed"
    index_name = "bm25_index.pkl"
    save_chunks = "data/processed"
    chunks_json = "/chunks.json"
    search_path = "data/output/search_results"
    search_file_name = "my_search.json"
    answared_code = "data/datasets/AnsweredQuestions/dataset_code_public.json"
    answared_other = "data/datasets/AnsweredQuestions/dataset_docs_public.json"
