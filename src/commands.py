from pathlib import Path
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.indexer.indexer import Indexer
from src.retriever.retrieval import Retrieval
from src.evaluator.evaluate import Evaluate
from src.generator.generator import Generator
# from icecream import ic

# ic.configureOutput(includeContext=True)


"""
========= PENDING TASKS ==========
[X] Revisar posibles errores en paths de ejecucion
[X] Revisar targets en Makefile
[] Añadir Docstrings
[] Añadir README.md
"""


def index(max_chunk_size: int = 800, get_embeddings: bool = False) -> None:
    """Chunk the corpus and build the BM25 index on disk.

    Python files are split with the code chunker and every other file with
    the generic chunker; the resulting BM25 index is pickled to the index
    path defined in ``PathsAndNames``.

    Args:
        max_chunk_size (int, optional): Maximum number of characters per
            chunk. Values outside ``[200, 2000]`` are reset to 800 with a
            warning, and ``Indexer`` caps any value above 800 at 800.
            Defaults to 800.
        get_embeddings (bool, optional): Whether to also compute dense
            embeddings for the chunks with all-MiniLM-L6-v2. Defaults to
            False.

    Raises:
        ValueError: If ``max_chunk_size`` is not an integer.
    """
    if not isinstance(max_chunk_size, int):
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.CHUNK_SIZE_NOK.value}")
    if not isinstance(get_embeddings, bool):
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.EMBEDDINGS_NOK.value}")
    if max_chunk_size > 2000 or max_chunk_size < 200:
        max_chunk_size = 800
        print(f"{Colors.YELLOW.value}[WARNING] - "
              f"{ErrorCodes.MAX_SIZE_CHUNK.value}"
              f"{Colors.RESET.value}")
    indexer = Indexer(max_chunk_size, get_embeddings=get_embeddings)
    indexer.run()


def search(query: str, k: int = 5) -> None:
    """Retrieve the top-k chunks for a single query and print them.

    Each result is printed as ``file_path [first_char:last_char]``.

    Args:
        query (str): Natural-language question to search for.
        k (int, optional): Number of chunks to retrieve. Defaults to 5.

    Raises:
        ValueError: If ``k`` is not a positive integer or ``query`` is not
            a string.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.K_NOK.value}")
    if not isinstance(query, str) or len(query) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.QUERY_NOK.value}")
    retrieval = Retrieval()
    retrieval.get_query_chunks(query=query, k=k, print_=True)


def search_dataset(
        dataset_path: str =
        "data/datasets/AnsweredQuestions/dataset_docs_public.json",
        k: int = 5,
        save_directory: str = "data/output/search_results") -> None:
    """Retrieve the top-k chunks for every question in a dataset.

    The results are written as a ``StudentSearchResults`` JSON file named
    after the dataset file inside ``save_directory``, which is created if
    it does not exist.

    Args:
        dataset_path (str, optional): Path to the ``RagDataset`` JSON file.
            Defaults to
            "data/datasets/AnsweredQuestions/dataset_docs_public.json".
        k (int, optional): Number of chunks to retrieve per question.
            Defaults to 5.
        save_directory (str, optional): Folder where the results file is
            saved. Defaults to "data/output/search_results".

    Raises:
        ValueError: If ``k`` is not a positive integer, a path is not a
            string, the dataset file does not exist, or ``save_directory``
            is the dataset's own folder (the dataset would be overwritten).
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.K_NOK.value}")
    if not isinstance(dataset_path, str) or len(dataset_path) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.DATASET_PATH_NOK.value}")
    if not isinstance(save_directory, str) or len(save_directory) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.SAVE_FOLDER_NOK.value}")

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.DATASET_PATH_NOK.value}")

    save_folder = Path(save_directory)
    if dataset_file.parent == save_folder:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.SAVE_FOLDER_EQ_DATASET.value}")
    save_folder.mkdir(parents=True, exist_ok=True)
    output_path = save_folder / dataset_file.name

    retrieval = Retrieval()
    retrieval.get_batch_query_chunks(dataset_path, k, str(output_path))


def answer(query: str, k: int = 3) -> None:
    """Answer a single query with the LLM (Qwen3-0.6B) and print the result.

    The top-k retrieved chunks are added to the prompt as context.

    Args:
        query (str): Natural-language question to answer.
        k (int, optional): Number of chunks used as context. Defaults to 3.

    Raises:
        ValueError: If ``k`` is not a positive integer or ``query`` is not
            a string.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.K_NOK.value}")
    if not isinstance(query, str) or len(query) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.QUERY_NOK.value}")
    generator = Generator()
    generator.get_single_answer(query, k, print_=True)


def answer_dataset(student_search_results_path: str,
                   save_directory: str) -> None:
    """Answer every question in a search-results file with the LLM.

    For each question, up to ``K_FOR_ANSWER`` of its retrieved chunks are
    used as context for Qwen3-0.6B. The answers are written as a
    ``StudentSearchResultsAndAnswer`` JSON file with the same name as the
    input file inside ``save_directory``, which is created if needed.

    Args:
        student_search_results_path (str): Path to a
            ``StudentSearchResults`` JSON file, as produced by
            ``search_dataset``.
        save_directory (str): Folder where the answers file is saved.

    Raises:
        ValueError: If a path is not a string, the input file does not
            exist, or ``save_directory`` is the input file's own folder
            (the input would be overwritten).
    """
    if not isinstance(student_search_results_path, str) or \
            len(student_search_results_path) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.STUDENT_FILE_NOK.value}")
    if not isinstance(save_directory, str) or \
            len(save_directory) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.SAVE_FOLDER_NOK.value}")

    student_file = Path(student_search_results_path)
    if not student_file.exists():
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.STUDENT_FILE_NOK.value}")
    save_folder = Path(save_directory)

    if student_file.parent == save_folder:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.SAVE_FOLDER_EQ_DATASET.value}")
    save_folder.mkdir(parents=True, exist_ok=True)
    output_path = save_folder / student_file.name

    generator = Generator()
    generator.get_batch_query_answer(
        student_search_results_path, str(output_path))


def evaluate(student_search_results_path: str,
             dataset_path: str,
             k: int = 10) -> None:
    """Compute recall@1 through recall@k of search results against a dataset.

    Questions in the dataset without reference sources are skipped. If
    ``k`` is larger than the number of sources retrieved per question, it
    is lowered to that number with a warning.

    Args:
        student_search_results_path (str): Path to the
            ``StudentSearchResults`` JSON file to evaluate.
        dataset_path (str): Path to the ``RagDataset`` JSON file with the
            reference sources.
        k (int, optional): Highest cutoff to evaluate. Defaults to 10.

    Raises:
        ValueError: If ``k`` is not a positive integer, a path is not a
            string, either file does not exist, or both paths point to the
            same file.
    """
    if not isinstance(k, int) or k < 1:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.K_NOK.value}")
    if not isinstance(dataset_path, str) or len(dataset_path) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.DATASET_PATH_NOK.value}")
    if not isinstance(student_search_results_path, str) or \
            len(student_search_results_path) == 0:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.STUDENT_FILE_NOK.value}")

    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.DATASET_PATH_NOK.value}")
    student_file = Path(student_search_results_path)
    if not student_file.exists():
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.STUDENT_FILE_NOK.value}")
    if student_file == dataset_file:
        raise ValueError(f"{Colors.RED.value}[ERROR] - "
                         f"{ErrorCodes.STUDENT_DATASET_SAME.value}")

    evaluate = Evaluate()
    evaluate.get_recall(student_search_results_path, dataset_path, k)
