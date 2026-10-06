"""Answer generation with the LLM from retrieved context."""
import os
from src.generator.prompt import PromptBuild
from src.generator.model import Model
from src.entities.data_model import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer,
    MinimalSource)
from src.cacher.cacher import CacheHandler
from src.retriever.retrieval import Retrieval
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import (
    TQDM_FMT,
    K_FOR_ANSWER,
    MAX_OUT_TOKENS,
    SYSTEM_PROMPT,
    PathsAndNames)
import pydantic
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BatchEncoding
# from icecream import ic


class Generator:
    """Answer questions with the LLM, using retrieved chunks as context."""

    def __init__(self, bonus: bool = False) -> None:
        """Load the retriever, the prompt builder and the LLM.

        This loads the BM25 index and the chunks from disk and the model
        from the Hugging Face cache (downloading it on first use), so it
        can take a while.

        Args:
            bonus (bool, optional): Whether to use the bonus features. If
                true, the answer cache is loaded from disk and the
                retriever uses its search cache and the embeddings.
                Defaults to False.
        """
        self.prompt_builder = PromptBuild()
        self.initial_prompt = SYSTEM_PROMPT
        model_inst = Model()
        self.model = model_inst.model
        self.tokenizer = model_inst.tokenizer
        self.bonus = bonus
        if self.bonus:
            self.cacher = CacheHandler(Path(os.path.join(
                PathsAndNames.cache_path.value,
                PathsAndNames.cache_answers_name.value)))
            self.cache = self.cacher.load_cache()
        self.retrieval = Retrieval(self.bonus)

    def model_launch(self, messages: list[dict[str, str]]) -> str:
        """Run the LLM on a chat conversation and return its reply.

        The messages are formatted with the model's chat template, with
        Qwen3's thinking mode turned off. Generation stops after
        ``MAX_OUT_TOKENS`` new tokens. Only the new tokens are decoded, with
        special tokens removed.

        Args:
            messages (list[dict[str, str]]): Chat messages, each with a
                ``"role"`` (``"system"`` or ``"user"``) and a ``"content"``.

        Returns:
            str: The model's reply.

        Raises:
            TypeError: If the tokenizer returns an unexpected type when
                encoding the prompt or decoding the output.
        """
        encoded = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
            )
        if not isinstance(encoded, BatchEncoding):
            raise TypeError(
                f"Expected BatchEncoding, got {type(encoded).__name__}")
        inputs = encoded.to(self.model.device)

        with torch.inference_mode():
            output = self.model.generate(
                **inputs, max_new_tokens=MAX_OUT_TOKENS)

        output_str = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True
            )
        if not isinstance(output_str, str):
            raise TypeError(
                f"Expected str, got {type(output_str).__name__}")

        return output_str

    def _retrieve_with_cache(
            self,
            query: str,
            k: int,
            prompt_context: str) -> tuple[list[MinimalSource], str, bool]:
        """Return the cached answer for a question, or generate and cache it.

        On a miss, the sources come from a new search and the answer from
        ``get_single_answer``, which repeats that search (the second time
        it is served from the search cache). Neither uses
        ``prompt_context``, which only goes into the cache key. The new
        entry is added to ``cache`` in memory but not saved to disk.

        Args:
            query (str): Question to answer.
            k (int): Number of sources to retrieve.
            prompt_context (str): Context prompt built from the search
                results file, used for the cache key.

        Returns:
            tuple[list[MinimalSource], str, bool]: The sources, the answer,
                and True if they came from the cache.

        Raises:
            OSError: If a chunk's source file cannot be read.
        """
        key = self.cacher.make_answer_key(
            query=query,
            k=k,
            prompt_context=prompt_context)
        if key in self.cache.keys():
            entry = self.cache[key]
            sources = [MinimalSource(**item) for item in entry["sources"]]
            return sources, entry["answer"], True
        sources = self.retrieval.get_query_chunks(query, k)
        answer = self.get_single_answer(query, k)
        self.cache[key] = {
            "sources": [item.model_dump() for item in sources],
            "answer": answer,
        }
        return sources, answer, False

    def get_single_answer(
            self, query: str,
            k: int = 3,
            print_: bool = False) -> str:
        """Answer one question using its top-k retrieved chunks.

        The answer is not cached, even when ``bonus`` is true.

        Args:
            query (str): Question to answer.
            k (int, optional): Number of chunks used as context. Defaults
                to 3.
            print_ (bool, optional): Whether to also print the answer.
                Defaults to False.

        Returns:
            str: The model's answer.

        Raises:
            OSError: If a chunk's source file cannot be read.
        """
        context_prompt = self.prompt_builder.create_context_prompt(
            self.retrieval,
            query=query,
            k=k)
        messages = [
            {"role": "system", "content": self.initial_prompt},
            {"role": "user", "content": context_prompt}
        ]

        output = self.model_launch(messages)
        if print_:
            print(f"\n{output}")
        return output

    def get_chunk(self, path_: str, from_: int, to_: int) -> str:
        """Return the text of a file between two offsets.

        Args:
            path_ (str): Path of the file.
            from_ (int): Start offset (inclusive).
            to_ (int): End offset (exclusive).

        Returns:
            str: ``file_content[from_:to_]``.

        Raises:
            OSError: If the file cannot be read.
        """
        try:
            with open(path_, mode='r') as fd:
                chunk = fd.read()[from_:to_]
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{path_}'{ErrorCodes.OS_ERROR.value}"
                f"{Colors.RESET.value}") from e
        return chunk

    def get_student_results(
            self,
            student_search_results_path: str) -> StudentSearchResults:
        """Load and validate a search-results JSON file.

        Args:
            student_search_results_path (str): Path to a
                ``StudentSearchResults`` JSON file.

        Returns:
            StudentSearchResults: The parsed search results.

        Raises:
            OSError: If the file cannot be read.
            ValueError: If the content does not match
                ``StudentSearchResults``.
        """
        try:
            with open(student_search_results_path, mode='r') as fd:
                student_results = StudentSearchResults.model_validate_json(
                    fd.read())
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{student_search_results_path}' "
                f"{ErrorCodes.OS_ERROR.value}") from e
        except pydantic.ValidationError as e:
            raise ValueError(e)
        return student_results

    def get_context_from_json(
            self,
            question_: str,
            sources: list[MinimalSource]) -> str:
        """Build the user prompt from already retrieved sources.

        Same format as ``PromptBuild.create_context_prompt``, but the
        chunks come from ``sources`` instead of a new search.

        Args:
            question_ (str): Question to answer.
            sources (list[MinimalSource]): Chunks to use as context, in
                order.

        Returns:
            str: The user prompt: the numbered chunks with their file
                paths, then the question and an ``Answer:`` cue.

        Raises:
            OSError: If a chunk's source file cannot be read.
        """
        context_prompt = "context: \n"
        chunk_count = 0
        for minimal_source in sources:
            path_ = minimal_source.file_path
            from_ = minimal_source.first_character_index
            to_ = minimal_source.last_character_index
            chunk = self.get_chunk(path_, from_, to_)
            context_prompt += f"[{chunk_count}] ({path_})\n" \
                f"{chunk}\n\n"
            chunk_count += 1
        context_prompt += f"Question: {question_}\n\nAnswer: "
        return context_prompt

    def get_batch_query_answer(
            self,
            student_search_results_path: str,
            output_file_path: str) -> StudentSearchResultsAndAnswer:
        """Answer every question in a search-results file and save them.

        Each question gets its own first ``K_FOR_ANSWER`` retrieved chunks
        as context (or all of them, if fewer were retrieved). The answers
        are saved as a ``StudentSearchResultsAndAnswer`` JSON file.

        Without ``bonus``, no new retrieval is done. With ``bonus``, each
        question is looked up in the answer cache first; on a miss the
        sources and answer come from a new search (see
        ``_retrieve_with_cache``). The cache is saved once at the end.

        Args:
            student_search_results_path (str): Path to a
                ``StudentSearchResults`` JSON file.
            output_file_path (str): Path of the answers file to write.

        Returns:
            StudentSearchResultsAndAnswer: The questions with their sources
                and answers, as written to ``output_file_path``.

        Raises:
            OSError: If a file cannot be read or written.
            ValueError: If the input file does not match
                ``StudentSearchResults``.
        """
        student_answers: list[MinimalAnswer] = []
        q_counter = 0

        student_results = self.get_student_results(student_search_results_path)
        max_k = len(student_results.search_results[0].retrieved_sources)

        total_questions = len(student_results.search_results)
        questions_in_cache = 0

        for minimal_search in tqdm(
                student_results.search_results,
                desc="Getting dataset answers...",
                bar_format=TQDM_FMT):
            question_ = minimal_search.question
            question_id_ = minimal_search.question_id
            k_ = K_FOR_ANSWER if max_k >= K_FOR_ANSWER else max_k
            sources: list[MinimalSource] = \
                minimal_search.retrieved_sources[0:k_]

            prompt_context = self.get_context_from_json(
                question_,
                sources)

            # ======= Checking cache =====
            if self.bonus:
                sources, answer_, is_in_cache = self._retrieve_with_cache(
                    question_,
                    k_,
                    prompt_context)
                if is_in_cache:
                    questions_in_cache += 1
            else:
                messages = [
                    {"role": "system", "content": self.initial_prompt},
                    {"role": "user", "content": prompt_context}
                    ]

                answer_ = self.model_launch(messages)
            student_answers.append(
                MinimalAnswer(
                    question_id=question_id_,
                    question=question_,
                    retrieved_sources=sources,
                    answer=answer_))
            q_counter += 1
        if self.bonus:
            self.cacher.save_cache(self.cache)

        student_result = StudentSearchResultsAndAnswer(
            search_results=student_answers,
            k=k_)

        self.print_result(total_questions, q_counter, questions_in_cache)
        self.save_student_answer(student_result, output_file_path)
        return student_result

    def print_result(
            self,
            total_questions: int,
            questions_processed: int,
            questions_in_cache: int) -> None:
        """Print how many questions were processed and found in the cache.

        The cache line is only printed when ``bonus`` is true.

        Args:
            total_questions (int): Number of questions in the input file.
            questions_processed (int): Number of questions answered.
            questions_in_cache (int): Number of answers taken from the
                cache.
        """
        print(
            f"{Colors.GREEN.value}"
            f"[INFO] - Loaded {total_questions} "
            f"questions ... "
            f"Processed {questions_processed} of "
            f"{total_questions} questions."
            f"{Colors.RESET.value}")
        if self.bonus:
            print(
                f"{Colors.GREEN.value}[INFO] - "
                f"Found {questions_in_cache} questions in cache out of "
                f"{questions_processed} total questions processed."
                f"{Colors.RESET.value}")

    def save_student_answer(
            self,
            to_save: StudentSearchResultsAndAnswer,
            output_file_path: str
            ) -> None:
        """Write the answers to a JSON file, overwriting it if it exists.

        Args:
            to_save (StudentSearchResultsAndAnswer): Answers to save.
            output_file_path (str): Path of the output file.

        Raises:
            OSError: If the file cannot be written.
        """
        try:
            with open(output_file_path, mode='w') as fd:
                fd.write(to_save.model_dump_json(indent=2))
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{output_file_path}'{ErrorCodes.OS_ERROR.value}"
                f"{Colors.RESET.value}") from e

        print(
            f"{Colors.GREEN.value}"
            f"Saved {Path(output_file_path).name} to "
            f"... {Path(output_file_path).parent}"
            f"{Colors.RESET.value}")
