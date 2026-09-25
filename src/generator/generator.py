"""Answer generation with the LLM from retrieved context."""
from src.generator.prompt import PromptBuild
from src.generator.model import Model
from src.entities.data_model import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer)
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT, K_FOR_ANSWER, MAX_OUT_TOKENS
import pydantic
from pathlib import Path
from tqdm import tqdm
import torch
from transformers import BatchEncoding
# from icecream import ic


class Generator:
    """Answer questions with the LLM, using retrieved chunks as context."""

    def __init__(self) -> None:
        """Load the retriever, the prompt builder and the LLM.

        This loads the BM25 index and the chunks from disk and the model
        from the Hugging Face cache (downloading it on first use), so it
        can take a while.
        """
        self.prompt_builder = PromptBuild()
        model_inst = Model()
        self.model = model_inst.model
        self.tokenizer = model_inst.tokenizer

    def model_launch(self, messages: list[dict[str, str]]) -> str:
        """Run the LLM on a chat conversation and return its reply.

        The messages are formatted with the model's chat template, with
        Qwen3's thinking mode turned off. Generation stops after
        ``MAX_OUT_TOKENS`` new tokens. Only the new tokens are decoded, and
        the text is cut at the last ``<|im_end|>`` marker if there is one.

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

    def get_single_answer(
            self, query: str,
            k: int = 3,
            print_: bool = False) -> None:
        """Answer one question using its top-k retrieved chunks.

        The answer is not returned or saved; it is only printed, and only
        when ``print_`` is true.

        Args:
            query (str): Question to answer.
            k (int, optional): Number of chunks used as context. Defaults
                to 3.
            print_ (bool, optional): Whether to print the answer. Defaults
                to False.

        Raises:
            OSError: If a chunk's source file cannot be read.
        """
        initial_prompt = self.prompt_builder.prompt
        context_prompt = self.prompt_builder.create_context_prompt(
                    query=query,
                    k=k
                )
        messages = [
            {"role": "system", "content": initial_prompt},
            {"role": "user", "content": context_prompt}
        ]

        output = self.model_launch(messages)
        if print_:
            print(output)

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
                f"The file '{path_}'{ErrorCodes.OS_ERROR.value}") from e
        return chunk

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
                f"The file '{output_file_path}'{ErrorCodes.OS_ERROR.value}") \
                    from e

        print(
            f"{Colors.GREEN.value}"
            f"Saved {Path(output_file_path).name} to "
            f"... {Path(output_file_path).parent}"
            f"{Colors.RESET.value}")

    def get_batch_query_answer(
            self,
            student_search_results_path: str,
            output_file_path: str) -> None:
        """Answer every question in a search-results file and save them.

        Each question gets its own first ``K_FOR_ANSWER`` retrieved chunks
        as context (or all of them, if fewer were retrieved). No new
        retrieval is done. The answers are saved as a
        ``StudentSearchResultsAndAnswer`` JSON file.

        Args:
            student_search_results_path (str): Path to a
                ``StudentSearchResults`` JSON file.
            output_file_path (str): Path of the answers file to write.

        Raises:
            OSError: If a file cannot be read or written.
            ValueError: If the input file does not match
                ``StudentSearchResults``.
        """
        student_answers: list[MinimalAnswer] = []
        q_counter = 0
        initial_prompt = self.prompt_builder.prompt

        try:
            with open(student_search_results_path, mode='r') as fd:
                student_results = StudentSearchResults.model_validate_json(
                    fd.read())
                total_questions = len(student_results.search_results)
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{student_search_results_path}' "
                f"{ErrorCodes.OS_ERROR.value}") from e
        except pydantic.ValidationError as e:
            raise ValueError(e)

        max_k = len(student_results.search_results[0].retrieved_sources)

        for minimal_search in tqdm(
                student_results.search_results,
                desc="Getting dataset answers...",
                bar_format=TQDM_FMT):
            context_prompt = "context: \n"
            chunk_count = 0
            question_ = minimal_search.question
            question_id_ = minimal_search.question_id
            k_ = K_FOR_ANSWER if max_k >= K_FOR_ANSWER else max_k
            sources = minimal_search.retrieved_sources[0:k_]

            for minimal_source in sources:
                path_ = minimal_source.file_path
                from_ = minimal_source.first_character_index
                to_ = minimal_source.last_character_index
                chunk = self.get_chunk(path_, from_, to_)
                context_prompt += f"[{chunk_count}] ({path_})\n" \
                    f"{chunk}\n\n"
                chunk_count += 1
            context_prompt += f"Question: {question_}\n\nAnswer: "

            messages = [
                {"role": "system", "content": initial_prompt},
                {"role": "user", "content": context_prompt}
                ]

            answer_ = self.model_launch(messages)
            student_answers.append(
                MinimalAnswer(
                    question_id=question_id_,
                    question=question_,
                    retrieved_sources=sources,
                    answer=answer_))
            q_counter += 1

        to_save = StudentSearchResultsAndAnswer(
            search_results=student_answers,
            k=k_)

        print(
            f"{Colors.GREEN.value}"
            f"Loaded {total_questions} "
            f"questions ... "
            f"Processed {q_counter} of "
            f"{total_questions} questions."
            f"{Colors.RESET.value}")
        self.save_student_answer(to_save, output_file_path)
