from src.generator.prompt import PromptBuid
from src.generator.model import Model
from src.entities.data_model import (
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
    MinimalAnswer)
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT, K_FOR_ANSWER
import pydantic
from tqdm import tqdm
import torch
from transformers import BatchEncoding
# from icecream import ic


class Generator:
    def __init__(self) -> None:

        self.prompt_builder = PromptBuid()
        model_inst = Model()
        self.model = model_inst.model
        self.tokenizer = model_inst.tokenizer

    def model_launch(self, messages: list[dict[str, str]]) -> str:

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
            output = self.model.generate(**inputs, max_new_tokens=150)

        output_str = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[-1]:]
            )
        if not isinstance(output_str, str):
            raise TypeError(
                f"Expected str, got {type(output_str).__name__}")

        to_cut = output_str.rfind('<|im_end|>')
        if to_cut == -1:
            return output_str
        return output_str[0:to_cut]

    def get_single_answer(
            self, query: str,
            k: int = 3,
            print_: bool = False) -> None:

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

        try:
            with open(output_file_path, mode='w') as fd:
                fd.write(to_save.model_dump_json(indent=2))
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{output_file_path}'{ErrorCodes.OS_ERROR.value}") \
                    from e
        except pydantic.ValidationError as e:
            raise ValueError(e)

        print(
            f"{Colors.GREEN.value}"
            f"Saved student_search_results_and_answer to "
            f"... {output_file_path}"
            f"{Colors.RESET.value}")

    def get_batch_query_answer(
            self,
            student_search_results_path: str,
            output_file_path: str) -> None:

        k_ = K_FOR_ANSWER
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

        for minimal_search in tqdm(
                student_results.search_results,
                desc="Getting dataset answers...",
                bar_format=TQDM_FMT):
            context_prompt = "context: \n"
            chunk_count = 0
            question_ = minimal_search.question
            question_id_ = minimal_search.question_id
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
