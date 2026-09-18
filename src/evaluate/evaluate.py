import pydantic
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.entities.data_model import (
    MinimalSource,
    StudentSearchResults,
    RagDataset)
from tqdm import tqdm


class Evaluate:

    def get_iou(
            self,
            original: tuple[int, int],
            retrieved: tuple[int, int]) -> float:

        orig_start, orig_end = original
        retr_start, retr_end = retrieved

        union = max(
            0, min(orig_end, retr_end) - max(orig_start, retr_start) + 1)
        total_len = (orig_end - orig_start + 1) + \
            (retr_end - retr_start + 1) - union
        return (union / total_len) if total_len > 0 else 0.0

    def get_recall(
            self,
            student_search_results_path: str,
            dataset_path: str,
            k: int) -> None:

        try:
            with open(student_search_results_path, mode='r') as fds:
                student = StudentSearchResults.model_validate_json(fds.read())
            with open(dataset_path, mode='r') as fdd:
                dataset = RagDataset.model_validate_json(fdd.read())
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

            student_results = [minimal
                               for min_search in student.search_results
                               for minimal in min_search.retrieved_sources
                               if min_search.question == question.question]
            # Getting recall
            for k_i in k_values:
                top_k = student_results[:k_i]
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
