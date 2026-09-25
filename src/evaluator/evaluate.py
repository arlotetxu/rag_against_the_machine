"""Recall@k evaluation of retrieval results against a reference dataset."""
import pydantic
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import TQDM_FMT
from src.entities.data_model import (
    MinimalSource,
    StudentSearchResults,
    RagDataset)
from tqdm import tqdm


class Evaluate:
    """Measure how many reference sources the retriever finds in its top k.

    A reference source counts as found when a retrieved chunk from the same
    file overlaps it with an intersection over union (IoU) above 0.05.
    """

    def get_iou(
            self,
            original: tuple[int, int],
            retrieved: tuple[int, int]) -> float:
        """Return the intersection over union of two offset ranges.

        Both ranges are half-open, ``[start, end)``, like a Python slice.

        Args:
            original (tuple[int, int]): Start and end offsets of the
                reference source.
            retrieved (tuple[int, int]): Start and end offsets of the
                retrieved chunk.

        Returns:
            float: Overlap ratio between 0.0 (disjoint) and 1.0 (identical).
        """
        orig_start, orig_end = original
        retr_start, retr_end = retrieved

        inter = max(
            0, min(orig_end, retr_end) - max(orig_start, retr_start))
        union = (orig_end - orig_start) + \
            (retr_end - retr_start) - inter
        return (inter / union) if union > 0 else 0.0

    def get_data(
            self,
            student_search_results_path: str,
            dataset_path: str
            ) -> tuple[StudentSearchResults, RagDataset]:
        """Load and validate the search-results file and the dataset.

        Args:
            student_search_results_path (str): Path to the
                ``StudentSearchResults`` JSON file.
            dataset_path (str): Path to the ``RagDataset`` JSON file.

        Returns:
            tuple[StudentSearchResults, RagDataset]: The parsed search
                results and dataset.

        Raises:
            OSError: If either file cannot be opened or read.
            ValueError: If either file does not match its model.
        """
        # Getting info from needed files
        try:
            with open(student_search_results_path, mode='r') as fds:
                student = StudentSearchResults.model_validate_json(fds.read())
            with open(dataset_path, mode='r') as fdd:
                dataset = RagDataset.model_validate_json(fdd.read())
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"The file '{dataset_path}'{ErrorCodes.OS_ERROR.value}") from e
        except pydantic.ValidationError as e:
            raise ValueError(e)
        return student, dataset

    def get_recall(
            self,
            student_search_results_path: str,
            dataset_path: str,
            k: int) -> None:
        """Compute recall@1 through recall@k and print the results.

        For each dataset question with reference sources, recall@i is the
        share of those sources found among the first ``i`` retrieved
        chunks. The printed values are averages over those questions.
        Retrieved chunks are matched to dataset questions by question text,
        not by id.

        If ``k`` is larger than the number of chunks retrieved per question
        (read from the first result), it is lowered to that number with a
        warning.

        Args:
            student_search_results_path (str): Path to the
                ``StudentSearchResults`` JSON file.
            dataset_path (str): Path to the ``RagDataset`` JSON file.
            k (int): Highest cutoff to evaluate.

        Raises:
            OSError: If either file cannot be opened or read.
            ValueError: If either file does not match its model.
        """
        # # Getting info from needed files
        student, dataset = self.get_data(
            student_search_results_path,
            dataset_path
        )

        # Checking f the k value indicated is greater than sources available
        max_k = len(student.search_results[0].retrieved_sources)
        if k > max_k:
            print(
                f"{Colors.YELLOW.value}[WARNING] - "
                f"The k value indicated is greater than sources in student "
                f"file sources. Changing to maximum ({max_k})."
                f"{Colors.RESET.value}")
            k = max_k

        k_values = list(range(1, k+1))
        score = {k_i: 0.0 for k_i in k_values}
        num_questions = 0

        for question in tqdm(dataset.rag_questions,
                             desc=f"Calculating Recall@{k}...",
                             bar_format=TQDM_FMT):
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
        """Print the recall values as a small report.

        Args:
            final_result (dict[int, float]): Average recall for each cutoff,
                keyed by cutoff.
            num_questions (int): Number of questions that were evaluated.
        """
        print()
        print("Evaluation Results")
        print("==" * 15)
        print(f"Questions evaluated: {num_questions}")
        for k, result in final_result.items():
            print(f"{Colors.YELLOW.value}"
                  f"Recall@{k}: {result:.2f} ({result * 100:.2f}%)")
        print(f"{Colors.RESET.value}")
