"""Prompt construction for answer generation."""
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.retriever.retrieval import Retrieval
from src.aux.constants import SYSTEM_PROMPT


# from icecream import ic


class PromptBuild:
    """Build the system prompt and the context prompt sent to the LLM.

    Attributes:
        retrieval (Retrieval): Retriever used to fetch context chunks.
        prompt (str): System prompt that tells the model to answer only
            from the given context and to say so when the context does not
            contain the answer.
        context_prompt (str): Last prompt built by
            ``create_context_prompt``. Only set after the first call.
    """

    def __init__(self) -> None:
        """Load the retriever and set the system prompt.

        Creating the ``Retrieval`` loads the BM25 index and the chunks
        from disk, so ``index`` must have been run first.
        """
        self.retrieval = Retrieval()
        self.prompt = SYSTEM_PROMPT

    def create_context_prompt(self, query: str, k: int) -> str:
        """Retrieve the top-k chunks for a query and build the user prompt.

        Each chunk is read from its source file and added as
        ``[n] (file_path)`` followed by its text. The prompt ends with the
        question and an ``Answer:`` cue for the model to continue.

        Args:
            query (str): Question to answer.
            k (int): Number of chunks to include as context.

        Returns:
            str: The full user prompt. It is also stored in
                ``context_prompt``.

        Raises:
            OSError: If a chunk's source file cannot be read.
        """
        self.context_prompt = "context: \n"
        num_chunk = 0
        query_chunks = self.retrieval.get_query_chunks(query, k)
        for chunk in query_chunks:
            file_path = chunk.file_path
            from_char = chunk.first_character_index
            to_char = chunk.last_character_index
            try:
                with open(file_path, mode='r') as fd:
                    text = fd.read()[from_char:to_char]
                self.context_prompt += f"[{num_chunk}] "
                self.context_prompt += f"({file_path})\n{text}\n\n"
                num_chunk += 1
            except OSError as e:
                raise OSError(f"{Colors.RED.value}[ERROR] - "
                              f"The file '{file_path}'"
                              f"{ErrorCodes.OS_ERROR.value}") from e
        self.context_prompt += f"Question: {query}\n\n"
        self.context_prompt += "Answer: "

        return self.context_prompt
