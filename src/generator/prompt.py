from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.retriever.retrieval import Retrieval

# from icecream import ic


class PromptBuid:

    def __init__(self) -> None:

        self.retrieval = Retrieval()
        self.prompt = "[SYSTEM]\n" \
            "You are the best assistant the answer question about source " \
            "font of vLLM based UNICALLY in the given portions of context. " \
            "If the context hasn't the answer, you MUST indicate that it " \
            "is not possible answer the question with the available " \
            "information. Do not use external knowledge or create new " \
            "information.\n" \
            "[USER]\n"

    def create_context_prompt(self, query: str, k: int) -> str:

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
