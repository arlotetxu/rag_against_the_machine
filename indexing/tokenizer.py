import re


class Tokenizer:

    def tokenize_code(self, text: str) -> list[str]:
        # Splitting camelCase/PascalCase
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', text)
        # Splitting snake_case
        text = re.sub(r'([_\-])', ' ', text)
        # text = text.lower()
        return re.findall(r"[a-z0-9]+", text.lower())

    def tokenize_other(self, text: str) -> list[str]:
        text = text.lower()
        # text = re.sub(r'([_\-])', ' ', text)
        return re.findall(r"[a-záéíóúñ0-9]+", text)
