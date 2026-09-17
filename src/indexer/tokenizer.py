import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.downloader import download
download('stopwords')  # type: ignore[no-untyped-call]


class Tokenizer:
    def __init__(self) -> None:
        self.stopwords = set(
            stopwords.words('english'))  # type: ignore[no-untyped-call]
        self.stemmer = SnowballStemmer(
            'english')  # type: ignore[no-untyped-call]

    def tokenize_code(self, text: str) -> list[str]:
        # Splitting camelCase/PascalCase
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', text)
        # Splitting snake_case
        # text = re.sub(r'([_\-])', ' ', text)
        # text = text.lower()
        return re.findall(r"[a-z0-9_]+", text.lower())

    def tokenize_other(self, text: str) -> list[str]:
        text = text.lower()
        # text = re.sub(r'([_\-])', ' ', text)
        return re.findall(r"[a-záéíóúñ0-9]+", text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        filtered = []
        for token in tokens:
            if token not in self.stopwords:
                filtered.append(token)
        return filtered

    def stem(self, tokens: list[str]) -> list[str]:
        stemmed = [
            self.stemmer.stem(token)  # type: ignore[no-untyped-call]
            for token in tokens
            ]
        return stemmed
