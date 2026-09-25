"""Tokenization, stopword removal and stemming for BM25.

Importing this module downloads the NLTK English stopword list if it is
not already installed, which needs network access the first time.
"""
import re
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.downloader import download
download('stopwords', quiet=True)  # type: ignore[no-untyped-call]


class Tokenizer:
    """Turn code and text into BM25 terms.

    Used both at indexing time (on chunks) and at query time, so a
    query's terms go through the same steps as the terms it is matched
    against.
    """

    def __init__(self) -> None:
        """Load the English stopword list and the Snowball stemmer."""
        self.stopwords = set(
            stopwords.words('english'))  # type: ignore[no-untyped-call]
        self.stemmer = SnowballStemmer(
            'english')  # type: ignore[no-untyped-call]

    def tokenize_code(self, text: str) -> list[str]:
        """Split source code into lowercase tokens.

        camelCase and PascalCase names are split into words
        (``getHTTPResponse`` -> ``get``, ``http``, ``response``), while
        snake_case names stay whole (``max_tokens``). Only ASCII letters,
        digits and underscores are kept.

        Args:
            text (str): Code to tokenize.

        Returns:
            list[str]: Tokens in order of appearance.
        """
        # Splitting camelCase/PascalCase
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
        text = re.sub(r'(?<=[A-Z])(?=[A-Z][a-z])', ' ', text)
        # Splitting snake_case
        # text = re.sub(r'([_\-])', ' ', text)
        # text = text.lower()
        return re.findall(r"[a-z0-9_]+", text.lower())

    def tokenize_other(self, text: str) -> list[str]:
        """Split prose into lowercase words.

        Any character other than a letter (including Spanish accented
        letters and ``ñ``) or a digit separates tokens, so ``max_tokens``
        gives ``max`` and ``tokens``.

        Args:
            text (str): Text to tokenize.

        Returns:
            list[str]: Tokens in order of appearance.
        """
        text = text.lower()
        # text = re.sub(r'([_\-])', ' ', text)
        return re.findall(r"[a-záéíóúñ0-9]+", text)

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        """Return ``tokens`` without English stopwords, keeping the order.

        Args:
            tokens (list[str]): Lowercase tokens.

        Returns:
            list[str]: The tokens that are not stopwords.
        """
        filtered = []
        for token in tokens:
            if token not in self.stopwords:
                filtered.append(token)
        return filtered

    def stem(self, tokens: list[str]) -> list[str]:
        """Reduce each token to its English Snowball stem.

        Args:
            tokens (list[str]): Tokens to stem.

        Returns:
            list[str]: The stems, in the same order as ``tokens``.
        """
        stemmed = [
            self.stemmer.stem(token)  # type: ignore[no-untyped-call]
            for token in tokens
            ]
        return stemmed
