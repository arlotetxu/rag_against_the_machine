"""Corpus ingestion: chunking, tokenization and BM25 index building."""
import os
from pathlib import Path
from tqdm import tqdm
from src.aux.colors import Colors
from src.aux.error_desc import ErrorCodes
from src.aux.constants import PathsAndNames
from src.aux.constants import TQDM_FMT
from src.indexer.tokenizer import Tokenizer
from rank_bm25 import BM25Okapi
from src.entities.data_model import IndexedChunk, RagIndex
import pickle
from pydantic import ValidationError
from src.chunker.gen_code_chunks import ChunkerCode
from src.chunker.gen_other_chunks import ChunkOther
from src.indexer.embeddings import Embeddings
# from icecream import ic


class Indexer:
    """Build the BM25 index and the chunks file from the raw corpus."""

    def __init__(
            self, max_chunk_size: int = 800,
            min_chunk_tokens: int = 10,
            bonus: bool = False) -> None:
        """Set the chunking limits and the embedding option.

        Also deletes the search and answer cache files, since their
        entries point to chunks of the index that is about to be replaced.

        Args:
            max_chunk_size (int, optional): Maximum chunk size passed to the
                chunkers. Values above 800 are capped at 800. Defaults to
                800.
            min_chunk_tokens (int, optional): Chunks with fewer tokens than
                this are dropped from the index. Defaults to 10.
            bonus (bool, optional): Whether to also compute and save the
                chunk embeddings. Defaults to False.
        """
        self.max_chunk = 800 if max_chunk_size > 800 else max_chunk_size
        self.min_chunk_tokens = min_chunk_tokens
        self.files_lst: dict[str, str] = {}
        self.chunks: dict[str, IndexedChunk] = {}
        self.bonus = bonus
        # Deleting the cache files since the index is being rebuilt
        cache_sources_path = Path(os.path.join(
            PathsAndNames.cache_path.value,
            PathsAndNames.cache_sources_name.value))
        cache_answer_path = Path(os.path.join(
            PathsAndNames.cache_path.value,
            PathsAndNames.cache_answers_name.value))
        cache_sources_path.unlink(missing_ok=True)
        cache_answer_path.unlink(missing_ok=True)

    def get_input_files(self) -> None:
        """Collect every file under the corpus folder into ``files_lst``.

        Subfolders are walked recursively. Files get the ids ``id0``,
        ``id1``... in the order they are found.
        """
        path = PathsAndNames.corpus_path.value
        index = 0
        for root, dirs, files in os.walk(path):
            for file in files:
                self.files_lst[f"id{index}"] = (os.path.join(root, file))
                index += 1
        print(f"{Colors.GREEN.value}"
              f"Documents read: {len(self.files_lst)}"
              f"{Colors.RESET.value}")

    def tokenize_chunks(self) -> list[list[str]]:
        """Turn each chunk into the list of terms that BM25 will index.

        Python chunks use the code tokenizer and the rest use the text
        tokenizer. Chunks with fewer than ``min_chunk_tokens`` tokens are
        removed from ``chunks``. The kept chunks also get the tokens of
        their file path, so a query can match on file and folder names.
        Stopwords are then removed and the tokens stemmed.

        Returns:
            list[list[str]]: The terms of each kept chunk, in the same order
                as ``chunks``.
        """
        corpus_tokens: list[list[str]] = []
        discarded: list[str] = []
        tokenizer = Tokenizer()
        for id, meta in tqdm(self.chunks.items(),
                             desc="Tokenizing...",
                             bar_format=TQDM_FMT):
            path = meta.metadata.file_path
            is_py = Path(path).suffix == '.py'
            if is_py:
                tokens = tokenizer.tokenize_code(meta.text)
            else:
                tokens = tokenizer.tokenize_other(meta.text)
            if len(tokens) < self.min_chunk_tokens:
                discarded.append(id)
                continue
            if is_py:
                tokens.extend(tokenizer.tokenize_code(path))
            else:
                tokens.extend(tokenizer.tokenize_other(path))
            tokens = tokenizer.remove_stopwords(tokens)
            tokens = tokenizer.stem(tokens)
            corpus_tokens.append(tokens)

        for id in discarded:
            del self.chunks[id]
        return corpus_tokens

    def bm25_index(self) -> BM25Okapi:
        """Tokenize the chunks and build the BM25 index over them.

        Uses ``k1=2.0`` and ``b=0.3`` (a low ``b`` penalises long chunks
        only slightly). Chunks dropped during tokenization are also
        removed from ``chunks``, so it stays aligned with the index.

        Returns:
            BM25Okapi: The index, with one document per kept chunk.
        """
        corpus_tokens = self.tokenize_chunks()
        bm25_index = BM25Okapi(
            corpus_tokens, k1=2.0, b=0.3)  # type: ignore[no-untyped-call]
        return bm25_index

    def save_index_chunks(self, bm25_index: BM25Okapi) -> None:
        """Save the BM25 index as a pickle and the chunks as JSON.

        Both files go to the folders set in ``PathsAndNames`` and
        overwrite any previous version. The index folder is created if
        needed. The chunks folder is the same one by default, so it exists
        by then.

        Args:
            bm25_index (BM25Okapi): Index built by ``bm25_index``.

        Raises:
            PermissionError: If either file cannot be written for lack of
                permissions.
            OSError: If either file cannot be written for another reason.
                These errors are not caught here.
            ValueError: If the chunks do not match ``RagIndex``.
        """
        file_2_save = PathsAndNames.index_name.value
        path_2_save = Path(PathsAndNames.save_index_path.value)

        try:
            path_2_save.mkdir(parents=True, exist_ok=True)

            with open(path_2_save / file_2_save, mode='wb') as fd:
                pickle.dump(bm25_index, fd)
        except PermissionError as e:
            raise PermissionError(
                f"{Colors.RED.value}[ERROR] -  "
                f"The file {path_2_save / file_2_save}"
                f"{ErrorCodes.PERMISSION.value}"
                f"{Colors.RESET.value}") from e

        file_2_save = PathsAndNames.chunks_json.value
        path = PathsAndNames.save_chunks.value
        try:
            chunk_list = list(self.chunks.values())
            with open(path + file_2_save, mode='w', encoding='utf-8') as fd:
                fd.write(RagIndex(chunks=chunk_list).model_dump_json(indent=2))
        except PermissionError as e:
            raise PermissionError(
                f"{Colors.RED.value}[ERROR] -  "
                f"The file {file_2_save}"
                f"{ErrorCodes.PERMISSION.value}"
                f"{Colors.RESET.value}") from e
        except ValidationError as e:
            raise ValueError(e)
        print(f"{Colors.GREEN.value}"
              f"Corpus ingestion complete! "
              f"Indexed {len(self.chunks)} chunks under {path + file_2_save}"
              f"{Colors.RESET.value}")

    def run(self) -> None:
        """Run the whole ingestion: read, chunk, index and save.

        With ``bonus``, the embeddings of the kept chunks are then computed
        and saved too, in the same order as the chunks file.

        Raises:
            OSError: If any file cannot be read or written. The original
                error is chained as its cause.
            ValueError: If the chunks do not match ``RagIndex``.
            UnicodeDecodeError: If a Python file is not valid UTF-8, or a
                chunk cut falls inside a multi-byte character.
        """
        try:
            self.get_input_files()
            self.chunks = ChunkerCode(
                files_lst=self.files_lst,
                chunks=self.chunks,
                max_chunk_size=self.max_chunk).chunk_py()
            self.chunks = ChunkOther(
                files_lst=self.files_lst,
                chunks=self.chunks,
                max_chunk_size=self.max_chunk
            ).chunk_others()
            bm25_index = self.bm25_index()
            self.save_index_chunks(bm25_index)
            if self.bonus:
                embedder = Embeddings()
                matrix = embedder.encode(
                    [c.text for c in self.chunks.values()], progress_bar=True)
                embedder.save_index_embeddings(matrix)

        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] - "
                f"Any file from corpus {ErrorCodes.OS_ERROR.value}."
                f"{Colors.RESET.value}") from e
