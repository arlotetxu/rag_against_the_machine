"""Dense embeddings of the chunks, used by the bonus hybrid search."""
import torch
from transformers import AutoTokenizer, AutoModel
from src.aux.constants import (
    PathsAndNames,
    TQDM_FMT,
    EMBEDDINGS_BATCH_SIZE)
from src.aux.colors import Colors
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm


class Embeddings:
    """Encode texts into unit-length vectors with ``all-MiniLM-L6-v2``.

    The same model encodes the chunks at index time and the queries at
    search time, so the cosine similarity between a query and a chunk is
    the dot product of their vectors.
    """

    def __init__(self, batch_size: int = EMBEDDINGS_BATCH_SIZE) -> None:
        """Load the tokenizer and the model on the best device.

        The device is ``"mps"``, ``"cuda"`` or ``"cpu"``, in that order of
        preference. The model is downloaded from the Hugging Face Hub on
        first use and read from the local cache afterwards.

        Args:
            batch_size (int, optional): Number of texts encoded per batch.
                Defaults to ``EMBEDDINGS_BATCH_SIZE``.
        """

        self.batch_size = batch_size
        self.device = "mps" if torch.backends.mps.is_available() \
            else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2").to(self.device).eval()

    def save_index_embeddings(self, embeddings: torch.Tensor) -> None:
        """Save the chunk embeddings and a JSON file describing them.

        The matrix goes to ``embeddings_name`` as a NumPy file, and the
        model name, encoding settings and matrix shape to
        ``embeddings_info_name``, both in ``save_index_path``. The settings
        in the JSON are written as fixed values, so they must be kept in
        sync with ``encode`` by hand.

        Args:
            embeddings (torch.Tensor): Matrix of shape
                ``(num_chunks, dimension)``, on the CPU, with one row per
                chunk in index order.

        Raises:
            OSError: If either file cannot be written.
        """
        embeddings_path = os.path.join(
            PathsAndNames.save_index_path.value,
            PathsAndNames.embeddings_name.value
        )

        embedding_info = {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "max_length": 256,
            "stride": 32,
            "pooling": "mean",
            "window_aggregation": "mean",
            "normalized": True,
            "num_chunks": int(embeddings.shape[0]),
            "dimension": int(embeddings.shape[1]),
        }
        try:
            np.save(embeddings_path, embeddings.numpy())
            embeddings_info_path = os.path.join(
                PathsAndNames.save_index_path.value,
                PathsAndNames.embeddings_info_name.value
            )
            with open(Path(embeddings_info_path), "w") as fd:
                json.dump(embedding_info, fd, indent=2)
        except OSError as e:
            raise OSError(
                f"{Colors.RED.value}[ERROR] -  "
                f"Error saving embeddings info: {e}"
                f"{Colors.RESET.value}") from e
        print(
            f"{Colors.GREEN.value}"
            f"Embeddings saved under {embeddings_path}."
            f"{Colors.RESET.value}")

    def encode(
            self,
            texts: list[str],
            progress_bar: bool = False) -> torch.Tensor:
        """Encode each text into one normalized embedding.

        Texts longer than 256 tokens are split into overlapping windows
        (stride of 32 tokens). Each window is mean-pooled over its tokens
        and normalized, the windows of a text are averaged, and the result
        is normalized again.

        Args:
            texts (list[str]): Texts to encode.
            progress_bar (bool, optional): Whether to show a tqdm bar over
                the batches. Defaults to False.

        Returns:
            torch.Tensor: Matrix of shape ``(len(texts), hidden_size)`` on
                the CPU, with one unit-length row per text, in order.
        """
        sums = torch.zeros(len(texts), self.model.config.hidden_size)
        counts = torch.zeros(len(texts), 1)
        starts = range(0, len(texts), self.batch_size)
        for start in tqdm(starts, disable=not progress_bar,
                          desc="Generating embeddings...",
                          bar_format=TQDM_FMT):
            batch_range = texts[start:start + self.batch_size]
            batch = self.tokenizer(
                batch_range, max_length=256, truncation=True, stride=32,
                return_overflowing_tokens=True, padding=True,
                return_tensors="pt")
            mapping = batch.pop(
                "overflow_to_sample_mapping") + start
            encoded = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                token_emb = self.model(**encoded).last_hidden_state
            # mean pooling per window
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            window_emb = (token_emb * mask).sum(1) / mask.sum(1).clamp(
                min=1e-9)
            window_emb = torch.nn.functional.normalize(window_emb, dim=1).cpu()

            # cumulate windows in its chunks
            sums.index_add_(0, mapping, window_emb)
            counts.index_add_(
                0, mapping, torch.ones(len(mapping), 1))

        embeddings = torch.nn.functional.normalize(sums / counts, dim=1)

        return embeddings
