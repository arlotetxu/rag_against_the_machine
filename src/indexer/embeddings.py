import torch
from transformers import AutoTokenizer, AutoModel
from src.aux.constants import PathsAndNames, TQDM_FMT, EMBEDDINGS_BATCH_SIZE
from src.aux.colors import Colors
import os
from pathlib import Path
import json
import numpy as np
from tqdm import tqdm


class Embeddings:

    def __init__(self, batch_size: int = EMBEDDINGS_BATCH_SIZE) -> None:
        self.batch_size = batch_size
        self.device = "mps" if torch.backends.mps.is_available() \
            else "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2")
        self.model = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2").to(self.device).eval()

    def save_index_embeddings(self, embeddings: torch.Tensor) -> None:
        self.embeddings_path = os.path.join(
            PathsAndNames.save_index_path.value,
            PathsAndNames.embeddings_name.value
        )
        np.save(self.embeddings_path, embeddings.numpy())

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
            f"Embeddings saved under {self.embeddings_path}."
            f"{Colors.RESET.value}")

    def encode(
            self,
            texts: list[str],
            progress_bar: bool = False) -> torch.Tensor:
        batch_size = self.batch_size
        sums = torch.zeros(len(texts), self.model.config.hidden_size)
        counts = torch.zeros(len(texts), 1)
        starts = range(0, len(texts), batch_size)
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
