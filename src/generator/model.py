from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_NAME2 = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_NAME3 = "Qwen/Qwen2.5-0.5B-Instruct"
MODEL_NAME4 = "HuggingFaceTB/SmolLM2-360M-Instruct"


class Model:

    def __init__(self) -> None:
        my_model = MODEL_NAME
        self.tokenizer = AutoTokenizer.from_pretrained(
            my_model,
            trust_remote_code=True)

        if self.tokenizer.pad_token_id is None:
            # ensure we have a pad token to keep batch helpers happy
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Auto-select device with priority: mps > cuda > cpu
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        _dtype = torch.float16 if self.device in ["cuda", "mps"] else \
            torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            my_model,
            dtype=_dtype,
            trust_remote_code=True,
            ).to(self.device)  # type: ignore[arg-type]
