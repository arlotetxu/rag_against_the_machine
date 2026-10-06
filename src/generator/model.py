"""Loading of the LLM used to generate the answers."""
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.aux.constants import MODEL_NAME
import torch


class Model:
    """Tokenizer and causal language model, loaded on the best device.

    Attributes:
        tokenizer: Tokenizer of ``MODEL_NAME``. If it has no pad token, the
            end-of-sequence token is used instead.
        device (str): ``"mps"``, ``"cuda"`` or ``"cpu"``, in that order of
            preference.
        model: ``MODEL_NAME`` loaded on ``device``, in float16 on a GPU and
            float32 on the CPU.
    """

    def __init__(self) -> None:
        """Load the tokenizer and the model and move the model to a device.

        The model is downloaded from the Hugging Face Hub on first use and
        read from the local cache afterwards.
        """
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
