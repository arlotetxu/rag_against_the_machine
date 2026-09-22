from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import torch
from llama_cpp import Llama

MODEL_NAME = "Qwen/Qwen3-0.6B"
MODEL_NAME2 = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True)

if tokenizer.pad_token_id is None:
    # ensure we have a pad token to keep batch helpers happy
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Auto-select device with priority: mps > cuda > cpu
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print(f"Device: {device}")

_dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32


llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen3-0.6B-GGUF",    # verifica el nombre exacto del repo en el Hub
    filename="*Qwen3-0.6B-Q8_0.gguf",  # único fichero cuantizado disponible en este repo
    n_ctx=4096,                        # tamaño de contexto que reservas
    n_threads=4,                       # cores de CPU a usar
    verbose=False,
    )


messages = [
    {"role": "user", "content": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM? /no_think"},
]

respuesta = llm.create_chat_completion(
    messages,
    max_tokens=128,
    temperature=0.7,
    top_p=0.8,
    top_k=20,
    min_p=0,
    presence_penalty=1.5
)

print(respuesta["choices"][0]["message"]["content"])
