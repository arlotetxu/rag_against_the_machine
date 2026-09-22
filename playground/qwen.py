from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

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

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=_dtype,
    trust_remote_code=True,
    ).to(device)


messages = [
    {"role": "user", "content": "What HTTP endpoint is used to dynamically load a LoRA adapter in vLLM?"},
]

inputs = tokenizer.apply_chat_template(
	messages,
	add_generation_prompt=True,
	tokenize=True,
	return_dict=True,
	return_tensors="pt",
    enable_thinking=False,
).to(model.device)

with torch.inference_mode():
    outputs = model.generate(**inputs, max_new_tokens=120)

outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))
