from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from icecream import ic

#Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


# Sentences we want sentence embeddings for
sentences = ['This is an example sentence', 'Each sentence is converted']

# Load model from HuggingFace Hub
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

# Tokenize sentences
encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')


# Compute token embeddings
with torch.no_grad():
    model_output = model(**encoded_input)
ic(model_output.last_hidden_state.shape)

# Perform pooling
sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

# Normalize embeddings
sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

# print("Sentence embeddings:")
# print(sentence_embeddings)


"""
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

sentences = [
    "How do I install vLLM?",
    "Setup instructions using pip",
    "The tokenizer splits text into subwords",
]
vectors = model.encode(sentences, normalize_embeddings=True)
print(vectors.shape)  # (3, 384) ← un vector de 384 números por frase

print(vectors[0] @ vectors[1])  # valor alto: significado parecido
print(vectors[0] @ vectors[2])  # valor más bajo: poca relación
"""


"""
MEJOR
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

documents = [
    "Install vLLM with pip install vllm",
    "LoRA adapters can be loaded at runtime",
    "The scheduler decides which requests run next",
    "def load_module_from_path(module_name, path):",
]

# Indexar: se hace una vez
document_matrix = model.encode(documents, normalize_embeddings=True)
print(document_matrix.shape)  # (4, 384)

# Buscar: se hace en cada pregunta
query_vector = model.encode("How can I set up vLLM?", normalize_embeddings=True)
scores = document_matrix @ query_vector   # (4, 384) @ (384,) → (4,) un score por documento
ranking = np.argsort(scores)[::-1]         # posiciones de mayor a menor score

for position in ranking:
    print(f"{scores[position]:.3f}  {documents[position]}")
"""
