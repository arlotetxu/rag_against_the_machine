*This project has been created as part of the 42 curriculum by joflorid.*

# RAG against the machine

## Description

**RAG against the machine** is a Retrieval-Augmented Generation (RAG) system
that answers questions about the [vLLM](https://github.com/vllm-project/vllm)
code base (version 0.10.1): both its Python source code and its documentation.

The goal is to answer a question with information that actually exists in the
corpus instead of what a language model remembers. To do that, the system:

1. **Indexes** the corpus: it splits about 2,900 files into ~42,500 chunks and
   builds a BM25 index over them (plus, optionally, dense embeddings).
2. **Retrieves** the chunks most relevant to a question.
3. **Generates** an answer with a small local LLM (Qwen3-0.6B), using only
   those chunks as context.
4. **Evaluates** retrieval quality with recall@k against reference datasets of
   questions whose answer location in the corpus is known.

Everything runs locally, from a command-line interface or a small HTTP API.

**Bonus features:**

- Hybrid search: BM25 combined with semantic embeddings (`all-MiniLM-L6-v2`)
  through Reciprocal Rank Fusion.
- A disk cache for search results and generated answers.
- A local HTTP API built with FastAPI.

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://docs.astral.sh/uv/) to manage the virtual environment and the
  dependencies
- About 2 GB of free disk space for the models, which are downloaded from the
  Hugging Face Hub on first use and cached afterwards
- Optional: a GPU. The code picks Apple Silicon (`mps`), then CUDA, then CPU.

### Installation

```bash
git clone <repository-url> rag_against_the_machine
cd rag_against_the_machine
make install            # same as: uv sync --all-groups
```

The corpus is not stored in the repository. Place the vLLM 0.10.1 source tree
in `data/raw/`, so that the files end up under `data/raw/vllm-0.10.1/`. The
reference datasets are already in `data/datasets/`.

The first run also downloads the NLTK English stopword list, so it needs
network access once.

### Running

Every command is run as a module. [Python Fire](https://github.com/google/python-fire)
turns the function arguments into command-line flags.

| Command | What it does |
|---|---|
| `index` | Chunk the corpus and build the BM25 index (and the embeddings with `--bonus y`) |
| `search` | Print the top-k chunks for one question |
| `search_dataset` | Retrieve the top-k chunks for every question in a dataset and save them as JSON |
| `evaluate` | Compute recall@1…recall@k of a search-results file against a dataset |
| `answer` | Answer one question with the LLM |
| `answer_dataset` | Answer every question in a search-results file |
| `serve` | Start the HTTP API |

The index must be built once before any other command:

```bash
uv run python -m src index --max_chunk_size 800 --bonus n
```

The `Makefile` wraps each command in an interactive target that asks for its
parameters: `make index`, `make search`, `make search_dataset`,
`make evaluate`, `make answer`, `make answer_dataset` and `make server`.
`make lint` runs flake8 and mypy.

## System architecture

The pipeline has an offline part (indexing) and an online part (retrieval and
generation). The two communicate only through the files in `data/processed/`.

```
                          OFFLINE (index)
 data/raw/ ──► Indexer ──► ChunkerCode (.py, tree-sitter) ──┐
                       └─► ChunkOther (docs, text)  ────────┤
                                                            ▼
                             Tokenizer (stopwords + stemming)
                                                            ▼
                      BM25Okapi ──► data/processed/bm25_index.pkl
                      chunks    ──► data/processed/chunks.json
            (bonus) Embeddings  ──► data/processed/embeddings.npy

                          ONLINE (search / answer)
 question ──► Retrieval ──► BM25 scores × type booster
                        └─► (bonus) cosine scores ─► RRF fusion
                                                            ▼
                                     top-k chunks (file + offsets)
                                                            ▼
              PromptBuild ──► system prompt + numbered chunks + question
                                                            ▼
                        Model (Qwen3-0.6B) ──► answer
```

| Component | File | Role |
|---|---|---|
| `Indexer` | `src/indexer/indexer.py` | Walks the corpus, runs both chunkers, builds and saves the index |
| `ChunkerCode` / `ChunkOther` | `src/chunker/` | Split Python files and documentation into chunks |
| `Tokenizer` | `src/indexer/tokenizer.py` | Turns code and text into BM25 terms |
| `Embeddings` | `src/indexer/embeddings.py` | Dense vectors for the hybrid search (bonus) |
| `Retrieval` | `src/retriever/retrieval.py` | Ranks the chunks for a question |
| `PromptBuild`, `Model`, `Generator` | `src/generator/` | Build the prompt, load the LLM, generate answers |
| `Evaluate` | `src/evaluator/evaluate.py` | Recall@k against a reference dataset |
| `CacheHandler` | `src/cacher/cacher.py` | JSON caches for search results and answers (bonus) |
| FastAPI app | `src/server/api.py` | HTTP API over `Retrieval` and `Generator` (bonus) |

All data exchanged between steps (datasets, chunks, search results, answers)
is validated with Pydantic models defined in `src/entities/data_model.py`. A
chunk is identified by its file path and its start and end offsets, not by
its text. The text is read back from the source file when building a prompt.

## Chunking strategy

Python files and documentation are chunked differently, because their natural
boundaries are different. Chunks are limited to `max_chunk_size` (800 by
default and 2000 as maximum).

**Python files (`ChunkerCode`)** are parsed with
[tree-sitter](https://tree-sitter.github.io/tree-sitter/), and chunks follow
the top-level nodes of the syntax tree:

- All the `import` statements of a file go into a single chunk, so they do not
  produce dozens of tiny, nearly identical chunks.
- Every other top-level node (a function, a class, a statement) becomes one
  chunk. A complete function is a meaningful unit to retrieve.
- A node longer than the limit is cut at the last line break before the limit,
  so lines are never split in the middle when it can be avoided.

**Other text files (`ChunkOther`)**, mainly Markdown, use a sliding window:

- Each chunk ends at the last line break before the size limit.
- Consecutive chunks overlap by 100 characters, so a sentence on a boundary
  appears whole in at least one chunk.
- Binary files (images, archives, PDFs, libraries) are skipped, and so are
  files that are not valid UTF-8.

At tokenization time, chunks with fewer than 10 tokens are dropped: they are
mostly noise (a closing bracket, a lone heading) and only take up room in the
top-k results. With the default settings the corpus gives 42,480 chunks.

## Retrieval method

### BM25 (default)

The ranking is [BM25](https://en.wikipedia.org/wiki/Okapi_BM25) (Okapi
variant, from `rank_bm25`), with `k1 = 2.0` and `b = 0.3`.

**Tokenization.** Code chunks and text chunks go through different tokenizers:

- For code, `camelCase` and `PascalCase` names are split into words
  (`getHTTPResponse` → `get`, `http`, `response`), while `snake_case` names
  are kept whole (`max_tokens`).
- For text, any non-alphanumeric character separates words.
- In both cases, English stopwords are removed and the remaining tokens are
  reduced to their Snowball stem.
- The tokens of the chunk's file path are added to the chunk. A question that
  mentions "prefix caching" can then match a file called
  `prefix_caching.md` even if the chunk itself does not say it.

A question can contain identifiers as well as plain words, so it goes through
**both** tokenizers, and their tokens are merged without duplicates.

**Type booster.** About 78% of the chunks come from Python files. If one file
type makes up less than 30% of the index, the BM25 scores of its chunks are
multiplied by 1.2. On this corpus that boosts the `.md`, `.txt` and `.rst`
chunks, which would otherwise be outnumbered by code in the results.

The k chunks with the highest boosted score are returned, best first.

### Hybrid search (bonus)

With `--bonus y`, each chunk also has a 384-dimension embedding from
`all-MiniLM-L6-v2`:

- Chunks longer than 256 tokens are split into overlapping windows (stride
  of 32 tokens).
- Each window is mean-pooled.
- The windows of a chunk are averaged, and the vector is normalized.

The question is embedded the same way, and the two rankings are fused with
**Reciprocal Rank Fusion**:

```
score(chunk) = 1 / (60 + rank_BM25(chunk)) + 1 / (60 + rank_cosine(chunk))
```

RRF only uses ranks, so it does not need the BM25 and cosine scores to be on
the same scale.

### Generation

`Generator` takes the top 3 chunks of each question and sends them to
**Qwen3-0.6B** in this order:

1. A system prompt telling the model to answer only from the given context,
   and to say so when the context does not contain the answer.
2. The chunks, numbered and labelled with their file path.
3. The question.

Qwen3's thinking mode is disabled, and an answer is at most 200 new tokens.

## Performance analysis

### How recall@k is measured

For each question of a dataset, recall@k is the share of its reference
sources found among the first k retrieved chunks. A reference source counts as
found when a retrieved chunk comes from the same file and overlaps it with an
intersection over union (IoU) above 0.05. The reported value is the average
over all questions.

### Results

These were measured on the two public datasets in
`data/datasets/AnsweredQuestions/`, with `max_chunk_size = 800`, on an Apple
M1 Pro:

| k | Docs: BM25 | Docs: hybrid | Code: BM25 | Code: hybrid |
|---:|---:|---:|---:|---:|
| 1 | 69.0% | 49.0% | 30.3% | 32.3% |
| 3 | 84.0% | 69.0% | 46.5% | 48.5% |
| 5 | **90.0%** | 77.0% | **62.6%** | 59.6% |
| 10 | **94.0%** | 86.0% | **73.7%** | 68.7% |

The docs dataset has 100 questions and the code dataset 99.

**Documentation questions** reach 90% recall@5 with BM25. They tend to use the
same words as the documentation that answers them, which is the case where
BM25 does best.

**Code questions** are harder: 62.6% at k=5 and 73.7% at k=10. A question is
written in natural language ("What activation formats does the fused batched
MoE layer return?"), while the answer is a piece of code whose vocabulary is
identifiers. Many chunks of the same module share those identifiers, so the
right one is often ranked just below the cut.

**The hybrid search did not improve recall.** It is clearly worse on the docs
dataset, and only better on the code dataset for k ≤ 3. With equal weights,
RRF gives the semantic ranking as much say as BM25, and `all-MiniLM-L6-v2`,
which was trained on general English text, is a weaker signal here than exact
term matching. That is why BM25 stays the default and the hybrid search is an
option. Giving BM25 a larger weight in the fusion would be the next thing to
try.

### Speed

| Step | Time |
|---|---|
| Searching a whole dataset (~100 questions, BM25), including loading the index | 13–19 s |
| Searching a whole dataset with the hybrid search | ~10 s, plus loading the embeddings model |
| Computing the embeddings of the 42,480 chunks (`index --bonus y`) | ~2 min |
| Loading Qwen3-0.6B | ~10 s |
| Generating one answer | 1–2 s |

With the cache enabled, a question that was already searched or answered is
returned straight from disk, without running the search or the model again.

## Design decisions

- **BM25 first, embeddings as an option.** BM25 needs no GPU, is easy to
  inspect (you can see which terms matched), and works well on a technical
  corpus full of exact names. The measurements above confirmed this choice.
- **Syntax-aware chunking for code.** Cutting Python files at top-level nodes
  keeps functions and classes whole, which are the units a question is
  usually about. Fixed-size windows would split them in arbitrary places.
- **One chunk for all imports.** Without it, files with long import blocks
  produced many near-duplicate chunks that crowded the results.
- **Chunks stored as offsets.** Search results are `file + start + end`, which
  is the format the evaluator expects, and it keeps the index small. The text
  is read from the source file only when it is needed.
- **Maximum chunk size of 2000.** Larger chunks gave lower recall in our tests,
  probably because the evaluation uses IoU and a big chunk overlaps a small
  reference span less. Any value above 800 is capped.
- **Small local LLM.** Qwen3-0.6B runs on a laptop in about a second per
  answer. Since the answer must come from the retrieved context, the model
  mainly needs to read and rephrase, not to know a lot.
- **Safe cache keys.** A cached answer is only reused if the question, k, the
  model, the token limit and a hash of the system prompt plus context are all
  the same. Re-running `index` deletes both caches, because their entries
  point to chunks of the old index.
- **Validated data at every step.** Every JSON file read or written goes
  through a Pydantic model, so a malformed dataset fails early with a clear
  error instead of in the middle of a run.

## Challenges faced

- **Low recall at first, especially on documentation.** The results of the
  first BM25 version did not match the reference sources well. These changes
  brought the docs dataset to 80% recall@5, and later to 90%:
  - removing stopwords and stemming;
  - adding the file path tokens to each chunk;
  - dropping chunks with very few tokens;
  - tuning `k1` and `b`;
  - limiting the chunk size to 800;
  - adding the booster for the under-represented file type.
- **Code outnumbering documentation.** Python chunks are almost four times as
  many as documentation chunks, so for general questions the top results were
  filled with code. The type booster fixed this without hard-coding which
  type to favour: it is computed from the corpus itself.
- **Computing the embeddings.** The first attempt to embed the whole corpus
  at once made the system crash. Chunks are now encoded
  in batches of 64, and long chunks are processed as overlapping windows
  instead of being truncated.
- **Interrupted progress bars.** The output of the Hugging Face model loader
  broke the `tqdm` progress bar when the embeddings model was loaded during
  the search loop. It is now loaded before the bar starts.
- **Stale caches.** After re-indexing, cached results pointed to chunks that
  no longer existed. Indexing now deletes both caches, and the answer cache
  key includes everything that changes an answer.
- **Concurrent requests on the server.** FastAPI runs requests in a thread
  pool, so two generation requests could use the model at the same time. A
  lock now makes the generation endpoints run one at a time.

## Example usage

### Build the index

```bash
uv run python -m src index --max_chunk_size 800            # BM25 only
uv run python -m src index --max_chunk_size 800 --bonus y  # also embeddings
```

### Search for one question

```bash
uv run python -m src search "How do I enable prefix caching in vLLM?" --k 3
```

```
data/raw/vllm-0.10.1/docs/features/automatic_prefix_caching.md [0:698]
data/raw/vllm-0.10.1/examples/offline_inference/automatic_prefix_caching.py [107:722]
data/raw/vllm-0.10.1/docs/design/prefix_caching.md [0:748]
```

### Search a whole dataset and evaluate it

```bash
uv run python -m src search_dataset \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results

uv run python -m src evaluate \
    data/output/search_results/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
```

```
Questions evaluated: 100
Recall@1: 0.69 (69.00%)
...
Recall@5: 0.90 (90.00%)
...
Recall@10: 0.94 (94.00%)
```

Add `--bonus y` to `search` or `search_dataset` to use the hybrid search and
the cache.

### Answer a question

```bash
uv run python -m src answer "How do I enable prefix caching in vLLM?" --k 3
```

```
Automatic Prefix Caching (APC) allows the vLLM engine to reuse cached KV cache
blocks from previous prompts when a new query shares the same prefix. This
reduces redundant computation and improves inference speed.
```

### Answer a whole dataset

```bash
uv run python -m src answer_dataset \
    data/output/search_results/dataset_docs_public.json \
    data/output/search_results_and_answer
```

### HTTP API

```bash
uv run python -m src serve --port 8000 --bonus n
```

The interactive documentation is then at <http://127.0.0.1:8000/docs>.

```bash
curl -X POST http://127.0.0.1:8000/search \
     -H "Content-Type: application/json" \
     -d '{"query": "How do I enable prefix caching in vLLM?", "k": 3}'

curl -X POST http://127.0.0.1:8000/answer \
     -H "Content-Type: application/json" \
     -d '{"query": "What does the --tensor-parallel-size option do?", "k": 3}'
```

| Endpoint | Method | Input |
|---|---|---|
| `/health` | GET | none |
| `/search` | POST | JSON body `{"query": str, "k": 1–10}` |
| `/answer` | POST | JSON body `{"query": str, "k": 1–10}` |
| `/search_dataset` | POST | Query parameters `dataset_path`, `k`, `save_directory` |
| `/answer_dataset` | POST | Query parameters `student_search_results_path`, `save_directory` |

## Resources

### Documentation

- `rank_bm25`: <https://github.com/dorianbrown/rank_bm25>
- tree-sitter: <https://tree-sitter.github.io/tree-sitter/>, and its Python
  bindings: <https://tree-sitter.github.io/py-tree-sitter/>
- NLTK (stopwords, Snowball stemmer): <https://www.nltk.org/>
- Hugging Face Transformers: <https://huggingface.co/docs/transformers>
- Qwen3-0.6B model card: <https://huggingface.co/Qwen/Qwen3-0.6B>
- all-MiniLM-L6-v2 model card:
  <https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2>
- FastAPI: <https://fastapi.tiangolo.com/>
- Pydantic: <https://docs.pydantic.dev/>
- Python Fire: <https://github.com/google/python-fire>
- uv: <https://docs.astral.sh/uv/>

### Use of AI

AI assistants were used as a support tool. The design, the implementation and
the tuning of the pipeline were done by hand. AI was used for:

- **Learning concepts:** explanations of BM25 and its parameters, tree-sitter
  syntax trees, embeddings and mean pooling, Reciprocal Rank Fusion, and the
  Hugging Face APIs used to load and run the models.
- **Debugging:** understanding error messages and unexpected behaviour during
  development, for example the progress bar interrupted by model loading.
- **Docstrings and code review:** writing and checking the PEP 257 docstrings
  (Google style) of the modules in `src/`. While doing this, the assistant
  pointed out mismatches between the code and its documentation, and a few
  bugs, which were then reviewed and fixed by hand.
