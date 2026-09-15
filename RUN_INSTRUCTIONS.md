# RUN_INSTRUCTIONS.md — NASA Mission Intelligence RAG System

Step-by-step instructions to set up, ingest data, chat, and evaluate this
project from scratch on your local machine.

All TODOs in `chat.py`, `embedding_pipeline.py`, `llm_client.py`,
`rag_client.py`, and `ragas_evaluator.py` have been completed. This guide
also covers a required fix to the `ragas` package (documented in the
original README / `fix_ragas.png`) and two files added to satisfy the
batch-evaluation rubric requirement: `test_questions.json` and
`batch_evaluate.py`.

---

## 0. Prerequisites

- Python 3.10–3.12
- An OpenAI API key with access to `gpt-3.5-turbo` (or another chat model)
  and `text-embedding-3-small`
- ~$0.05–$0.20 of OpenAI credit is enough to embed all the sample data and
  run the evaluation set (the corpus is small — 12 text files).

---

## 1. Create and activate a virtual environment

```bash
cd Project-NASA-Mission-Intelligence
python3 -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
venv\Scripts\Activate.ps1
```

## 2. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` includes everything needed, including three packages
(`sacrebleu`, `rouge_score`, `rapidfuzz`) that RAGAS's BLEU/ROUGE/Context
Precision metrics import lazily and will otherwise fail on with a
`ModuleNotFoundError` the first time you compute those metrics.

## 3. Apply the required RAGAS compatibility fix

The installed version of `ragas` still imports `ChatVertexAI`/`VertexAI`
from the now-relocated `langchain_community.chat_models.vertexai` module,
which raises `ModuleNotFoundError` the moment you `import ragas`. This is
a known incompatibility between `ragas==0.4.3` and current `langchain_*`
packages — not a bug in this project's code. Fix it by pointing those two
imports at `langchain_google_vertexai` instead.

**Important:** don't locate `base.py` with `import ragas` — that's the
exact line that's broken, so the import will fail before it can print the
path, `$RAGAS_BASE` will end up empty, and `sed` will fail with
`can't read : No such file or directory`. Use `importlib.util.find_spec`
instead, which locates the package on disk without executing its
(broken) `__init__.py`:

**macOS / Linux:**
```bash
RAGAS_BASE=$(python -c "import importlib.util, os; loc = importlib.util.find_spec('ragas').submodule_search_locations[0]; print(os.path.join(loc, 'llms', 'base.py'))")
echo "$RAGAS_BASE"   # sanity check: should print a real path, not be empty

sed -i.bak 's/from langchain_community.chat_models.vertexai import ChatVertexAI/from langchain_google_vertexai import ChatVertexAI/' "$RAGAS_BASE"
sed -i.bak 's/from langchain_community.llms import VertexAI/from langchain_google_vertexai import VertexAI/' "$RAGAS_BASE"
```

**Windows (PowerShell):**
```powershell
$RAGAS_BASE = python -c "import importlib.util, os; loc = importlib.util.find_spec('ragas').submodule_search_locations[0]; print(os.path.join(loc, 'llms', 'base.py'))"
Write-Host $RAGAS_BASE   # sanity check: should print a real path, not be empty

(Get-Content $RAGAS_BASE) -replace 'from langchain_community.chat_models.vertexai import ChatVertexAI', 'from langchain_google_vertexai import ChatVertexAI' | Set-Content $RAGAS_BASE
(Get-Content $RAGAS_BASE) -replace 'from langchain_community.llms import VertexAI', 'from langchain_google_vertexai import VertexAI' | Set-Content $RAGAS_BASE
```

Verify the fix worked:
```bash
python -c "import ragas; print('ragas OK:', ragas.__version__)"
```
If this prints a version number with no traceback, you're good. (See
`fix_ragas.png` in this folder for a screenshot of the same fix.)

## 4. Set your OpenAI API key

```bash
# macOS / Linux
export OPENAI_API_KEY="sk-...your-key..."

# Windows (PowerShell)
$env:OPENAI_API_KEY = "sk-...your-key..."
```

`chat.py`'s API-key field pre-fills from this environment variable, and
`batch_evaluate.py` / `embedding_pipeline.py` take it as a `--openai-key`
CLI argument (you can pass `$OPENAI_API_KEY` directly).

**⚠️ Using a classroom/Vocareum key instead of a personal OpenAI key?**
A Vocareum key does **not** need to start with `sk-` — that's fine. But
it will **only authenticate against Vocareum's own proxy endpoint**
(typically `https://openai.vocareum.com/v1`), not the real
`api.openai.com`. You must also point the OpenAI SDK at that endpoint,
or every request will silently go to the real OpenAI API and get
rejected. Two ways to do this:

```bash
# Option A: environment variable (works for chat.py, embedding_pipeline.py, batch_evaluate.py)
export OPENAI_BASE_URL="https://openai.vocareum.com/v1"
```
```bash
# Option B: pass it directly as a CLI flag (embedding_pipeline.py / batch_evaluate.py)
python embedding_pipeline.py --openai-key "$OPENAI_API_KEY" --base-url "https://openai.vocareum.com/v1" ...
```
In `chat.py`'s Streamlit UI, there's also a "Custom API Base URL
(optional)" field in the sidebar for this.

**Important naming gotcha:** only `OPENAI_BASE_URL` is auto-detected by
the current `openai` Python SDK. The older name `OPENAI_API_BASE` — still
used in some classroom setup guides — is silently ignored by the SDK
itself, though this project's code checks for both, so either name works
here. If you're integrating with other tools/notebooks outside this
project, use `OPENAI_BASE_URL`.

## 5. Build the vector database (ingest the NASA mission documents)

This reads every `.txt` file under `data_text/`, chunks it, embeds each
chunk with OpenAI, and stores it in a local ChromaDB directory.

```bash
python embedding_pipeline.py \
    --openai-key "$OPENAI_API_KEY" \
    --data-path ./data_text \
    --chroma-dir ./chroma_db_openai \
    --collection-name nasa_space_missions_text \
    --chunk-size 1000 \
    --chunk-overlap 200 \
    --update-mode skip
```

This will take a few minutes and makes roughly one OpenAI embedding API
call per **batch of chunks** (default 50 chunks/call), not one call per
chunk — this matters a lot on rate-limited keys.

**If you're on a classroom/Vocareum-provided key** and see errors like:
```
HTTP Request: POST https://openai.vocareum.com/v1/embeddings "HTTP/1.1 429 Too Many Requests"
INFO - Retrying request to /embeddings in 60.000000 seconds
```
this means the key's requests-per-minute limit is being hit. The pipeline
already batches many chunks into a single API call to minimize this, but
if you're still seeing 429s, lower `--batch-size` and add a
`--request-delay` (seconds to pause between embedding calls):

```bash
python embedding_pipeline.py \
    --openai-key "$OPENAI_API_KEY" \
    --data-path ./data_text \
    --chroma-dir ./chroma_db_openai \
    --collection-name nasa_space_missions_text \
    --chunk-size 500 \
    --chunk-overlap 200 \
    --update-mode skip \
    --batch-size 10 \
    --request-delay 2
```
The 60-second retries you see in the log are the OpenAI SDK's own
automatic backoff — the run isn't stuck, it's just slow. It's safe to
just let it finish, but the flags above will get you there much faster
by avoiding 429s in the first place. If it's interrupted partway through,
re-run the exact same command with `--update-mode skip` (the default) —
already-embedded chunks will be skipped, not re-embedded.

**Check it worked:**
```bash
python embedding_pipeline.py \
    --openai-key "$OPENAI_API_KEY" \
    --chroma-dir ./chroma_db_openai \
    --collection-name nasa_space_missions_text \
    --stats-only
```
You should see a document count and a breakdown by mission
(`apollo_11`, `apollo_13`, `challenger`).

**Re-running later?** Use `--update-mode`:
- `skip` (default) — only adds chunks that don't exist yet
- `update` — re-embeds and overwrites chunks that already exist
- `replace` — deletes all existing chunks for a file and re-adds them from scratch

## 6. Launch the chat app

```bash
streamlit run chat.py
```

This opens a browser tab (usually `http://localhost:8501`). In the app:

1. Confirm/enter your OpenAI API key in the sidebar (pre-filled from
   `OPENAI_API_KEY` if set).
2. Pick the ChromaDB backend/collection you just created
   (`chroma_db_openai / nasa_space_missions_text`).
3. Optionally set "Focus on mission" to restrict retrieval to one mission,
   and adjust "Documents to retrieve."
4. Ask a question, e.g. *"What caused the Challenger disaster?"* or
   *"How did the Apollo 13 crew survive the oxygen tank explosion?"*
5. Each answer shows live RAGAS evaluation metrics (Response Relevancy,
   Faithfulness) alongside the sources used.

## 7. Run batch evaluation against the test set

This runs all 7 questions in `test_questions.json` through the full
pipeline (retrieve → generate → evaluate) and prints per-question and
aggregate RAGAS scores, including BLEU/ROUGE/Context Precision since each
question has a reference answer.

```bash
python batch_evaluate.py \
    --openai-key "$OPENAI_API_KEY" \
    --chroma-dir ./chroma_db_openai \
    --collection-name nasa_space_missions_text \
    --test-file test_questions.json \
    --output evaluation_results.json
```

Review `evaluation_results.json` for the full report, and
`evaluation_dataset.txt` for the human-readable version of the same
question set with expected answers.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'` | You skipped Step 3 — apply the `ragas` patch. |
| `sed: can't read : No such file or directory` while applying the patch | `$RAGAS_BASE` was empty because it was located with `import ragas` (which fails). Re-run the Step 3 command exactly as written — it uses `importlib.util.find_spec` instead, which works even though `ragas` itself can't yet be imported. |
| `HTTP/1.1 429 Too Many Requests` during ingestion, with `Retrying request to /embeddings in 60.000000 seconds` | Your OpenAI key (often a classroom/Vocareum proxy key) has a low requests-per-minute limit. The pipeline already batches chunks into one API call per `--batch-size` chunks; lower `--batch-size` (e.g. `10`) and add `--request-delay 2` to slow down further. The 60s waits are the SDK's own retry logic — it will finish, just slowly, if you leave it running. |
| `401 Unauthorized` / `Incorrect API key provided` when using a classroom/Vocareum key | You also need to set the base URL, not just the key — see the "Using a classroom/Vocareum key" note in Step 4. A Vocareum key only works against Vocareum's proxy endpoint, not `api.openai.com`. |
| `ModuleNotFoundError: No module named 'sacrebleu'` (or `rouge_score`, `rapidfuzz`) | Run `pip install -r requirements.txt` again — these are in the file. |
| Chat app says "No relevant mission documents were retrieved" | Run Step 5 (ingestion) first, or check you selected the right collection in the sidebar. |
| `initialize_rag_system` returns `success=False` | Double check `--chroma-dir` / `--collection-name` match what you used during ingestion. |
| RAGAS metrics show `0.0` or errors | Confirm `OPENAI_API_KEY` is set in your shell (not just typed into the Streamlit UI — `ragas_evaluator.py` reads it from the environment). |

---

## Rubric self-check summary

- ✅ Chunking is configurable (`--chunk-size`, `--chunk-overlap`), never
  exceeds the configured size, and applies overlap consistently.
- ✅ Each chunk is embedded via OpenAI and stored with `mission`,
  `source`, `chunk_index`, etc. in metadata.
- ✅ `--update-mode skip|update|replace` all verified against a live
  ChromaDB collection.
- ✅ `--stats-only` prints collection size and per-mission breakdown.
- ✅ Retrieval issues real similarity queries with configurable top-k and
  optional mission-based metadata filtering (added to the chat UI).
- ✅ Context is deduplicated, source-attributed, and injected into a
  well-scoped system prompt with trimmed conversation history.
- ✅ LLM answers are grounded in context; the system prompt instructs the
  model to flag when context is insufficient.
- ✅ RAGAS computes Response Relevancy and Faithfulness always, plus
  BLEU/ROUGE/Context Precision when a reference answer is supplied.
  Context Precision is judged by the evaluator LLM (semantic relevance),
  not raw string overlap, since we only have short reference answers
  rather than hand-curated reference passages.
  Malformed/empty inputs return a clear error dict instead of crashing.
- ✅ `batch_evaluate.py` + `test_questions.json` run the full pipeline
  end-to-end and report per-question and aggregate metrics.
- ✅ `test_questions.json` / `evaluation_dataset.txt` cover all 6 required
  categories (overview, emergency, disaster analysis, crew, technical,
  timeline) across all 3 missions.
