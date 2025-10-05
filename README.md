**Semantic Cache (Job ID 6)**

Proof-of-concept semantic caching system that reduces LLM API costs through context-aware query matching.

---
## Setup

### Prerequisites
- Python 3.8+
- Gemini API key ([get free key](https://ai.google.dev/))
  
### Installation
```bash
git clone https://github.com/wenqian66/semantic_cache.git
cd semantic_cache
pip install -r requirements.txt
```

### Configuration

Create a `.env` file in the project root:

```dotenv
GEMINI_API_KEY=your_api_key_here
```


---

## Usage

### Quick Demo

```bash
python demo.py
```

### Full Evaluation

```bash
python evaluation.py
```

Processes 200 queries across 40 sessions (`evaluation_dataset.json`).

Runtime depends on API latency and rate limits.

Input: evaluation_dataset.json

Output: evaluation_results.json

### Parameters Evaluation

```bash
python paras_evaluate.py
```

Processes 56 queries across 18 sessions (`evaluation_dataset.json`).

Input: paras_dataset.json

Output: threshold_dim_comparison.json

---

## Project Structure

```text
semantic_cache/
├── semantic_cache.py       # main functions
├── evaluation.py           # Evaluation framework
├── evaluation_dataset.json # 200 test queries
├── evaluation_results.json # Results (θ1 = 0.90, θ2 = 0.85, mode = global)
├── utils.py                # Term extraction (not used at end)
├── demo.py
├── paras_evaluate.py
├── paras_dataset.json
├── threshold_dim_comparison.json
└── REPORT.md               # Detailed analysis (results)
```

---

## Implementation (summary)

### Architecture

* **Two-Stage Retrieval:** query-level filtering → context-aware matching, adapted from **ContextCache** (Yan et al., 2025).
* **Embedding Model:** `gemini-embedding-001` (768-dim, L2-normalized)
* **Model:** `gemini-2.5-flash-lite` 
* **Vector Store:** FAISS (inner product; unit vectors ≈ cosine)
* **Context Modeling:** weighted historical embeddings (decay = 0.5, window = 5)

### Parameters (defaults)

* `threshold_stage1 = 0.90`  (query similarity)
* `threshold_stage2 = 0.85`  (context similarity)
* `window = 5`               (recent turns)
* `k ≤ 10`                   (Stage-1 candidates)
* `mode = "global(default)"` or `"session"`

---

## Results (θ1 = 0.90, θ2 = 0.85, mode = global)

Numbers below are sample results; your run may vary

- Total queries: **187**
- Cache hits: **68**
- **Hit rate:** **36.4%**
- **Avg latency — LLM:** **11.72 s**
- **Avg latency — Cache:** **0.21 s**
- **Speedup (Cache vs LLM):** **55.7×**
- **LLM Calls Saved:** **68 / 187**

See `REPORT.md` for details.
