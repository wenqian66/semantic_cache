# Semantic Cache

## 1. Thought Process

Demo example:

Q1: "What is the impact of climate change on corn yields?"

Q2: "What about wheat?"

Q3: "How does global warming affect corn production?"

### 1) embedding the whole conversation (failed)

At first, I embedded the current query together with 3 previous turns, then used this single embedding for similarity search.
However, even though Q2 is about wheat, it still matched the cached entry for corn with a similarity score above 0.92.
Raising the threshold didn’t help because the embedding was mixed with too much context, the historical “corn” content made “wheat” look similar.
In short, including the historical conversation and simple similarity filter caused semantic contamination and incorrect cache hits.

### 2) term-based filter (failed)

Next, I tried extracting keywords from each query (using a simple stemmer or spaCy) and compared them using Jaccard similarity.
This method successfully blocked cache hits between Q1 and Q2 (corn & wheat), but it also wrongly rejected Q3, which is actually semantically close to Q1.
The term filter reduced false positives but also blocked correct matches, which defeats the purpose of saving LLM calls.
So, using only a term filter was too strict and not reliable.

### 3) Final design: two-stage semantic matching (inspired by ContextCache)

To balance accuracy and efficiency, I designed a two-stage retrieval process inspired by the ContextCache paper.

Stage 1: Query-level retrieval

get_embedding(user_query) -> faiss.IndexFlatIP(EMBEDDING_DIMENSION)

Only the current user query is embedded (no conversation history).

FAISS retrieves top-k similar queries based on cosine similarity (threshold_stage1 = 0.90).

This ensures that only semantically similar questions (like Q1 and Q3) are considered, while unrelated ones (like Q2) are filtered out early.

Stage 2: Context-level validation
```
def compute_context_embedding(query_emb, history_embs, decay=0.5):
    if not history_embs:
        return query_emb

    all_embs = history_embs[-5:] + [query_emb]
    weights = np.array([decay ** (len(all_embs) - i - 1)
                        for i in range(len(all_embs))])
    weights = weights / weights.sum()

    context_emb = np.average(all_embs, axis=0, weights=weights)
    return context_emb / np.linalg.norm(context_emb)
```  
A context embedding is computed by exponentially weighting recent query embeddings (compute_context_embedding with decay=0.5).

Each candidate’s stored context embedding is compared with the new one (threshold_stage2 = 0.85).

A cache hit is confirmed only when both query-level and context-level similarities are above the thresholds.

## 2. System Design

### 2.1 Architecture

The system consists of four components:

1. **Session Manager** — maintains per-user conversation history.
2. **Context Encoder** — builds context-aware embeddings from the last *k* turns using decay-weighted attention.
3. **Semantic Cache** — stores responses and embeddings; retrieves via FAISS vector search.
4. **LLM Interface** — calls Gemini model when cache misses occur.

### 2.2 Workflow

1. Intercept user query and generate its embedding via Gemini Embeddings API.
2. Compute a **context-enhanced representation** combining the query embedding and prior turns using attention weights.
3. Perform similarity search in FAISS for the session and global caches.
4. If similarity ≥ threshold (τ₁, τ₂), return cached answer; otherwise call Gemini LLM.
5. Store new query–embedding–answer triplet into cache for future reuse.

### 2.3 Embedding Strategy

Each turn’s representation is:
[
g = \beta \cdot \text{AttentionWeighted}(H, q) + (1 - \beta)q
]
where *H* are historical embeddings and *q* is the current query embedding.
This ensures **semantic continuity** across related turns while minimizing contamination from unrelated context.

### 2.4 Vector Store

The project uses **FAISS (CPU)** due to its mature performance, fast cosine similarity search, and easy in-memory setup suitable for proof-of-concept evaluation.

## 3. Implementation Details

* **Language:** Python 3.8+
* **Dependencies:** `faiss-cpu`, `google-genai`, `numpy`, `python-dotenv`
* **Embeddings Model:** `models/embedding-001` from Google Gemini
* **Cache Storage:** In-memory FAISS index + metadata dictionary
* **Thresholds:** Two-stage comparison: session τ₁ and global τ₂
* **Context window:** Last 5 turns per session
* **Decay:** 0.7 per turn
* **Blending factor:** β = 0.6

## 4. Evaluation Setup

### 4.1 Dataset

A manually constructed evaluation set of **200+ queries across 40 sessions**, spanning domains such as:

* Agriculture and Climate
* Machine Learning
* Python Programming
* Health and Medicine
* Energy and Environment
* Math and Economics

### 4.2 Metrics

The system is evaluated on:

* **Cache Hit Rate** — proportion of reused responses.
* **LLM Calls Avoided** — reduction in total API requests.
* **Average Latency (Cache vs LLM)** — runtime comparison.
* **Speedup** — ratio of average LLM latency to cache latency.

### 4.3 Testing Conditions

Each configuration was tested under three threshold regimes:

* **Strict:** τ₁ = 0.95, τ₂ = 0.90
* **Balanced:** τ₁ = 0.90, τ₂ = 0.85
* **Relaxed:** τ₁ = 0.85, τ₂ = 0.80

and embedding dimensions **256, 512, 768**.

## 5. Results Summary

| Dim | Setting  | Hit rate | Speedup | LLM calls | Cache hits | Total |
| --- | -------- | -------- | ------- | --------- | ---------- | ----- |
| 256 | Strict   | 25.4%    | 19.2x   | 53        | 18         | 71    |
| 256 | Balanced | 57.7%    | 22.8x   | 30        | 41         | 71    |
| 256 | Relaxed  | 78.9%    | 23.4x   | 15        | 56         | 71    |
| 512 | Strict   | 12.7%    | 24.5x   | 62        | 9          | 71    |
| 512 | Balanced | 40.8%    | 18.0x   | 42        | 29         | 71    |
| 512 | Relaxed  | 63.4%    | 19.6x   | 26        | 45         | 71    |
| 768 | Strict   | 15.5%    | 20.6x   | 60        | 11         | 71    |
| 768 | Balanced | 43.7%    | 20.4x   | 40        | 31         | 71    |
| 768 | Relaxed  | 69.0%    | 19.8x   | 22        | 49         | 71    |

**Best Trade-off:** 256d (Balanced) — *57.7% hit rate, 22.8× latency reduction*.

**Aggregated metrics:**

* Total queries: 187
* Cache hits: 68
* LLM calls: 119
* Average LLM latency: 11.72 s
* Average cache latency: 0.21 s
* Overall speedup: **55.7×**

## 6. Analysis and Discussion

### 6.1 Threshold Sensitivity

Lower thresholds (Relaxed) improve reuse but risk semantic drift; stricter thresholds preserve accuracy at the cost of more LLM calls.
A balanced configuration (τ₁=0.9, τ₂=0.85) achieves the best performance–fidelity tradeoff.

### 6.2 Embedding Dimensionality

Higher dimensions (768d) did not significantly outperform 256d due to increased vector noise relative to the small dataset size.
This suggests diminishing returns for larger embeddings in low-data semantic caching tasks.

### 6.3 Scalability

The FAISS index supports sub-linear retrieval complexity (O(\log n)), but storing millions of sessions would require:

* Sharding by topic or user ID
* Periodic persistence to disk
* Asynchronous re-indexing

### 6.4 Eviction Strategy

A future extension could implement **LRU (Least Recently Used)** or **semantic distance-based** eviction to control memory usage without losing relevant cached entries.

## 7. Limitations and Extensions

### 7.1 Limitations

* Context decay and attention may lose relevance in long conversations with topic drift.
* Current system assumes a single-turn context scope (last 5 turns).
* No dynamic update of thresholds based on conversation domain or entropy.

### 7.2 Advanced Caching Proposal for AI Agents

For multi-agent reasoning frameworks (e.g., ReAct or Plan-Execute):

* **What to cache:** intermediate tool outputs, reasoning traces, and final responses.
* **Cache key:** embedding of *(user goal + current agent state + tool call signature)*.
* **Example:**

  ```
  Goal: “Summarize research papers on AI policy”
  State: “tool=ScholarSearch, query='AI regulation'”
  ```

  → Embed this combined context for reusing tool outputs across similar reasoning branches.

## 8. Conclusion

This proof-of-concept demonstrates that **context-aware semantic caching** can significantly reduce redundant LLM calls, yielding up to **23× latency improvement** and **nearly 80% reuse** under relaxed thresholds.
The architecture generalizes to any conversational system with minimal cost, serving as a scalable foundation for future **LLM efficiency optimization** in research and production environments.

---

