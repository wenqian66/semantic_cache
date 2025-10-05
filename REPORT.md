# Semantic Cache

## 1. Thought Process

Demo example:

Q1: "What is the impact of climate change on corn yields?"

Q2: "What about wheat?"

Q3: "How does global warming affect corn production?"

### 1) embedding the whole conversation (failed)

At first, I embedded the current query together with 3 previous turns, then used this simple embedding for similarity search.
However, even though Q2 is about wheat, it still matched the cached entry for corn with a similarity score above 0.92.
Raising the threshold didn’t help because the embedding was mixed with too much context, the historical "corn" content made "wheat" look similar.
In short, including the historical conversation and simple similarity filter caused semantic contamination and incorrect cache hits.

### 2) term-based filter (failed)

Next, I tried extracting keywords from each query (using a simple stemmer or spaCy) and compared them using Jaccard similarity.
See utils.py.
This method successfully blocked cache hits between Q1 and Q2 (corn & wheat), but it rejected Q3 by mistake, which is actually semantically close to Q1.
The term filter reduced false positives but also blocked correct matches, which defeats the purpose of saving LLM calls.
So, using only a term filter was too strict and not reliable.

### 3) Final design: two-stage semantic matching

To balance accuracy and efficiency, I applied a two-stage retrieval process inspired by the ContextCache paper.

#### Stage 1: Query-level retrieval

```python
  q_emb = get_embedding(user_query)  # L2-normalized
```

get_embedding(user_query) -> faiss.IndexFlatIP(EMBEDDING_DIMENSION)

Only the current user query is embedded (no conversation history) to avoid content bledding.

FAISS retrieves top-k similar queries based on cosine similarity. This ensures that only semantically similar questions (like Q1 and Q3) are considered, while unrelated ones (like Q2) are filtered out early.

```python
    if query_index.ntotal > 0:
            k = min(10, query_index.ntotal)
            query_sims, query_inds = query_index.search(query_emb.reshape(1, -1), k=k)

```
#### Stage 2: Context-level validation

  ```python
  def compute_context_embedding(query_emb, history_embs, decay=0.5):
      if not history_embs: 
          return query_emb
      seq = history_embs[-5:] + [query_emb]
      w = np.array([decay ** (len(seq) - i - 1) for i in range(len(seq))])
      w = w / w.sum()
      c = np.average(seq, axis=0, weights=w)
      return c / np.linalg.norm(c)
  ```

A context embedding is computed by exponentially weighting recent query embeddings. (compute_context_embedding with decay=0.5; history_embs[-5:]+query_emb)

A cache hit is confirmed only when both query-level and context-level similarities are above the thresholds.

* Confirm cache hit **only if** both pass: `query_sim > τ1` **and** `ctx_sim > τ2`.
* **Metadata:** answer, session_id, and the context embedding used when the answer was created (`context_embedding`), plus timestamp.
> Defaults: `threshold_stage1 = 0.90`, `threshold_stage2 = 0.85`, decay `= 0.5`, last `5` turns.


## 2. System Design

### 2.1 Architecture

The system consists of four components:

1. **Session Manager** — maintains per-user conversation history.
2. **Context Encoder** — builds context-aware embeddings from the last *k* turns using decay-weighted attention.
3. **Semantic Cache** — stores responses and embeddings; retrieves via FAISS vector search.
4. **LLM Interface** — calls Gemini model when cache misses occur. In this project, the function call_llm in semantic_cache.py uses gemini-2.5-flash-lite.

### 2.2 Workflow

1. Intercept user query and generate its embedding via Gemini Embeddings API.
2. Compute a **context-enhanced representation** combining the query embedding and prior turns using attention weights.
3. Perform similarity search in FAISS for the session and global caches.
4. If similarity ≥ threshold (τ₁, τ₂), return cached answer; otherwise call Gemini LLM.
5. Store new query–embedding–answer triplet into cache for future reuse.

### 2.3 Vector Store

The project uses **FAISS (CPU)** considering its mature performance, fast cosine similarity search, and easy in-memory setup suitable for POF evaluation.

## 3. Details

* **Language:** Python 3.8+
* **Embeddings Model:** `embedding-001` 
* **Cache Storage:** In-memory FAISS index + metadata dictionary
* **Thresholds:** Two-stage comparison: session τ₁ and global τ₂
* **Context window:** Last 5 turns per session
* **Mode:** Global mode by default， each session can access embeddings from previous sessions. This is useful for evaluation but would not be allowed in real-world applications due to privacy and data isolation concerns.


## 4. Evaluation Setup

### 4.1 Dataset

A manually constructed evaluation set of **200+ queries across 40 sessions**, see **evaluation_dataset.json**
A smaller dataset **paras_dataset.json** that contains **50+ sessions**

### 4.2 Metrics

The system is evaluated on:

* **Cache Hit Rate** 
* **LLM Calls Avoided** 
* **Average Latency (Cache vs LLM)** 
* **Speedup** : average LLM latency / cache latency.

### 4.3 Testing Conditions

Each configuration was tested under three threshold:

* **Strict:** τ₁ = 0.95, τ₂ = 0.90
* **Balanced:** τ₁ = 0.90, τ₂ = 0.85
* **Relaxed:** τ₁ = 0.85, τ₂ = 0.80

and embedding dimensions **256, 512, 768**.
All tests were run with top-k = 10 and decay = 0.5

## 5. Results Summary
### 5.1 
**threshold_dim_comparison.json**
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
Based on latency and hit-rate metrics only; the actual quality of cached answers still needs manual or model-based accuracy checking.

**Aggregated metrics for τ₁ = 0.90, τ₂ = 0.85, 768 dim**(evaluation_results.json)

* Total queries: 187
* Cache hits: 68
* LLM calls: 119
* Average LLM latency: 11.72 s
* Average cache latency: 0.21 s
* Overall speedup: **55.7×**

### 5.2 **threshold_dim_comparison.json** accuracy sample check

### Session 000 — Climate Change & Agriculture
| Q# | Query | Expected | 256d-Strict | 256d-Balanced | 256d-Relaxed | 512d-Balanced | 768d-Balanced |
|----|--------|-----------|--------------|----------------|---------------|----------------|----------------|
| Q1 | What is the impact of climate change on corn yields in the US? | llm | llm | llm | llm | llm | llm |
| Q2 | How does global warming affect corn production? | cache | cache | cache | cache | cache | cache |
| Q3 | What factors influence maize crop productivity? | cache | llm | llm | cache | llm | llm |
| Q4 | Can you explain corn yield variations? | cache | llm | cache | cache | llm | llm |
| Q5 | What about wheat yields under climate change? | llm | llm | cache | cache | cache | cache |

---

### Session 002 — Wheat Production
| Q# | Query | Expected | 256d-Strict | 256d-Balanced | 256d-Relaxed | 512d-Balanced | 768d-Balanced |
|----|--------|-----------|--------------|----------------|---------------|----------------|----------------|
| Q1 | How does climate change affect wheat production in North America? | llm | llm | cache | cache | cache | cache |
| Q2 | What are the main threats to wheat farming? | cache | llm | llm | llm | llm | llm |
| Q3 | How do rising temperatures impact wheat? | cache | llm | cache | cache | llm | llm |

---

### Session 005 — Supervised Learning
| Q# | Query | Expected | 256d-Strict | 256d-Balanced | 256d-Relaxed | 512d-Balanced | 768d-Balanced |
|----|--------|-----------|--------------|----------------|---------------|----------------|----------------|
| Q1 | What is supervised learning? | llm | llm | cache | cache | llm | cache |
| Q2 | Explain supervised learning | cache | cache | cache | cache | cache | llm |
| Q3 | How does supervised ML work? | cache | cache | cache | cache | cache | cache |
| Q4 | What are some examples? | cache | llm | llm | llm | llm | llm |
| Q5 | What about unsupervised learning? | llm | llm | cache | cache | llm | llm |


## 6. Analysis and Discussion

### 6.1 Threshold Sensitivity

The result clearly shows that lower thresholds (Relaxed: τ₁=0.85, τ₂=0.80) increase cache hits but introduces more false positives, such as cases where unrelated questions are incorrectly matched.
In contrast, stricter thresholds minimize such mismatches but cause unnecessary LLM calls, increasing cost.

### 6.2 Embedding Dimensionality

Higher dimensions (768d) did not significantly outperform 256d due to increased vector noise relative to the small dataset size.
This suggests diminishing returns for larger embeddings in low-data semantic caching tasks.

### 6.3 Scalability

The cache currently runs fully in memory, storing the FAISS index, text, answers, and session info. This works for small datasets but doesn’t scale. The IndexFlatIP compares every new query with all stored vectors, which becomes slow as data grows.

Possible improvements include using approximate indexing or sharding by topic/user to handle larger datasets efficiently.


### 6.4 Eviction Strategy

In the current version, the cache keeps growing without limits. We can use an LFU (Least Frequently Used) strategy with time decay to handle this issue. This would track how often and how recently each entry is used, then remove those that are old or rarely accessed.

Another possible way is to assign a TTL (time-to-live) based on topic, shorter for fast-changing content like news, and longer for stable or factual information.

## 7. Limitations and Extensions

### 7.1 Limitations

The current version works well for short chats but not for long or complex ones.
Because the cache only looks at the last five turns, it quickly loses track of older context.
For example, if a user talks about topic A in turns 1–4, switches to topic B in turns 5–9, and then comes back to topic A at turn 10, the cache won’t recognize it.
By that point, the earlier turns are forgotten, the context embedding is dominated by topic B, and even a previously cached answer about topic A won't be reused.
  
Also, the system doesn’t check factual accuracy but only focus on semantic similarity, two questions might sound similar but need different answers. For example "What is the impact of climate change on corn yields?" and "What is the impact of climate change on wheat yields?" have high similarity because the sentence pattern is the same. The system might mistakenly return a cached answer about corn when the user is actually asking about wheat. This happens because embeddings capture question structure without distinguishing that corn and wheat are different. 

### 7.2 Caching Proposal for AI Agents




