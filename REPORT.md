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
def get_embedding(text, task_type="SEMANTIC_SIMILARITY",):
  ...
    # return the embedding from API gemini-embedding-001, then L2-normalized
    embedding = np.array(result.embeddings[0].values, dtype='float32')
    return embedding / np.linalg.norm(embedding)

q_emb = get_embedding(user_query) 
```

get_embedding(user_query) -> faiss.IndexFlatIP(EMBEDDING_DIMENSION)

Only the current user query is embedded (no conversation history) to avoid content blending.

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

The project uses **FAISS (CPU)** considering its mature performance, fast cosine similarity search, and easy in-memory setup suitable for evaluation.

## 3. Details

* **Language:** Python 3.8+
* **Embeddings Model:** `gemini-embedding-001` 
* **Cache Storage:** In-memory FAISS index + metadata dictionary
* **Thresholds:** Two-stage comparison: session τ₁ and global τ₂
* **Context window:** Last 5 turns per session
* **Mode:** Global mode by default, each session can access embeddings from previous sessions. Session mode restricts matching to within-session only


## 4. Evaluation Setup

### 4.1 Dataset

**evaluation_dataset.json**: 187 queries across 40 sessions.

A smaller dataset **paras_dataset.json** that contains **71 queries across 18 sessions**


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
### 5.1 threshold_dim_comparison.json
| Dim | Setting | Hit Rate | Speedup | LLM Calls | Cache Hits | Total Queries |
|:----|:---------|:----------|:----------|:------------|:--------------|:----------------|
| 256 | Strict | 0.316 | 26.54 | 39 | 18 | 57 |
| 256 | Balanced | 0.509 | 36.07 | 28 | 29 | 57 |
| 256 | Relaxed | 0.789 | 37.65 | 12 | 45 | 57 |
| 512 | Strict | 0.211 | 30.15 | 45 | 12 | 57 |
| 512 | Balanced | 0.351 | 3.23 | 37 | 20 | 57 |
| 512 | Relaxed | 0.596 | 28.84 | 23 | 34 | 57 |
| 768 | Strict | 0.246 | 27.48 | 43 | 14 | 57 |
| 768 | Balanced | 0.404 | 31.04 | 34 | 23 | 57 |
| 768 | Relaxed | 0.632 | 30.76 | 21 | 36 | 57 |
| 1536 | Strict | 0.193 | 27.08 | 46 | 11 | 57 |
| 1536 | Balanced | 0.368 | 30.40 | 36 | 21 | 57 |
| 1536 | Relaxed | 0.614 | 28.45 | 22 | 35 | 57 |
| 3072 | Strict | 0.246 | 28.70 | 43 | 14 | 57 |
| 3072 | Balanced | 0.439 | 31.99 | 32 | 25 | 57 |
| 3072 | Relaxed | 0.684 | 31.76 | 18 | 39 | 57 |


**Best Trade-off:** 256d (Balanced) — *57.7% hit rate, 22.8× latency reduction*. 
Based on latency and hit-rate metrics only; the actual quality of cached answers still needs manual or model-based accuracy checking.

### 5.2 evaluation_results.json
**Aggregated metrics for τ₁ = 0.90, τ₂ = 0.85, 768 dim**

* Total queries: 187
* Cache hits: 68
* LLM calls: 119
* Average LLM latency: 11.72 s
* Average cache latency: 0.21 s
* Overall speedup: **55.7×**

### 5.3 accuracy sample check
Best Accuracy: 256d-Balanced, 256d-Strict, 512d-Balanced 
For 256d-Balanced: mostly false positives (irrelevant cache hits)
For 256d-Strict, 512d-Balanced: mostly false negatives (missed reuse opportunities)
check accuracy for **threshold_dim_comparison.json** result

| Question | Query | Manual | 256d-Balanced | 256d-Relaxed | 256d-Strict | 512d-Balanced | 512d-Relaxed | 512d-Strict | 768d-Balanced | 768d-Relaxed | 768d-Strict | 1536d-Balanced | 1536d-Relaxed | 1536d-Strict | 3072d-Balanced | 3072d-Relaxed | 3072d-Strict |
|:----------|:-------|:--------|:---------------|:--------------|:-------------|:---------------|:--------------|:-------------|:---------------|:--------------|:-------------|:----------------|:---------------|:--------------|:----------------|:---------------|:--------------|
| **=== session_000 ===** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Q1 | "What is the impact of climate change on corn yields?" | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm |
| Q2 | "How does global warming affect corn production?" | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache |
| Q3 | "What factors influence maize crop productivity?" | llm | llm | cache | llm | llm | cache | llm | llm | cache | llm | llm | cache | llm | llm | cache | llm |
| Q4 | "Can you explain corn yield variations?" | cache | cache | cache | llm | llm | cache | llm | cache | cache | llm | llm | cache | llm | cache | cache | llm |
| Q5 | "What about wheat yields under climate change?" | llm | cache | cache | llm | cache | cache | llm | cache | cache | llm | cache | cache | llm | cache | cache | llm |
| **=== session_001 ===** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Q1 | "How does climate change affect wheat production?" | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache |
| Q2 | "What are the main threats to wheat farming?" | llm | llm | cache | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm |
| Q3 | "How do rising temperatures impact wheat?" | cache | cache | cache | llm | llm | cache | llm | llm | cache | llm | llm | cache | llm | llm | cache | llm |
| Q4 | "What about soybeans?" | llm | llm | cache | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm |
| Q5 | "What factors influence maize crop productivity?" | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache |
| **=== session_002 ===** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Q1 | "What is the impact of climate change on corn yields?" | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache |
| Q2 | "How does drought affect crop production?" | llm | cache | cache | llm | llm | cache | llm | llm | cache | llm | llm | cache | llm | llm | cache | llm |
| Q3 | "What irrigation methods work best?" | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm |
| Q4 | "Can technology help improve yields?" | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm |
| Q5 | "How does global warming affect corn production?" | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache | cache |
| **=== session_005 ===** |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |  |
| Q1 | "What is supervised learning?" | llm | cache | cache | llm | llm | cache | llm | cache | cache | llm | cache | cache | llm | cache | cache | llm |
| Q2 | "Explain supervised learning" | cache | cache | cache | cache | cache | cache | cache | llm | cache | cache | llm | cache | cache | llm | cache | cache |
| Q3 | "How does supervised ML work?" | cache | cache | cache | cache | cache | cache | llm | cache | cache | cache | cache | cache | llm | cache | cache | cache |
| Q4 | "What are some examples?" | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm | llm |
| Q5 | "What about unsupervised learning?" | llm | cache | cache | llm | llm | cache | llm | llm | llm | llm | llm | llm | llm | cache | cache | llm |

Complete version see [final_tabel.md](https://github.com/wenqian66/semantic_cache/blob/main/final_table.md)

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
The current version works well for small chats but not for long or complex ones. Because the system stored both the single embedding and combined embedding, and the index, for longer conversations, the system will overload.
  
Also, the system doesn’t check factual accuracy but only focus on semantic similarity, two questions might sound similar but need different answers. For example "What is the impact of climate change on corn yields?" and "What is the impact of climate change on wheat yields?" have high similarity because the sentence pattern is the same. The system might mistakenly return a cached answer about corn when the user is actually asking about wheat. This happens because embeddings capture question structure without distinguishing that corn and wheat are different. 

### 7.2 Caching Proposal for AI Agents

**Example**
The following approach and example is from paper *“Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching.”*  
In our example, we are not focusing on agent caching, so the template may not pretty useful here. I planed only apply a small, cost-efficient LLM model here to extract keywords, then stored them in cache and if the the keywords are found in cache, then treat as cache hit.

Example from Zhang et al.: "What is FY2019 working capital ratio for Costco?" 
->
keyword: working capital ratio
->
stored in cache

**Key insight**: Keyword matching outperforms semantic similarity for agent caching-avoids threshold tuning and false positives. However, I still think similarity should be used here, with the keyword algorithm serving as a supplement, since this is not an agent problem. Those context-specific details that were considered overemphasized in agent caching are still important in our problem.

**Trade-offs**: Effective for structured workflows; less useful for novel reasoning tasks.

Reference: 

arXiv:2506.14852

arXiv:2506.22791









