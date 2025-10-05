from dotenv import load_dotenv
from google import genai
from google.genai import types
import os
import numpy as np
import faiss
import time


load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIM", "768"))


def get_embedding(text, task_type="SEMANTIC_SIMILARITY",):
    result = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type=task_type,
            output_dimensionality=EMBEDDING_DIMENSION
        )
    )
    embedding = np.array(result.embeddings[0].values, dtype='float32')
    return embedding / np.linalg.norm(embedding)


def build_context(history, query, window=5, max_len=7000):
    parts = []
    recent = history[-window:]
    for user_q, assistant_ans in recent:
        parts.append(f"User: {user_q}")
        parts.append(f"Assistant: {assistant_ans[:200]}")

    parts.append(f"User: {query}")
    result = "\n".join(parts)

    if len(result) > max_len:
        result = result[-max_len:]

    return result


#weighted?
def compute_context_embedding(query_emb, history_embs, decay=0.5):
    """Compute weighted context embedding from query and history"""
    if not history_embs:
        return query_emb

    all_embs = history_embs[-5:] + [query_emb]
    weights = np.array([decay ** (len(all_embs) - i - 1)
                        for i in range(len(all_embs))])
    weights = weights / weights.sum()

    context_emb = np.average(all_embs, axis=0, weights=weights)
    return context_emb / np.linalg.norm(context_emb)


def add_to_cache(query_emb, context_emb, query, answer, session_id,
                 query_index, global_metadata):
    query_index.add(query_emb.reshape(1, -1))
    global_metadata.append({
        'query': query,
        'answer': answer,
        'session_id': session_id,
        'context_embedding': context_emb,
        'timestamp': time.time()
    })


def call_llm(context, model="gemini-2.5-flash-lite", max_retries=3):
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=context,
                config=types.GenerateContentConfig(
                    temperature=0.7,
                    thinking_config=types.ThinkingConfig(thinking_budget=0)
                )
            )
            return response.text
        except Exception as e:
            if "503" in str(e) or "overloaded" in str(e).lower():
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 10  # 10s, 20s, 30s
                    print(f"API overloaded, waiting {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise


def initialize_cache():
    query_index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    global_metadata = []
    sessions = {}
    session_embeddings = {}
    return query_index, global_metadata, sessions, session_embeddings


def process_query(session_id, user_query, sessions, query_index,
                  global_metadata, session_embeddings,
                  threshold_stage1=0.90, threshold_stage2=0.85, mode='global'):
    start = time.time()

    #Get query embedding
    query_emb = get_embedding(user_query)

    if session_id not in session_embeddings:
        session_embeddings[session_id] = []

    #Compute context embedding
    history_embs = session_embeddings[session_id]
    context_emb = compute_context_embedding(query_emb, history_embs)

    #Stage 1: Query-level retrieval
    cached_answer = None
    best_sim = 0

    if query_index.ntotal > 0:
        k = min(10, query_index.ntotal)
        query_sims, query_inds = query_index.search(query_emb.reshape(1, -1), k=k)

        candidates = []
        for sim, idx in zip(query_sims[0], query_inds[0]):
            if idx != -1 and sim > threshold_stage1:
                candidates.append(idx)

        #Stage 2: Context-aware matching
        if candidates:
            for idx in candidates:
                meta = global_metadata[idx]

                if mode == 'session' and meta['session_id'] != session_id:
                    continue

                #Context similarity
                cached_context_emb = meta['context_embedding']
                context_sim = float(np.dot(context_emb, cached_context_emb))

                if context_sim > threshold_stage2 and context_sim > best_sim:
                    best_sim = context_sim
                    cached_answer = meta['answer']

    if cached_answer:
        source = "cache"
        answer = cached_answer
    else:
        history = sessions.get(session_id, [])
        context = build_context(history, user_query)
        answer = call_llm(context)

        add_to_cache(query_emb, context_emb, user_query, answer,
                     session_id, query_index, global_metadata)
        source = "llm"

    if session_id not in sessions:
        sessions[session_id] = []

    sessions[session_id].append((user_query, answer))
    session_embeddings[session_id].append(query_emb)

    latency = time.time() - start

    return {
        'answer': answer,
        'source': source,
        'similarity': best_sim,
        'latency': latency
    }