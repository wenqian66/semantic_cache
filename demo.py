from semantic_cache import initialize_cache, process_query


def run_demo():
    query_index, global_metadata, sessions, session_embeddings = initialize_cache()

    print("Two-Stage Semantic Cache Demo")
    print("-" * 60)

    session1 = "user_001"
    queries1 = [
        "What is the impact of climate change on corn yields?",
        "How does global warming affect corn production?",
        "What about wheat?",
        "What is the impact of climate change on soybeans yields?",
    ]

    for i, q in enumerate(queries1, 1):
        result = process_query(
            session1, q, sessions, query_index, global_metadata,
            session_embeddings, threshold_stage1=0.90, threshold_stage2=0.85
        )

        print(f"\n[{i}] {q}")
        print(f"    Source: {result['source']}")
        if result['source'] == 'cache':
            print(f"    Similarity: {result['similarity']:.3f}")
        print(f"    Latency: {result['latency']:.2f}s")
        print(f"    Answer preview: {result['answer'][:200]}...")

    session2 = "user_002"
    queries2 = [
        "What are the benefits of renewable energy?",
        "How does solar power help the environment?",
        "What is the impact of climate change on corn yields?",
    ]

    print("\n" + "-" * 60)
    for i, q in enumerate(queries2, 1):
        result = process_query(
            session2, q, sessions, query_index, global_metadata,
            session_embeddings, threshold_stage1=0.90, threshold_stage2=0.85
        )

        print(f"\n[{i}] {q}")
        print(f"    Source: {result['source']}")
        if result['source'] == 'cache':
            print(f"    Similarity: {result['similarity']:.3f}")
        print(f"    Latency: {result['latency']:.2f}s")
        print(f"    Answer preview: {result['answer'][:200]}...")

    print("\n" + "=" * 60)
    print("\nCache Statistics:")
    total = len(queries1) + len(queries2)
    print(f"  Total queries: {total}")
    print(f"  LLM calls: {query_index.ntotal}")
    print(f"  Cache hits: {total - query_index.ntotal}")
    print(f"  Hit rate: {(total - query_index.ntotal) / total * 100:.1f}%")


def test_thresholds():
    """Test different threshold configurations"""
    print("\n" + "=" * 60)
    print("\nTesting Different Thresholds")
    print("-" * 60)

    configs = [
        {"stage1": 0.95, "stage2": 0.90, "name": "Strict"},
        {"stage1": 0.90, "stage2": 0.85, "name": "Balanced"},
        {"stage1": 0.85, "stage2": 0.80, "name": "Relaxed"},
    ]

    for config in configs:
        idx, meta, sess, sess_emb = initialize_cache()

        queries = [
            "What is machine learning?",
            "Explain machine learning",
            "Tell me about ML",
        ]

        llm_calls = 0
        for q in queries:
            result = process_query(
                "test", q, sess, idx, meta, sess_emb,
                threshold_stage1=config["stage1"],
                threshold_stage2=config["stage2"]
            )
            if result['source'] == 'llm':
                llm_calls += 1

        print(f"\n{config['name']} (θ1={config['stage1']}, θ2={config['stage2']}):")
        print(f"  LLM calls: {llm_calls}/{len(queries)}")
        print(f"  Hit rate: {(len(queries) - llm_calls) / len(queries) * 100:.0f}%")


if __name__ == "__main__":
    run_demo()
    test_thresholds()