import json
from semantic_cache import initialize_cache, process_query
import time
import os

def load_dataset(filename='evaluation_dataset.json'):
    with open(filename, 'r') as f:
        return json.load(f)


def run_evaluation(threshold_stage1=0.90, threshold_stage2=0.85, mode='global',
                   dataset_path='evaluation_dataset.json',sleep_s=5):
    data = load_dataset(dataset_path)
    query_index, global_metadata, sessions, session_embeddings = initialize_cache()

    results = {
        'config': {'threshold_stage1': threshold_stage1, 'threshold_stage2': threshold_stage2, 'mode': mode},
        'metrics': {'total_queries': 0, 'llm_calls': 0, 'cache_hits': 0,
                    'total_latency_llm': 0.0, 'total_latency_cache': 0.0},
        'sessions': []
    }

    print(f"\nθ1={threshold_stage1}, θ2={threshold_stage2}, mode={mode}")

    for session_data in data['sessions']:
        session_id = session_data['session_id']
        session_result = {'session_id': session_id, 'queries': []}

        for i, query in enumerate(session_data['queries'], 1):
            result = process_query(session_id, query, sessions, query_index,
                                   global_metadata, session_embeddings,
                                   threshold_stage1, threshold_stage2, mode)
            time.sleep(sleep_s)

            session_result['queries'].append({
                'query': query,
                'source': result['source'],
                'similarity': float(result['similarity']),
                'latency': float(result['latency'])
            })

            results['metrics']['total_queries'] += 1
            if result['source'] == 'llm':
                results['metrics']['llm_calls'] += 1
                results['metrics']['total_latency_llm'] += result['latency']
            else:
                results['metrics']['cache_hits'] += 1
                results['metrics']['total_latency_cache'] += result['latency']

            print(f"[{session_id}] Q{i}: {result['source']} ({result['similarity']:.3f})")

        results['sessions'].append(session_result)

    m = results['metrics']
    if m['total_queries'] > 0:
        m['hit_rate'] = m['cache_hits'] / m['total_queries']
    if m['llm_calls'] > 0:
        m['avg_latency_llm'] = m['total_latency_llm'] / m['llm_calls']
    if m['cache_hits'] > 0:
        m['avg_latency_cache'] = m['total_latency_cache'] / m['cache_hits']
        m['speedup'] = m['avg_latency_llm'] / m['avg_latency_cache']

    return results



def print_summary(results):
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)

    m = results['metrics']
    cfg = results['config']

    print(f"\nConfig: θ1={cfg['threshold_stage1']}, θ2={cfg['threshold_stage2']}")
    print(f"\nQueries: {m['total_queries']}")
    print(f"LLM calls: {m['llm_calls']}")
    print(f"Cache hits: {m['cache_hits']}")
    print(f"Hit rate: {m.get('hit_rate', 0):.1%}")

    if 'avg_latency_llm' in m and 'avg_latency_cache' in m:
        print(f"\nLatency - LLM: {m['avg_latency_llm']:.2f}s, Cache: {m['avg_latency_cache']:.2f}s")
        if 'speedup' in m:
            print(f"Speedup: {m['speedup']:.1f}x")


def print_comparison(comparison):
    print("\n" + "=" * 50)
    print("THRESHOLD COMPARISON")
    print("=" * 50)

    for c in comparison:
        m = c['metrics']
        hit_rate = m.get('hit_rate', 0)
        speedup = m.get('speedup', 0)
        print(f"\n{c['name']}: {hit_rate:.1%} hit rate, {speedup:.1f}x speedup")


if __name__ == "__main__":
    if not os.path.exists('evaluation_dataset.json'):
        print("Error: evaluation_dataset.json not found")
        exit(1)

    results = run_evaluation(threshold_stage1=0.90, threshold_stage2=0.85)
    print_summary(results)

    with open('evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved: evaluation_results.json")

