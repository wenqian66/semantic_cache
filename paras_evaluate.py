import os
import json

from evaluation import run_evaluation, print_comparison
import semantic_cache as sc

SMALL_DATASET = "paras_dataset.json"
SLEEP_S = 1
MODE = "global"# or "session"
DIMS = [256, 512, 768]

CONFIGS = [
    {"name": "Strict",   "stage1": 0.95, "stage2": 0.90},
    {"name": "Balanced", "stage1": 0.90, "stage2": 0.85},
    {"name": "Relaxed",  "stage1": 0.85, "stage2": 0.80},
]

def run_thresholds_for_dim(dim: int):
    sc.EMBEDDING_DIMENSION = dim
    results = []
    for cfg in CONFIGS:
        print("\n" + "=" * 50)
        print(f"Testing {dim}d - {cfg['name']} (θ1={cfg['stage1']}, θ2={cfg['stage2']}, mode={MODE})")
        out = run_evaluation(
            threshold_stage1=cfg["stage1"],
            threshold_stage2=cfg["stage2"],
            mode=MODE,
            dataset_path=SMALL_DATASET,
            sleep_s=SLEEP_S,
        )
        results.append({
            "name": f"{dim}d - {cfg['name']}",
            "config": {
                "dim": dim,
                "stage1": cfg["stage1"],
                "stage2": cfg["stage2"],
                "name": cfg["name"],
                "mode": MODE,
            },
            "metrics": out["metrics"],
        })
    return results

def compare_dims_thresholds():
    all_results = []
    for dim in DIMS:
        all_results.extend(run_thresholds_for_dim(dim))
    return all_results

def print_markdown_table(comparison):
    lines = []
    lines.append("| Dim | Setting | Hit rate | Speedup | LLM calls | Cache hits | Total |")
    lines.append("|-----|---------|----------|---------|-----------|------------|-------|")
    for item in comparison:
        m = item["metrics"]
        hit = m.get("hit_rate", 0.0)
        spd = m.get("speedup", 0.0)
        lines.append(
            f"| {item['config']['dim']} | {item['config']['name']} | "
            f"{hit:.1%} | {spd:.1f}x | {m.get('llm_calls',0)} | {m.get('cache_hits',0)} | "
            f"{m.get('total_queries',0)} |"
        )
    print("\n".join(lines))

if __name__ == "__main__":
    if not os.path.exists(SMALL_DATASET):
        print(f"Error: {SMALL_DATASET} not found")
        raise SystemExit(1)

    print("\n" + "=" * 50)
    print("Running dimension × threshold comparison...")
    comparison = compare_dims_thresholds()

    print_comparison(comparison)
    print("\nMarkdown table:\n")
    print_markdown_table(comparison)

    with open("threshold_dim_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    print("Saved: threshold_dim_comparison.json")
