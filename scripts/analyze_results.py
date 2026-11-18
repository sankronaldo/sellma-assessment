"""
Post-run analysis and figure generation.

Reads results/summary.json and per-condition JSONL files, generates:
  - results/figures/lang_id_comparison.png
  - results/figures/stratification.png
  - results/metrics_table.csv
  - Prints qualitative examples to stdout

Run after run_experiment.py completes.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def load_summary() -> dict:
    p = RESULTS_DIR / "summary.json"
    if not p.exists():
        raise FileNotFoundError(f"summary.json not found at {p}. Run run_experiment.py first.")
    with open(p) as f:
        return json.load(f)


def save_metrics_csv(summary: dict) -> None:
    import csv
    rows = []
    for cond, m in summary.items():
        rows.append({
            "condition": cond,
            "lang_id_accuracy": m.get("lang_id_accuracy", ""),
            "lang_id_ci_lower": m.get("lang_id_ci_lower", ""),
            "lang_id_ci_upper": m.get("lang_id_ci_upper", ""),
            "perplexity_sv": m.get("perplexity_sv", ""),
            "english_lang_id": m.get("english_lang_id", ""),
            "stratum_A_n": m.get("stratum_A", {}).get("n", ""),
            "stratum_A_edit_accuracy": m.get("stratum_A", {}).get("edit_accuracy", ""),
            "stratum_B_n": m.get("stratum_B", {}).get("n", ""),
            "stratum_B_edit_accuracy": m.get("stratum_B", {}).get("edit_accuracy", ""),
        })
    csv_path = RESULTS_DIR / "metrics_table.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Metrics table saved to {csv_path}")


def plot_lang_id_comparison(summary: dict) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = []
    accs = []
    ci_lo = []
    ci_hi = []

    for cond, m in summary.items():
        if "lang_id_accuracy" not in m:
            continue
        conditions.append(cond.replace("_", "\n"))
        acc = m["lang_id_accuracy"]
        accs.append(acc)
        ci_lo.append(acc - m.get("lang_id_ci_lower", acc))
        ci_hi.append(m.get("lang_id_ci_upper", acc) - acc)

    if not conditions:
        print("No lang_id_accuracy data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(conditions))
    colors = ["#4c72b0"] * len(conditions)
    ax.bar(x, [a * 100 for a in accs],
           yerr=[np.array(ci_lo) * 100, np.array(ci_hi) * 100],
           color=colors, capsize=5, error_kw={"linewidth": 1.5})
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=8)
    ax.set_ylabel("Language-ID Accuracy (%)")
    ax.set_title("Swedish Language-ID Accuracy by Condition\n(test set, with 95% bootstrap CI)")
    ax.set_ylim(0, 110)
    ax.axhline(y=accs[0] * 100 if accs else 0, color="gray", linestyle="--", linewidth=1,
               label="C0 baseline")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path = FIGURES_DIR / "lang_id_comparison.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Figure saved to {out_path}")


def plot_stratification(summary: dict) -> None:
    """Bar chart showing stratum A vs B for C3 (failure removal analysis)."""
    import matplotlib.pyplot as plt
    import numpy as np

    c3 = summary.get("C3_direction_ablation", {})
    sa = c3.get("stratum_A", {})
    sb = c3.get("stratum_B", {})

    if not sa and not sb:
        print("No stratification data for C3.")
        return

    labels = ["Stratum A\n(base→Swedish)", "Stratum B\n(base→English/drift)"]
    base_accs = [
        sa.get("base_accuracy", 0) * 100,
        sb.get("base_accuracy", 0) * 100,
    ]
    edit_accs = [
        sa.get("edit_accuracy", 0) * 100,
        sb.get("edit_accuracy", 0) * 100,
    ]
    ns = [sa.get("n", 0), sb.get("n", 0)]

    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, base_accs, width, label="Base (C0)", color="#4c72b0")
    ax.bar(x + width / 2, edit_accs, width, label="Edit (C3)", color="#dd8452")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{l}\n(n={n})" for l, n in zip(labels, ns)], fontsize=9)
    ax.set_ylabel("Swedish Language-ID Accuracy (%)")
    ax.set_title("Failure Removal vs New Capability\nStratum A: already Swedish | Stratum B: drifted to English")
    ax.set_ylim(0, 110)
    ax.legend()
    plt.tight_layout()
    out_path = FIGURES_DIR / "stratification.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"Stratification figure saved to {out_path}")


def print_examples(n: int = 10) -> None:
    """Print sample successes and failures for C0 and C3."""
    import jsonlines
    examples_dir = RESULTS_DIR / "examples"
    for cond in ["C0_baseline", "C3_direction_ablation"]:
        for kind in ["successes", "failures"]:
            fp = examples_dir / f"{cond}_{kind}.jsonl"
            if not fp.exists():
                continue
            print(f"\n--- {cond} {kind.upper()} (first {n}) ---")
            with jsonlines.open(str(fp)) as reader:
                for i, rec in enumerate(reader):
                    if i >= n:
                        break
                    print(f"  ID: {rec.get('id', '?')}")
                    print(f"  RESPONSE: {rec.get('response', '')[:120]}")
                    print()


def main():
    summary = load_summary()
    save_metrics_csv(summary)
    plot_lang_id_comparison(summary)
    plot_stratification(summary)
    print_examples(n=10)
    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
