"""
Rank attention heads by how consistently they sink onto the name token, and
save a heatmap for each of the top heads.

Usage:
    python pipeline/2_find_sinks.py --config phi3_med_4k_it --epsilon 0.1
"""

import argparse
import json

import torch

from attentions.models import PROJECT_ROOT, load_config
from attentions.sinks import find_name_sinks, name_alpha, name_token_indices, plot_attention, top_heads


def save_heatmaps(out_dir, heads):
    """Plot each head from the raw attention kept for the first name.

    Uses whichever prompt scores highest for that head, since that is the one
    that shows the sink most clearly.
    """
    raw_files = sorted((out_dir / "raw").glob("*.pt"))
    if not raw_files:
        print("No raw/ attention saved - rerun step 1 with --raw-names 1 for heatmaps")
        return

    name = raw_files[0].stem
    records = torch.load(out_dir / f"{name}.pt", weights_only=False)["results"]
    raw = torch.load(raw_files[0], weights_only=False)["attentions"]
    alphas = [name_alpha(r, name) for r in records]

    hm_dir = out_dir / "heatmaps"
    hm_dir.mkdir(exist_ok=True)
    for h in heads:
        scores = [a[h["layer"], h["head"]] if a is not None else -1 for a in alphas]
        i = max(range(len(scores)), key=lambda j: scores[j])
        occurrences = name_token_indices(records[i]["tokens"], name)
        path = hm_dir / f"L{h['layer']:02d}H{h['head']:02d}_{name}.png"
        plot_attention(raw[i], records[i]["tokens"], h["layer"], h["head"],
                       name_idx=occurrences[0] if occurrences else (), path=path)
    print(f"Saved {len(heads)} heatmaps to {hm_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name or path to YAML")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Sink threshold on the mean attention the name receives")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--heatmaps", type=int, default=10,
                        help="Save heatmaps for the top N heads (0 to skip)")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = PROJECT_ROOT / config["output_dir"]
    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)

    counts, total = find_name_sinks(out_dir, meta["names"], args.epsilon)
    if counts is None:
        raise SystemExit(f"No name tokens located in {out_dir}")

    heads = top_heads(counts, total, k=args.top_k)
    print(f"\n{meta['model_id']}  ({total} name x prompt pairs, epsilon={args.epsilon})")
    print(f"{'layer':>6} {'head':>5} {'frac':>7} {'count':>7}")
    for h in heads:
        print(f"{h['layer']:>6} {h['head']:>5} {h['frac']:>7.3f} {h['count']:>7}")

    path = out_dir / f"sinks_eps{args.epsilon}.json"
    with open(path, "w") as f:
        json.dump({"model_id": meta["model_id"], "epsilon": args.epsilon,
                   "total_pairs": total, "heads": heads}, f, indent=2)
    print(f"\nSaved to {path}")

    if args.heatmaps:
        save_heatmaps(out_dir, heads[:args.heatmaps])


if __name__ == "__main__":
    main()
