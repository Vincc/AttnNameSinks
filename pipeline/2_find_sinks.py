"""
Rank attention heads by how consistently they sink onto the name token.

Usage:
    python pipeline/2_find_sinks.py --context prompts_intro --config phi3_med_4k_it --epsilon 0.1
"""

import argparse
import json

from attentions.models import PROJECT_ROOT, load_config
from attentions.sinks import find_name_sinks, top_heads

PRINT_ROWS = 20


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True,
                        help="Prompt set / config directory under experiments/")
    parser.add_argument("--config", required=True, help="Config name within the context")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Sink threshold on the mean attention the name receives")
    args = parser.parse_args()

    config = load_config(args.config, args.context)
    out_dir = PROJECT_ROOT / config["output_dir"]
    with open(out_dir / "metadata.json") as f:
        meta = json.load(f)

    counts, total = find_name_sinks(out_dir, meta["names"], args.epsilon)
    if counts is None:
        raise SystemExit(f"No name tokens located in {out_dir}")

    heads = top_heads(counts, total)
    print(f"\n{meta['model_id']}  ({total} names x {len(meta['prompts'])} prompts, "
          f"epsilon={args.epsilon}, sink required on every prompt)")
    print(f"{'layer':>6} {'head':>5} {'frac':>7} {'count':>7}")
    # The full ranking goes to the JSON; the terminal only needs the head of it.
    for h in heads[:PRINT_ROWS]:
        print(f"{h['layer']:>6} {h['head']:>5} {h['frac']:>7.3f} {h['count']:>7}")
    if len(heads) > PRINT_ROWS:
        print(f"... {len(heads) - PRINT_ROWS} more heads in the JSON")

    path = out_dir / f"sinks_eps{args.epsilon}.json"
    with open(path, "w") as f:
        json.dump({"model_id": meta["model_id"], "epsilon": args.epsilon,
                   "total_names": total, "heads": heads}, f, indent=2)
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    main()
