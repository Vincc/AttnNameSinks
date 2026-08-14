"""
Given a model, cache attention sink scores over the name dataset.

Usage:
    python pipeline/1_get_attention.py --config phi3_med_4k_it
    python pipeline/1_get_attention.py --config experiments/qwen25_7b.yaml --max-names 5
"""

import argparse
import json

from attentions.activations import AttentionExtractor
from attentions.models import PROJECT_ROOT, load_config, load_model


def read_lines(path):
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Config name or path to YAML")
    parser.add_argument("--max-names", type=int, default=None,
                        help="Only run the first N names")
    parser.add_argument("--raw-names", type=int, default=1,
                        help="Keep full attention for the first N names, for heatmaps")
    parser.add_argument("--no-resume", action="store_true",
                        help="Recompute names that already have an output file")
    args = parser.parse_args()

    config = load_config(args.config)
    out_dir = PROJECT_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_lines(config["prompts_file"])
    names = read_lines(config["names_file"])
    if args.max_names:
        names = names[:args.max_names]

    model, tokenizer = load_model(config)
    extractor = AttentionExtractor(model, tokenizer,
                                   use_chat_template=config.get("chat_template", True))
    if not extractor.use_chat_template:
        print("No chat template (base model) - feeding raw prompts")

    with open(out_dir / "metadata.json", "w") as f:
        json.dump({
            "model_id": config["model_id"],
            "use_chat_template": extractor.use_chat_template,
            "prompts": prompts,
            "names": names,
        }, f, indent=2)

    print(f"{len(names)} names x {len(prompts)} prompts = {len(names) * len(prompts)} forward passes")
    extractor.compute_all_attention(prompts, names, out_dir,
                                    raw_names=args.raw_names, resume=not args.no_resume)


if __name__ == "__main__":
    main()
