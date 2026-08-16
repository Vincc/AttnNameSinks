"""
Given a model, cache attention sink scores over the name dataset.

Usage:
    python pipeline/1_get_attention.py --context prompts_intro --config phi3_med_4k_it
    python pipeline/1_get_attention.py --context prompts_intro_long --config qwen25_7b
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
    parser.add_argument("--context", required=True,
                        help="Prompt set / config directory under experiments/")
    parser.add_argument("--config", required=True, help="Config name within the context")
    parser.add_argument("--no-resume", action="store_true",
                        help="Recompute names that already have an output file")
    args = parser.parse_args()

    config = load_config(args.config, args.context)
    out_dir = PROJECT_ROOT / config["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    prompts = read_lines(config["prompts_file"])
    names = read_lines(config["names_file"])

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
    extractor.compute_all_attention(prompts, names, out_dir, resume=not args.no_resume)


if __name__ == "__main__":
    main()
