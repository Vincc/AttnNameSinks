"""
Given a model, cache activations on name dataaset

Usage:
    python pipeline/1_get_attention.py --config configs/models/tinyllama_1b.yaml
"""

import argparse
import os
from pathlib import Path
from attentions.activations import AttentionExtractor
from attentions.models import load_config, load_model
PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to experiment config YAML")
    args = parser.parse_args()
    config = load_config(args.config)
    out_dir = PROJECT_ROOT / config["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"Loading model: {config['model_id']}")
    model, tokenizer = load_model(config)

    with open(config["prompts_file"]) as f:
        prompts = [line.strip() for line in f if line.strip()]
    with open(config["names_file"]) as f:
        names = [line.strip() for line in f if line.strip()]

    print(f"Extracting attentions for {len(prompts)} prompts...")
    extractor = AttentionExtractor(model, tokenizer)
    results = extractor.compute_all_attention(prompts, names, out_dir) #computes and saves activations
    #print(results)

if __name__ == "__main__":
    main()