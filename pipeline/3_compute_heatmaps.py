"""
Plot one head's full attention map for a single (name, prompt).

Step 1 keeps only the reduced sink scores, so the (T, T) attention has to be
recomputed here. That is a single forward pass, which is negligible next to
loading the model, and it keeps step 1's outputs T times smaller.

Usage:
    python pipeline/3_compute_heatmaps.py --context prompts_intro --config qwen25_0p5b \
        --layer 0 --head 6 --name Zachary
    python pipeline/3_compute_heatmaps.py --context prompts_intro --config qwen25_0p5b \
        --layer 0 --head 6 --name Zachary --prompt 3
    python pipeline/3_compute_heatmaps.py --context prompts_intro --config qwen25_0p5b \
        --layer 0 --head 6 --name Zachary --prompt "Introduce yourself to {name}."
"""

import argparse

from attentions.activations import AttentionExtractor, sink_scores
from attentions.models import PROJECT_ROOT, load_config, load_model
from attentions.sinks import name_token_indices, plot_attention


def read_lines(path):
    with open(PROJECT_ROOT / path, encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def resolve_prompt(spec, config):
    """Resolve --prompt to (template, label).

    Accepts an index into the config's prompts_file, or a literal template
    containing {name}. The label goes in the output filename so heatmaps from
    different prompts don't overwrite each other.
    """
    templates = read_lines(config["prompts_file"])
    if spec is None:
        return templates[0], "p0"
    if spec.isdigit():
        i = int(spec)
        if i >= len(templates):
            raise SystemExit(f"--prompt {i} out of range: {len(templates)} prompts in "
                             f"{config['prompts_file']}")
        return templates[i], f"p{i}"
    return spec, "custom"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--context", required=True,
                        help="Prompt set / config directory under experiments/")
    parser.add_argument("--config", required=True, help="Config name within the context")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--head", type=int, required=True)
    parser.add_argument("--name", required=True, help="Name to substitute into the prompt")
    parser.add_argument("--prompt", default=None,
                        help="Index into prompts_file, or a literal template with {name}")
    parser.add_argument("--out", default=None, help="Output PNG path (default: <output_dir>/heatmaps/)")
    args = parser.parse_args()

    config = load_config(args.config, args.context)
    template, label = resolve_prompt(args.prompt, config)
    prompt = template.format(name=args.name)
    print(f"Prompt: {prompt!r}")

    model, tokenizer = load_model(config)
    extractor = AttentionExtractor(model, tokenizer,
                                   use_chat_template=config.get("chat_template", True))
    attentions, tokens = extractor.compute_attention(prompt)

    # plot_attention indexes [layer][head], so drop the batch dim first
    raw = [a[0].float().cpu() for a in attentions]
    if not 0 <= args.layer < len(raw):
        raise SystemExit(f"--layer {args.layer} out of range: model has {len(raw)} layers")
    if not 0 <= args.head < raw[0].shape[0]:
        raise SystemExit(f"--head {args.head} out of range: model has {raw[0].shape[0]} heads")

    occurrences = name_token_indices(tokens, args.name)
    if occurrences:
        # Last token, matching the convention in sinks.name_alpha
        alpha = sink_scores(attentions)[args.layer, args.head, occurrences[0][-1]]
        marked = [tokens[i] for i in occurrences[0]]
        print(f"Name tokens {marked} at {occurrences[0]}, alpha={float(alpha):.4f}")
    else:
        print(f"Warning: {args.name!r} not found in the tokenization - no marker drawn")

    if args.out:
        path = PROJECT_ROOT / args.out
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        hm_dir = PROJECT_ROOT / config["output_dir"] / "heatmaps"
        hm_dir.mkdir(parents=True, exist_ok=True)
        path = hm_dir / f"L{args.layer:02d}H{args.head:02d}_{args.name}_{label}.png"

    plot_attention(raw, tokens, args.layer, args.head,
                   name_idx=occurrences[0] if occurrences else (), path=path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
