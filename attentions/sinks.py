"""
Analysis over the sink scores written by pipeline/1_get_attention.py.
"""

import torch


def name_token_indices(tokens, name):
    """Token indices covering each occurrence of `name`.

    Substring-matching a single token misses names the tokenizer splits
    ("Heidi" -> "He" + "idi"), which is most of them once you sweep 200 names.
    Reconstructing the text from the decoded tokens and mapping character
    offsets back handles multi-token names and any tokenizer.

    Returns:
        list of occurrences, each a list of token indices.
    """
    text = ""
    spans = []
    for tok in tokens:
        spans.append((len(text), len(text) + len(tok)))
        text += tok

    occurrences = []
    start = text.find(name)
    while start != -1:
        end = start + len(name)
        occurrences.append([i for i, (a, b) in enumerate(spans) if a < end and b > start])
        start = text.find(name, start + 1)
    return occurrences


def name_alpha(record, name):
    """Sink score at the name's first mention, as a (layers, heads) tensor.

    Multi-token names are represented by their last token. Attention is causal,
    so only the final subword has seen the whole name; the earlier pieces ("He"
    of "Heidi") are ambiguous prefixes shared with other words, and taking a max
    over them lets an unrelated prefix sink stand in for the name. None if the
    name isn't found in the tokens.
    """
    occurrences = name_token_indices(record["tokens"], name)
    if not occurrences:
        return None
    return record["alpha"].float()[:, :, occurrences[0][-1]]


def find_name_sinks(out_dir, names, epsilon=0.1):
    """Count, per (layer, head), how many names the head sinks onto.

    A head counts as a name sink for one name only if the name's sink score
    exceeds epsilon under *every* prompt. Requiring all prompts is what
    separates a head that sinks on the name from one that sinks on a phrasing
    the name happens to sit in: a single prompt can't tell those apart.

    Prompts where the name could not be located are skipped rather than
    counted as failures, so "every prompt" means every prompt the name was
    found in. A name found in none of them contributes nothing.

    Returns:
        (counts, total) where counts is a (layers, heads) int tensor and total
        is the number of names that were located in at least one prompt.
    """
    counts, total = None, 0

    for name in names:
        path = out_dir / f"{name}.pt"
        if not path.exists():
            continue

        every_prompt = None
        for record in torch.load(path, weights_only=False)["results"]:
            alpha = name_alpha(record, name)
            if alpha is None:
                continue
            hit = alpha > epsilon
            every_prompt = hit if every_prompt is None else every_prompt & hit

        if every_prompt is None:
            continue
        if counts is None:
            counts = torch.zeros_like(every_prompt, dtype=torch.int32)
        counts += every_prompt.int()
        total += 1

    return counts, total


def top_heads(counts, total, k=None):
    """Rank (layer, head) pairs by the fraction of names they sink onto.

    Returns every head unless `k` is given.
    """
    frac = counts.float() / max(total, 1)
    out = []
    for pos in frac.flatten().argsort(descending=True)[:k]:
        layer, head = divmod(pos.item(), frac.shape[1])
        out.append({
            "layer": layer,
            "head": head,
            "count": int(counts[layer, head]),
            "frac": float(frac[layer, head]),
        })
    return out


def plot_attention(attentions, tokens, layer, head, name_idx=(), path=None):
    """Heatmap of one head's attention. `attentions` is one prompt's raw list of
    (heads, T, T) tensors, as recomputed by pipeline/3_compute_heatmaps.py.

    matplotlib is imported here rather than at module scope so that step 2,
    which only reads sink scores, does not pay for it.
    """
    import matplotlib.pyplot as plt

    attn = attentions[layer][head].float().numpy()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(attn, cmap="viridis", aspect="auto")

    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    for i in name_idx:
        ax.axvline(i, color="red", lw=0.6, alpha=0.6)

    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(f"Layer {layer}, Head {head}")

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=150)
        plt.close(fig)
    return fig
