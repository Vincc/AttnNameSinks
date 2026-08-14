"""
Analysis over the sink scores written by pipeline/1_get_attention.py.
"""

import matplotlib.pyplot as plt
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

    Multi-token names are reduced with max over their tokens. None if the name
    isn't found in the tokens.
    """
    occurrences = name_token_indices(record["tokens"], name)
    if not occurrences:
        return None
    return record["alpha"].float()[:, :, occurrences[0]].max(dim=-1).values


def find_name_sinks(out_dir, names, epsilon=0.1):
    """Count, per (layer, head), how often the name token is an attention sink.

    A head counts as a name sink for one (name, prompt) pair when the name's
    sink score exceeds epsilon.

    Returns:
        (counts, total) where counts is a (layers, heads) int tensor and total
        is the number of pairs in which the name token was located.
    """
    counts, total = None, 0

    for name in names:
        path = out_dir / f"{name}.pt"
        if not path.exists():
            continue
        for record in torch.load(path, weights_only=False)["results"]:
            alpha = name_alpha(record, name)
            if alpha is None:
                continue
            if counts is None:
                counts = torch.zeros_like(alpha, dtype=torch.int32)
            counts += (alpha > epsilon).int()
            total += 1

    return counts, total


def top_heads(counts, total, k=20):
    """Rank (layer, head) pairs by how consistently they sink onto the name."""
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
    (heads, T, T) tensors, as saved in raw/."""
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
