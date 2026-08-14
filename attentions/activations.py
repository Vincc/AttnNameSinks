import os
import torch

from tqdm.auto import tqdm


def sink_scores(attentions):
    """Reduce raw attention to the per-token sink statistic.

    alpha[layer, head, t] = mean over queries k >= t of A[k, t]

    i.e. the average attention token t receives from itself and every token
    after it. Attention is causal so A[k, t] == 0 for k < t, which makes the
    mean over k >= t just the column sum divided by (T - t).

    Args:
        attentions: tuple of (1, heads, T, T) tensors, one per layer.

    Returns:
        (layers, heads, T) float32 tensor.
    """
    T = attentions[0].shape[-1]
    counts = torch.arange(T, 0, -1, device=attentions[0].device, dtype=torch.float32)
    return torch.stack([a[0].float().sum(dim=-2) / counts for a in attentions])


class AttentionExtractor:
    def __init__(self, model, tokenizer, use_chat_template=True):
        self.model = model
        self.tokenizer = tokenizer
        # Base models have no chat template. Qwen base repos ship one anyway,
        # so this has to be configured rather than inferred.
        self.use_chat_template = use_chat_template and tokenizer.chat_template is not None

    def encode(self, prompt):
        """Tokenize a prompt, applying the chat template for instruct models.

        Templating and tokenizing in one call (rather than templating to a
        string and re-tokenizing) is what keeps BOS correct: templates disagree
        about whether they emit BOS themselves, so re-tokenizing gives Llama-3
        two. tokenize=True adds none, so we prepend exactly one here. BOS is
        *the* canonical attention sink, so getting this wrong shifts attention
        mass onto other tokens and inflates every score we measure.
        """
        if not self.use_chat_template:
            return self.tokenizer(prompt, return_tensors="pt")

        enc = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
        )
        bos = self.tokenizer.bos_token_id  # None for Qwen, which has no BOS
        if bos is not None and enc["input_ids"][0, 0] != bos:
            pad = torch.full((1, 1), bos, dtype=enc["input_ids"].dtype)
            enc["input_ids"] = torch.cat([pad, enc["input_ids"]], dim=1)
            mask = torch.ones((1, 1), dtype=enc["attention_mask"].dtype)
            enc["attention_mask"] = torch.cat([mask, enc["attention_mask"]], dim=1)
        return enc

    def compute_attention(self, prompt):
        """Run one forward pass and return (attentions, tokens).

        attentions is a tuple of num_layers tensors, each (1, heads, T, T).
        """
        inputs = self.encode(prompt)
        token_ids = inputs["input_ids"][0]
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)

        tokens = [self.tokenizer.decode(t) for t in token_ids]
    
        return outputs.attentions, tokens

    def compute_all_attention(self, prompts, names, out_dir, raw_names=1, resume=True):
        """Run every (name, prompt) pair, writing one .pt file per name.

        Only the reduced sink scores are kept.

        One file per name keeps a long run resumable: if the pod dies at name
        150 you keep the first 149, and `resume` skips them next launch.
        """

        for i, name in enumerate(tqdm(names, desc="names")):
            path = f"{out_dir}/{name}.pt"
            if resume and os.path.exists(path):
                continue
            results = []
            for template in prompts:
                prompt = template.format(name=name)
                attentions, tokens = self.compute_attention(prompt)
                results.append({
                    "prompt": prompt,
                    "template": template,
                    "tokens": tokens,
                    # (layers, heads, T), T times smaller than the raw attention
                    "alpha": sink_scores(attentions).half().cpu(),
                })
                
                del attentions

            torch.save({"name": name, "results": results}, path)

        print(f"Saved {len(names)} names to {out_dir}")
