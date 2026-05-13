import os
import torch

class AttentionExtractor:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer


    def compute_attention(self, prompt, debug = False):
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)

        # Forward pass with attention outputs
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)


        attentions = outputs.attentions  # tuple of len num_layers
        # outputs.attentions is a tuple of num_layers tensors
        # each tensor shape: (batch, num_heads, seq_len, seq_len)

        if debug:
            print(f"Num layers: {len(attentions)}")
            print(f"Attention shape per layer: {attentions[0].shape}")
            # (1, num_heads, seq_len, seq_len)

        return attentions

    def compute_all_attention(self, prompts, names, out_dir):
        results_all_names = []
        for name in names:
            results = []
            subbed_prompts = [p.format(name=name) for p in prompts]
            for prompt in subbed_prompts:
                attentions = self.compute_attention(prompt)
                token_ids = self.tokenizer(
                    self.tokenizer.apply_chat_template(
                        [{"role": "user", "content": prompt}],
                        tokenize=False, add_generation_prompt=True
                    ),
                    return_tensors="pt"
                )["input_ids"][0]
                tokens = [self.tokenizer.decode(t) for t in token_ids]

                results.append({
                    "prompt": prompt,
                    "tokens": tokens,
                    "attentions": [a.cpu() for a in attentions],
                })
            os.makedirs(f"{out_dir}/{name}", exist_ok=True)
            torch.save(results, f"{out_dir}/{name}/attention_results.pt")
            print(f"Saved to {out_dir}/attention_results.pt")
            results_all_names.append({name: results})
        return results_all_names



