"""
Model config loading and model instantiation.

Usage:
    from attentions.models import load_config, load_model

    config = load_config("phi3_med_4k_it", "prompts_intro")
    model, tokenizer = load_model(config)
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "experiments"


def load_config(model_name, context):
    """Load experiments/<context>/<model_name>.yaml.

    Args:
        model_name: Short name like "tinyllama_1b".
        context: The prompt set the run belongs to, e.g. "prompts_intro". Every
            context holds the same model names against a different prompts_file,
            so the name alone does not identify a config and both are required.
    """
    context_dir = CONFIGS_DIR / context
    if not context_dir.is_dir():
        raise FileNotFoundError(
            f"Context not found: {context}\n"
            f"Available: {list_contexts()}"
        )

    path = context_dir / f"{model_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {context}/{model_name}\n"
            f"Available: {list_models(context)}"
        )

    with open(path) as f:
        return yaml.safe_load(f)


def resolve_device(device="auto"):
    """Resolve a config device string: "auto" means cuda if available, else cpu."""
    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype="auto", device="cpu"):
    """Resolve a config dtype string: "auto" means bfloat16 on GPU, float32 on CPU."""
    if dtype in (None, "auto"):
        return torch.bfloat16 if device.startswith("cuda") else torch.float32
    return getattr(torch, dtype)


def load_model(config):
    """Load model and tokenizer from a config dict.

    Returns:
        (model, tokenizer)
    """
    device = resolve_device(config.get("device", "auto"))
    dtype = resolve_dtype(config.get("dtype", "auto"), device)
    print(f"Loading {config['model_id']} on {device} ({dtype})")

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        dtype=dtype,
        attn_implementation="eager",  # need eager attention to get raw weights
    )
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def list_contexts():
    """List available context names, i.e. the config directories."""
    return sorted([p.name for p in CONFIGS_DIR.iterdir() if p.is_dir()])


def list_models(context):
    """List available model config names within a context."""
    return sorted([p.stem for p in (CONFIGS_DIR / context).glob("*.yaml")])
