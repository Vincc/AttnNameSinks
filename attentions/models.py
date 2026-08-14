"""
Model config loading and model instantiation.

Usage:
    from attentions.models import load_config, load_model

    config = load_config("phi3_med_4k_it")
    model, tokenizer = load_model(config)
"""

import yaml
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).parent.parent
CONFIGS_DIR = PROJECT_ROOT / "experiments"


def load_config(model_name):
    """
        model_name: Either a short name like "tinyllama_1b" (looks up in experiments/)
                    or a full path to a YAML file.
    """
    path = Path(model_name)
    if not path.exists():
        path = CONFIGS_DIR / f"{model_name}.yaml"

    if not path.exists():
        raise FileNotFoundError(
            f"Config not found: {model_name}\n"
            f"Available: {list_models()}"
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


def list_models():
    """List available model config names."""
    return sorted([p.stem for p in CONFIGS_DIR.glob("*.yaml")])
