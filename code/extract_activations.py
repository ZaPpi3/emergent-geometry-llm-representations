"""Phase 1: Substrate Extraction.

Registers a forward hook on a target decoder layer and runs each probe token
through the model, capturing the mid-layer hidden-state activation vector.
Output feeds Phase 2 (relational graph construction).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
DEFAULT_LAYER = 12


def load_probe_tokens(path: Path) -> tuple[list[str], list[str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    tokens, categories = [], []
    for category, items in data.items():
        for item in items:
            tokens.append(item)
            categories.append(category)
    return tokens, categories


def extract_activations(model_name: str, layer_idx: int, tokens: list[str], device: str,
                         load_in_4bit: bool = False) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config,
                                                      device_map={"": 0})
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
        model.to(device)
    model.eval()

    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["hidden"] = hidden.detach()

    target_layer = model.model.layers[layer_idx]
    handle = target_layer.register_forward_hook(hook)

    vectors = []
    token_counts = []
    with torch.no_grad():
        for token in tokens:
            inputs = tokenizer(token, return_tensors="pt").to(device)
            model(**inputs)
            hidden = captured["hidden"][0]  # (seq_len, hidden_dim)
            # Last-token position, not mean-pool: with causal attention it has already
            # attended to the whole prompt, so it's a fair comparison regardless of how
            # many BPE tokens a given probe happens to split into.
            vectors.append(hidden[-1].float().cpu().numpy())
            # count content tokens only, excluding any BOS/special tokens the
            # tokenizer adds by default (Mistral/Llama-style tokenizers prepend
            # a BOS, Qwen's does not) so "single token" means the same thing everywhere
            content_ids = tokenizer(token, add_special_tokens=False)["input_ids"]
            token_counts.append(len(content_ids))

    handle.remove()
    return np.stack(vectors), np.array(token_counts)


def main():
    parser = argparse.ArgumentParser(description="Phase 1: extract mid-layer activations for probe tokens")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--probe-file", default="probe_tokens.json")
    parser.add_argument("--out", default="activations.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-in-4bit", action="store_true")
    args = parser.parse_args()

    tokens, categories = load_probe_tokens(Path(args.probe_file))
    print(f"Loaded {len(tokens)} probe tokens across {len(set(categories))} categories")

    vectors, token_counts = extract_activations(args.model, args.layer, tokens, args.device,
                                                 load_in_4bit=args.load_in_4bit)
    print(f"Extracted activations: {vectors.shape}")

    np.savez(
        args.out,
        vectors=vectors,
        tokens=np.array(tokens, dtype=object),
        categories=np.array(categories, dtype=object),
        token_counts=token_counts,
        model=args.model,
        layer=args.layer,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
