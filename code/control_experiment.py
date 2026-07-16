"""Hard Constraint: the g=0 Decoupling Control.

Runs the identical Phase 1-3 pipeline on randomized vocabulary tokens instead
of the structured semantic probe set. If the earlier structure is real signal
(not a pipeline artifact), this run's spectrum should collapse toward a flat,
featureless noise envelope with no comparable spectral gap.
"""
import argparse

import numpy as np
import torch
from transformers import AutoTokenizer

from build_graph import cosine_similarity_matrix, standardize, unnormalized_laplacian
from extract_activations import DEFAULT_LAYER, DEFAULT_MODEL, extract_activations


def sample_random_tokens(model_name: str, n: int, seed: int) -> list[str]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    rng = np.random.default_rng(seed)
    vocab_size = tokenizer.vocab_size
    words = []
    while len(words) < n:
        tid = int(rng.integers(0, vocab_size))
        text = tokenizer.decode([tid]).strip()
        if not (text and text.isascii() and text.isprintable()):
            continue
        # keep only tokens that round-trip to exactly one BPE token, matching the
        # single-token filtering used for the real probe set (fair comparison)
        if len(tokenizer(text)["input_ids"]) != 1:
            continue
        words.append(text)
    return words


def main():
    parser = argparse.ArgumentParser(description="g=0 control: randomized-token spectrum")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--n", type=int, default=53)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default="control_eigenvalues.npz")
    args = parser.parse_args()

    tokens = sample_random_tokens(args.model, args.n, args.seed)
    print(f"Sampled {len(tokens)} random vocabulary tokens, e.g. {tokens[:8]}")

    vectors, _ = extract_activations(args.model, args.layer, tokens, args.device)
    vectors = standardize(vectors)
    weights = cosine_similarity_matrix(vectors)
    laplacian = unnormalized_laplacian(weights)

    eigenvalues, _ = np.linalg.eigh(laplacian)
    print("Control lowest 10 eigenvalues:", np.round(eigenvalues[:10], 5))

    np.savez(args.out, eigenvalues=eigenvalues, tokens=np.array(tokens, dtype=object))
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
