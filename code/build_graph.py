"""Phase 2: Relational Graph Construction.

Loads the activation vectors from Phase 1, builds the pairwise cosine-similarity
kernel, and assembles the unnormalised graph Laplacian directly from it.
"""
import argparse

import numpy as np


def standardize(vectors: np.ndarray) -> np.ndarray:
    # Transformer hidden states commonly have a handful of huge-magnitude outlier
    # dimensions that would otherwise dominate cosine similarity across the whole set.
    mean = vectors.mean(axis=0, keepdims=True)
    std = vectors.std(axis=0, keepdims=True)
    return (vectors - mean) / np.clip(std, 1e-12, None)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized = vectors / np.clip(norms, 1e-12, None)
    sim = normalized @ normalized.T
    # A graph Laplacian L = D - W is only positive-semidefinite for non-negative
    # weights; cosine similarity ranges [-1, 1], so anti-correlated pairs are
    # treated as unconnected rather than negatively connected.
    sim = np.clip(sim, 0.0, None)
    np.fill_diagonal(sim, 0.0)  # no self-loops
    return sim


def unnormalized_laplacian(weights: np.ndarray) -> np.ndarray:
    degree = np.diag(weights.sum(axis=1))
    return degree - weights


def main():
    parser = argparse.ArgumentParser(description="Phase 2: build relational graph + Laplacian")
    parser.add_argument("--activations", default="activations.npz")
    parser.add_argument("--out", default="graph.npz")
    parser.add_argument("--single-token-only", action="store_true",
                        help="Restrict to single-BPE-token probes, removing the "
                             "token-length confound from mixed-length inputs.")
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    vectors = data["vectors"]
    tokens = data["tokens"]
    categories = data["categories"]

    if args.single_token_only and "token_counts" in data:
        mask = data["token_counts"] == 1
        vectors, tokens, categories = vectors[mask], tokens[mask], categories[mask]
        print(f"Filtered to {mask.sum()} single-token probes")

    print(f"Loaded {vectors.shape[0]} activation vectors of dim {vectors.shape[1]}")

    vectors = standardize(vectors)
    weights = cosine_similarity_matrix(vectors)
    laplacian = unnormalized_laplacian(weights)

    print(f"Built {weights.shape[0]}x{weights.shape[1]} similarity matrix and Laplacian")

    np.savez(
        args.out,
        weights=weights,
        laplacian=laplacian,
        tokens=tokens,
        categories=categories,
    )
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
