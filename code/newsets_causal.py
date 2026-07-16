"""Does the rotation-generator mechanism found for days/months generalize to
other small closed enumerable sets discovered in the 500-token scale-up?

Tests zodiac signs (genuinely cyclic, like months), planets (linear ordinal
by distance, no natural cycle), and playing-card ranks (linear ordinal) with
the same two-part recipe that worked for days/months: (1) real-donor identity
swap at the subject-token position, at a fixed layer; (2) a full-rank rotation
generator fit via orthogonal Procrustes, applied with no real donor data.

Unlike days/months, these names are multi-token with varying lengths, so the
subject-token position is computed per item rather than assumed fixed.
"""
import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-v0.1"

ZODIAC = ["Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo", "Libra", "Scorpio",
          "Sagittarius", "Capricorn", "Aquarius", "Pisces"]
PLANETS = ["Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune"]
CARDS = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten",
         "Jack", "Queen", "King"]

TASKS = {
    "zodiac_signs": (ZODIAC, "The zodiac sign after {} is", True),
    "planets": (PLANETS, "The planet after {} in distance from the sun is", False),
    "playing_cards": (CARDS, "The playing card rank after {} is", False),
}


def subject_position(tokenizer, template, name):
    prefix = template.split("{}")[0] + name
    return len(tokenizer(prefix)["input_ids"]) - 1


def extract_layer(model, tokenizer, prompt, layer_idx, position):
    captured = {}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        captured["v"] = hidden[0, position, :].detach().float().cpu().numpy()

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    handle.remove()
    return captured["v"]


def run_patched(model, tokenizer, prompt, layer_idx, position, patch_vector):
    patch_tensor = torch.tensor(patch_vector, dtype=model.dtype, device=model.device)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[0, position, :] = patch_tensor
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    handle.remove()
    top_id = int(torch.argmax(logits))
    return tokenizer.decode([top_id]).strip().lower()


def fit_rotation_generator(coords, cyclic, n):
    source = coords if cyclic else coords[:-1]
    target = np.roll(coords, -1, axis=0) if cyclic else coords[1:]
    M = source.T @ target
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    residual = np.linalg.norm(source @ R - target) / np.linalg.norm(target)
    return R, residual


def matrix_power_real(R, p):
    eigvals, eigvecs = np.linalg.eig(R)
    powered = eigvals ** p
    return (eigvecs @ np.diag(powered) @ np.linalg.inv(eigvecs)).real


def matches(expected, pred):
    return expected.lower() in pred or pred in expected.lower()[:max(1, len(pred))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()

    results = {}
    for cat_name, (items, template, cyclic) in TASKS.items():
        n = len(items)
        positions = [subject_position(tokenizer, template, item) for item in items]
        vectors = np.stack([extract_layer(model, tokenizer, template.format(item), args.layer, pos)
                            for item, pos in zip(items, positions)])

        k = min(n - 1, vectors.shape[0] - 1)
        pca = PCA(n_components=k)
        coords = pca.fit_transform(vectors)
        R, residual = fit_rotation_generator(coords, cyclic, n)
        print(f"[{cat_name}] n={n} k={k} cyclic={cyclic} Procrustes residual={residual:.4f}")

        pair_range = range(n) if cyclic else range(n - 1)
        real_hits, rot_hits, count = 0, 0, 0
        for i in pair_range:
            base, base_pos = items[i], positions[i]
            j = (i + 1) % n
            if not cyclic and j == 0:
                continue
            expected = items[(j + 1) % n] if cyclic else (items[j + 1] if j + 1 < n else None)
            if expected is None:
                continue

            donor_vec = vectors[j]
            real_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                    base_pos, donor_vec)

            rotated_coord = coords[i] @ R
            rotated_vec = pca.inverse_transform(rotated_coord[None, :])[0]
            rot_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                   base_pos, rotated_vec)

            real_hit = matches(expected, real_pred)
            rot_hit = matches(expected, rot_pred)
            real_hits += real_hit
            rot_hits += rot_hit
            count += 1
            print(f"[{cat_name}] base={base:14s} expected={expected:14s} "
                  f"real={real_pred!r:12s}({'OK' if real_hit else 'x'})  "
                  f"rot={rot_pred!r:12s}({'OK' if rot_hit else 'x'})")

        results[cat_name] = (real_hits, rot_hits, count)
        print(f"{cat_name}: real={real_hits}/{count}  full-rank-rotation={rot_hits}/{count}\n")

    print("=== SUMMARY ===")
    for cat_name, (real_hits, rot_hits, count) in results.items():
        print(f"{cat_name:16s} real={real_hits}/{count} ({real_hits/count:.2f})  "
              f"rotation={rot_hits}/{count} ({rot_hits/count:.2f})")


if __name__ == "__main__":
    main()
