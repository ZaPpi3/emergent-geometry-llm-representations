"""2026-10-28: the direct test flagged in NOTES.md (added 2026-07-16, never
run) for the "token-length explains the generalization gap" hypothesis in
newsets_causal.py. That script's `subject_position()` locates only the LAST
sub-token of a multi-token subject name (e.g. "Capricorn" tokenizes to
several pieces; only the final piece's hidden state gets read or patched).
The hypothesis, stated as correlational in NOTES.md and main.tex's
Limitations: a multi-token subject leaves its earlier sub-tokens' hidden
states untouched by the patch, diluting the causal effect, which is why
longer subject names show weaker rotation-patching accuracy.

This script tests that directly, not correlationally: same recipe as
newsets_causal.py (real-donor swap and full-rank rotation generator, same
layer, same tasks), except the patch is broadcast across EVERY sub-token
position of the base subject's own span, not just the last one, using the
same single donor vector (still extracted at the donor's own last
subject-token position, unchanged) as the broadcast source. If the
dilution hypothesis is right, hit rates should rise relative to
newsets_causal.py's last-token-only results, especially for the
longest-name categories (zodiac/planets) where the original gap was
largest; if hit rates don't change, the token-length correlation wasn't
actually causal in this specific sense.
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


def subject_span(tokenizer, template, name):
    """Returns (start, end) inclusive token-index range covering every
    sub-token of `name` as it appears in `template.format(name)`, instead
    of newsets_causal.py's single last-sub-token position."""
    prefix_before = template.split("{}")[0]
    before_len = len(tokenizer(prefix_before)["input_ids"])
    full_prefix_len = len(tokenizer(prefix_before + name)["input_ids"])
    start = before_len - 1  # same convention as the original: 0-indexed, counts from 0
    end = full_prefix_len - 1
    return start, end


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


def run_patched_span(model, tokenizer, prompt, layer_idx, start, end, patch_vector):
    """Broadcasts patch_vector across every position in [start, end]
    inclusive, instead of newsets_causal.py's single-position write."""
    patch_tensor = torch.tensor(patch_vector, dtype=model.dtype, device=model.device)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        for pos in range(start, end + 1):
            hidden[0, pos, :] = patch_tensor
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
    span_stats = {}
    for cat_name, (items, template, cyclic) in TASKS.items():
        n = len(items)
        spans = [subject_span(tokenizer, template, item) for item in items]
        last_positions = [e for (_s, e) in spans]
        n_subtokens = [e - s + 1 for (s, e) in spans]
        vectors = np.stack([extract_layer(model, tokenizer, template.format(item), args.layer, pos)
                            for item, pos in zip(items, last_positions)])

        k = min(n - 1, vectors.shape[0] - 1)
        pca = PCA(n_components=k)
        coords = pca.fit_transform(vectors)
        R, residual = fit_rotation_generator(coords, cyclic, n)
        print(f"[{cat_name}] n={n} k={k} cyclic={cyclic} Procrustes residual={residual:.4f} "
              f"n_subtokens={n_subtokens}")

        pair_range = range(n) if cyclic else range(n - 1)
        real_hits, rot_hits, count = 0, 0, 0
        real_hits_multitoken, rot_hits_multitoken, count_multitoken = 0, 0, 0
        for i in pair_range:
            base, (base_start, base_end) = items[i], spans[i]
            is_multitoken = (base_end > base_start)
            j = (i + 1) % n
            if not cyclic and j == 0:
                continue
            expected = items[(j + 1) % n] if cyclic else (items[j + 1] if j + 1 < n else None)
            if expected is None:
                continue

            donor_vec = vectors[j]
            real_pred = run_patched_span(model, tokenizer, template.format(base), args.layer,
                                         base_start, base_end, donor_vec)

            rotated_coord = coords[i] @ R
            rotated_vec = pca.inverse_transform(rotated_coord[None, :])[0]
            rot_pred = run_patched_span(model, tokenizer, template.format(base), args.layer,
                                        base_start, base_end, rotated_vec)

            real_hit = matches(expected, real_pred)
            rot_hit = matches(expected, rot_pred)
            real_hits += real_hit
            rot_hits += rot_hit
            count += 1
            if is_multitoken:
                real_hits_multitoken += real_hit
                rot_hits_multitoken += rot_hit
                count_multitoken += 1
            print(f"[{cat_name}] base={base:14s} (n_subtok={base_end-base_start+1}) "
                  f"expected={expected:14s} "
                  f"real={real_pred!r:12s}({'OK' if real_hit else 'x'})  "
                  f"rot={rot_pred!r:12s}({'OK' if rot_hit else 'x'})")

        results[cat_name] = (real_hits, rot_hits, count)
        span_stats[cat_name] = (real_hits_multitoken, rot_hits_multitoken, count_multitoken)
        print(f"{cat_name}: real={real_hits}/{count}  full-rank-rotation={rot_hits}/{count}  "
              f"(multitoken-subjects only: real={real_hits_multitoken}/{count_multitoken}  "
              f"rotation={rot_hits_multitoken}/{count_multitoken})\n")

    print("=== SUMMARY (full-span patch, this script) ===")
    for cat_name, (real_hits, rot_hits, count) in results.items():
        print(f"{cat_name:16s} real={real_hits}/{count} ({real_hits/count:.2f})  "
              f"rotation={rot_hits}/{count} ({rot_hits/count:.2f})")
    print("\n=== SUMMARY, multi-token subjects only (where this script's fix can matter) ===")
    for cat_name, (real_hits, rot_hits, count) in span_stats.items():
        if count == 0:
            print(f"{cat_name:16s} no multi-token subjects in this task")
            continue
        print(f"{cat_name:16s} real={real_hits}/{count} ({real_hits/count:.2f})  "
              f"rotation={rot_hits}/{count} ({rot_hits/count:.2f})")
    print("\nCompare against newsets_causal.py's last-token-only baseline "
          "(see NOTES.md for the original numbers) to test whether full-span "
          "patching recovers the gap, especially for multi-token subjects.")


if __name__ == "__main__":
    main()
