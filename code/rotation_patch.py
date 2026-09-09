"""Phase 5: does rotating along the discovered PCA loop suffice, or do you need
the donor's whole real vector?

The causal-tracing result (causal_patch_v2.py) showed that swapping in a
donor month's real activation at the subject-token position causally shifts
the model's answer. That only proves the identity is *some* swappable code at
that position, not that it's specifically a point on a 2D circle. This script
tests the stronger, geometric claim directly: fit a circle to the 12 months'
own position-4 activations (at a fixed layer), then construct a *synthetic*
patch vector by rotating the base month's own point around that circle by the
angle corresponding to a given offset, reconstructed back into the full
hidden-dim space, added to the (unchanged) residual outside the 2D subspace.
If the model still shifts its answer to match, the circle itself is doing the
causal work, not just "this happens to be a real month's vector."
Control: rotate by an angle that does not correspond to any integer month
offset, and check that it does *not* produce a coherent, real-month answer.
"""
import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-v0.1"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
SUBJECT_POSITION = 4


def extract_layer(model, tokenizer, prompt, layer_idx, position=SUBJECT_POSITION):
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


def fit_circle_pca(vectors):
    """Return (pca, coords2d, center2d) for the given N x D vectors."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    center = coords.mean(axis=0)
    return pca, coords, center


def rotate_in_subspace(base_vector, pca, center2d, angle_rad):
    """Rotate base_vector's projection onto the PCA plane by angle_rad around
    center2d, keeping the orthogonal residual component unchanged."""
    coord2d = pca.transform(base_vector[None, :])[0]
    rel = coord2d - center2d
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    rotated_rel = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]])
    rotated_coord2d = center2d + rotated_rel
    # reconstruct: replace only the 2D-subspace component, keep residual
    delta = pca.inverse_transform(rotated_coord2d[None, :])[0] - pca.inverse_transform(coord2d[None, :])[0]
    return base_vector + delta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--out-prefix", default="rotation")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()

    for cat_name, items, template in [("days_of_week", DAYS, "The day after {} is"),
                                       ("months", MONTHS, "The month after {} is")]:
        n = len(items)
        vectors = np.stack([extract_layer(model, tokenizer, template.format(item), args.layer)
                            for item in items])
        pca, coords, center = fit_circle_pca(vectors)
        print(f"[{cat_name}] layer {args.layer} position-4 PCA explained variance: "
              f"{pca.explained_variance_ratio_.round(3)}")
        np.savez(f"{args.out_prefix}_{cat_name}_pca.npz", coords=coords, items=np.array(items, dtype=object),
                  explained_variance=pca.explained_variance_ratio_)

        step = 2 * np.pi / n
        real_hits, rot_hits, control_hits, count = 0, 0, 0, 0
        for i, base in enumerate(items):
            for k in [1, 2, 3, 4, 5]:
                donor = items[(i + k) % n]
                expected = items[(i + k + 1) % n]
                base_vec = vectors[i]
                donor_vec = vectors[(i + k) % n]

                real_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                        SUBJECT_POSITION, donor_vec)

                rotated_vec = rotate_in_subspace(base_vec, pca, center, k * step)
                rot_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                       SUBJECT_POSITION, rotated_vec)

                wrong_angle = (k + 0.5) * step  # deliberately off-lattice, matches no integer offset
                control_vec = rotate_in_subspace(base_vec, pca, center, wrong_angle)
                control_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                           SUBJECT_POSITION, control_vec)

                real_hits += expected.lower() in real_pred
                rot_hits += expected.lower() in rot_pred
                control_hits += expected.lower() in control_pred
                count += 1
                print(f"[{cat_name}] base={base:10s} k={k} expected={expected:10s} "
                      f"real={real_pred!r:12s} rot={rot_pred!r:12s} control={control_pred!r:12s}")

        print(f"\n{cat_name} @ layer {args.layer}: real donor={real_hits}/{count} "
              f"rotation-only={rot_hits}/{count} off-lattice control={control_hits}/{count}\n")


if __name__ == "__main__":
    main()
