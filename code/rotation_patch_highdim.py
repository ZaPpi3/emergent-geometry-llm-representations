"""Phase 5b: does a full-rank rotation (not just a 2D one) explain the effect?

The first rotation test (rotation_patch.py) used only the top-2 PCA
components, which captured just 39-50% of the variance at this position, and
found the reconstructed vector carried none of the causal effect. This script
uses PCA with n_items-1 components instead, which captures effectively 100%
of the variance among these items' own activation vectors (the most any
linear basis restricted to these n examples can capture), and fits a single
best-fit orthogonal "rotation" matrix R via the orthogonal Procrustes problem:
the R that best maps each item's coordinate to the next item's, in natural
cyclic order (Monday->Tuesday->...->Sunday->Monday, or Jan->Feb->...->Dec->Jan).

If a single consistent rotation generator explains the cyclic structure well
(low Procrustes residual) and R^k, applied as a synthetic patch, reproduces
the causal identity-swap effect, that supports a genuinely higher-dimensional
rotational code. A fractional-power control (R raised to a non-integer power
matching no real calendar offset, via eigendecomposition) checks any effect is
specific to real integer steps, not just "perturb along the same manifold."
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


def fit_rotation_generator(coords):
    """coords: n x k, rows already in cyclic order. Returns R (k x k) minimizing
    ||coords @ R - roll(coords, -1)||_F over orthogonal R (orthogonal Procrustes),
    plus the relative residual of that fit."""
    source = coords
    target = np.roll(coords, -1, axis=0)
    M = source.T @ target
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    residual = np.linalg.norm(source @ R - target) / np.linalg.norm(target)
    return R, residual


def matrix_power_real(R, p):
    """R^p for real orthogonal R and real (possibly non-integer) p, via
    eigendecomposition (works because rotation eigenvalues are complex
    conjugate pairs e^{+-i theta}; raising to a real power and recombining
    keeps the result real up to numerical noise)."""
    eigvals, eigvecs = np.linalg.eig(R)
    powered = eigvals ** p
    R_p = eigvecs @ np.diag(powered) @ np.linalg.inv(eigvecs)
    return R_p.real


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=12)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()

    for cat_name, items, template in [("days_of_week", DAYS, "The day after {} is"),
                                       ("months", MONTHS, "The month after {} is")]:
        n = len(items)
        k = n - 1  # full rank within these n points' own affine span
        vectors = np.stack([extract_layer(model, tokenizer, template.format(item), args.layer)
                            for item in items])

        pca = PCA(n_components=k)
        coords = pca.fit_transform(vectors)
        print(f"[{cat_name}] layer {args.layer}, k={k} components, "
              f"explained variance sum={pca.explained_variance_ratio_.sum():.4f}")

        R, residual = fit_rotation_generator(coords)
        print(f"[{cat_name}] single-rotation-generator Procrustes residual: {residual:.4f} "
              f"(0 = perfect single consistent rotation explains all steps)")

        real_hits, rot_hits, control_hits, count = 0, 0, 0, 0
        for i, base in enumerate(items):
            base_coord = coords[i]
            for step in [1, 2, 3, 4, 5]:
                donor = items[(i + step) % n]
                expected = items[(i + step + 1) % n]
                donor_vec = vectors[(i + step) % n]

                real_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                        SUBJECT_POSITION, donor_vec)

                R_step = matrix_power_real(R, step)
                rotated_coord = base_coord @ R_step
                rotated_vec = pca.inverse_transform(rotated_coord[None, :])[0]
                rot_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                       SUBJECT_POSITION, rotated_vec)

                R_wrong = matrix_power_real(R, step + 0.5)
                wrong_coord = base_coord @ R_wrong
                wrong_vec = pca.inverse_transform(wrong_coord[None, :])[0]
                control_pred = run_patched(model, tokenizer, template.format(base), args.layer,
                                           SUBJECT_POSITION, wrong_vec)

                real_hits += expected.lower() in real_pred
                rot_hits += expected.lower() in rot_pred
                control_hits += expected.lower() in control_pred
                count += 1
                print(f"[{cat_name}] base={base:10s} step={step} expected={expected:10s} "
                      f"real={real_pred!r:12s} rot(k-dim)={rot_pred!r:12s} control={control_pred!r:12s}")

        print(f"\n{cat_name} @ layer {args.layer}, k={k}: real donor={real_hits}/{count} "
              f"full-rank rotation={rot_hits}/{count} fractional-power control={control_hits}/{count}\n")


if __name__ == "__main__":
    main()
