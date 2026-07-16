"""Phase 4: Causal Validation via Activation Patching.

For each (base, donor) pair, run the base prompt ("The day after {base} is")
but splice the donor's layer-16 last-token hidden state in place of the base's
own — then check whether the model's next-token prediction shifts to the
donor's natural answer. A random-direction control (same-norm noise instead of
the donor's real vector) should NOT produce that shift.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-v0.1"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def run_patched(model, tokenizer, layer_idx, base_item, patch_vector, template):
    """Run template(base_item) with layer_idx's last-token hidden state replaced by patch_vector."""
    prompt = template.format(base_item)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    seq_len = inputs["input_ids"].shape[1]

    patch_tensor = torch.tensor(patch_vector, dtype=model.dtype, device=model.device)

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        hidden[0, seq_len - 1, :] = patch_tensor
        return (hidden,) + output[1:] if isinstance(output, tuple) else hidden

    handle = model.model.layers[layer_idx].register_forward_hook(hook)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1]
    handle.remove()

    top_id = int(torch.argmax(logits))
    return tokenizer.decode([top_id]).strip().lower()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations", default="template_activations.npz")
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    data = np.load(args.activations, allow_pickle=True)
    vectors = data["vectors"]
    labels = [str(l) for l in data["labels"]]
    categories = data["categories"]

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()

    rng = np.random.default_rng(args.seed)

    for cat_name, items, template in [("days_of_week", DAYS, "The day after {} is"),
                                       ("months", MONTHS, "The month after {} is")]:
        idx_of = {items[i]: i for i in range(len(items)) if items[i] in labels}
        vecs = {label: vectors[i] for i, label in enumerate(labels) if categories[i] == cat_name}

        real_hits, control_hits, n = 0, 0, 0
        for i, base in enumerate(items):
            donor = items[(i + 3) % len(items)]  # a non-adjacent offset, harder than +1
            if base == donor:
                continue
            expected = items[(idx_of[donor] + 1) % len(items)]

            real_pred = run_patched(model, tokenizer, args.layer, base, vecs[donor], template)

            diff_norm = np.linalg.norm(vecs[donor] - vecs[base])
            noise = rng.normal(size=vecs[base].shape)
            noise = noise / np.linalg.norm(noise) * diff_norm
            control_vec = vecs[base] + noise
            control_pred = run_patched(model, tokenizer, args.layer, base, control_vec, template)

            real_hit = expected.lower() in real_pred
            control_hit = expected.lower() in control_pred
            real_hits += real_hit
            control_hits += control_hit
            n += 1
            print(f"[{cat_name}] base={base:10s} donor={donor:10s} expected={expected:10s} "
                  f"real_pred={real_pred!r:15s} ({'HIT' if real_hit else 'miss'})  "
                  f"control_pred={control_pred!r:15s} ({'HIT' if control_hit else 'miss'})")

        print(f"\n{cat_name}: real-patch accuracy {real_hits}/{n} = {real_hits/n:.2f}   "
              f"random-direction control accuracy {control_hits}/{n} = {control_hits/n:.2f}\n")


if __name__ == "__main__":
    main()
