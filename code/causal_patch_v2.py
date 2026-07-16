"""Phase 4, take 2: proper causal tracing.

Patches the subject-token position itself (the day/month name, always token
index 4 in "The day/month after {X} is") at a chosen layer, sweeping across
every layer of the network. Unlike the first attempt (which patched only the
final token and was silently circumvented by attention re-reading the literal
input word), this intervention actually removes the base representation at
the position that matters before the rest of the network runs.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-v0.1"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
SUBJECT_POSITION = 4  # "The <day/month> after {NAME} is" -> NAME is always token 4


def extract_all_layers(model, tokenizer, prompt, position=SUBJECT_POSITION):
    captured = {}
    handles = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden[0, position, :].detach().float().cpu().numpy()
        return hook

    for j, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(make_hook(j)))

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        model(**inputs)
    for h in handles:
        h.remove()
    return captured  # {layer_idx: vector}


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=3, help="donor = item at base_index + offset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="causal_trace.npz")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()
    n_layers = len(model.model.layers)
    rng = np.random.default_rng(args.seed)

    results = {}
    for cat_name, items, template in [("days_of_week", DAYS, "The day after {} is"),
                                       ("months", MONTHS, "The month after {} is")]:
        n = len(items)
        # cache every item's per-layer subject-position representation in one pass each
        per_item_layers = {item: extract_all_layers(model, tokenizer, template.format(item))
                           for item in items}

        pairs = [(items[i], items[(i + args.offset) % n]) for i in range(n)
                 if items[i] != items[(i + args.offset) % n]]
        expected = {base: items[(items.index(donor) + 1) % n] for base, donor in pairs}

        real_acc = np.zeros(n_layers)
        control_acc = np.zeros(n_layers)
        for layer_idx in range(n_layers):
            real_hits, control_hits = 0, 0
            for base, donor in pairs:
                donor_vec = per_item_layers[donor][layer_idx]
                base_vec = per_item_layers[base][layer_idx]
                exp = expected[base]

                real_pred = run_patched(model, tokenizer, template.format(base), layer_idx,
                                        SUBJECT_POSITION, donor_vec)
                diff_norm = np.linalg.norm(donor_vec - base_vec)
                noise = rng.normal(size=base_vec.shape)
                noise = noise / np.linalg.norm(noise) * diff_norm
                control_pred = run_patched(model, tokenizer, template.format(base), layer_idx,
                                           SUBJECT_POSITION, base_vec + noise)

                real_hits += exp.lower() in real_pred
                control_hits += exp.lower() in control_pred
            real_acc[layer_idx] = real_hits / len(pairs)
            control_acc[layer_idx] = control_hits / len(pairs)
            print(f"[{cat_name}] layer {layer_idx:2d}: real={real_acc[layer_idx]:.2f} "
                  f"control={control_acc[layer_idx]:.2f}")

        results[cat_name] = {"real": real_acc, "control": control_acc}

    np.savez(args.out,
             days_real=results["days_of_week"]["real"], days_control=results["days_of_week"]["control"],
             months_real=results["months"]["real"], months_control=results["months"]["control"],
             n_layers=n_layers)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
