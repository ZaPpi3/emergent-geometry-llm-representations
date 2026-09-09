"""Template-based extraction: elicit the model actually computing "day/month after X"
in context, and capture the hidden state at the position right before it answers.
Closer to the actual published methodology than probing bare isolated words.
"""
import argparse

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-v0.1"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def build_prompts():
    prompts, labels, categories = [], [], []
    for day in DAYS:
        prompts.append(f"The day after {day} is")
        labels.append(day)
        categories.append("days_of_week")
    for month in MONTHS:
        prompts.append(f"The month after {month} is")
        labels.append(month)
        categories.append("months")
    return prompts, labels, categories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--out", default="template_activations.npz")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()

    captured = {}

    def hook(_module, _inputs, output):
        captured["hidden"] = output[0].detach() if isinstance(output, tuple) else output.detach()

    handle = model.model.layers[args.layer].register_forward_hook(hook)

    prompts, labels, categories = build_prompts()
    vectors = []
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            model(**inputs)
            vectors.append(captured["hidden"][0, -1].float().cpu().numpy())
    handle.remove()

    vectors = np.stack(vectors)
    print(f"Extracted {vectors.shape} template activations at layer {args.layer}")
    np.savez(args.out, vectors=vectors, labels=np.array(labels, dtype=object),
              categories=np.array(categories, dtype=object), layer=args.layer)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
