"""Sanity check: does the model actually solve the day/month-after task?"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "mistralai/Mistral-7B-v0.1"
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()

    print("--- Day-after task ---")
    for i, day in enumerate(DAYS):
        expected = DAYS[(i + 1) % 7]
        prompt = f"The day after {day} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        top_id = int(torch.argmax(logits))
        top_tok = tokenizer.decode([top_id])
        print(f"{prompt!r:35s} -> top: {top_tok!r:15s} expected: {expected!r} "
              f"{'OK' if expected.lower() in top_tok.lower() else 'WRONG'}")

    print("\n--- Month-after task ---")
    for i, month in enumerate(MONTHS):
        expected = MONTHS[(i + 1) % 12]
        prompt = f"The month after {month} is"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            logits = model(**inputs).logits[0, -1]
        top_id = int(torch.argmax(logits))
        top_tok = tokenizer.decode([top_id])
        print(f"{prompt!r:35s} -> top: {top_tok!r:15s} expected: {expected!r} "
              f"{'OK' if expected.lower() in top_tok.lower() else 'WRONG'}")


if __name__ == "__main__":
    main()
