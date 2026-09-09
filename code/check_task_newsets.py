"""Sanity check: does the model solve a natural 'successor' task for each of
the newly-discovered small-enumerable-set categories, before we spend time
causally patching them?
"""
import torch
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


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                                       bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, device_map={"": 0})
    model.eval()

    for name, (items, template, cyclic) in TASKS.items():
        print(f"--- {name} ---")
        n = len(items)
        correct = 0
        pairs = n if cyclic else n - 1
        for i in range(pairs):
            item = items[i]
            expected = items[(i + 1) % n]
            prompt = template.format(item)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**inputs).logits[0, -1]
            top_id = int(torch.argmax(logits))
            top_tok = tokenizer.decode([top_id])
            stripped = top_tok.strip().lower()
            # correct if the predicted token is the full word or its correct
            # leading sub-token piece (multi-token names only ever predict
            # their first piece at this single greedy step)
            ok = bool(stripped) and expected.lower().startswith(stripped)
            correct += ok
            print(f"{prompt!r:55s} -> top: {top_tok!r:15s} expected: {expected!r} "
                  f"{'OK' if ok else 'WRONG'}")
        print(f"{name}: {correct}/{pairs}\n")


if __name__ == "__main__":
    main()
