# Searching for Emergent Geometric Coordinate Systems in LLM Hidden Representations

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

Official repository for the paper **"Searching for Emergent Geometric Coordinate Systems in LLM Hidden Representations: A Replication Attempt" (2026)** by Paul W. Jarvis. This work is a replication attempt: can a graph-Laplacian / manifold-learning pipeline reconstruct geometric structure (circles, lines) from LLM hidden-layer activations, the way recent interpretability work has found for concepts like days-of-week and months? The investigation runs the pipeline, hits a null result, diagnoses two successive flaws in the experiment design rather than in the underlying claim, and ends with a causal-patching result strong enough to confirm a genuinely rotational (if not literally two-dimensional) geometric code.

**[Read the paper: `main.pdf`](main.pdf)**, built from `main.tex`.

---

## 🔍 Conceptual Overview

Recent interpretability work has reported that some LLMs represent cyclic concepts (days of the week, months) as literal 2D circles in activation space. This repository probes how robust and how general that finding is: whether it shows up from bare-word probing alone, whether a naive circular fit is actually the representation the model *uses*, and whether the same graph-Laplacian diagnostic that finds "geometric" clusters is picking up something specific to cyclic/temporal concepts or something more general about small, closed, named categories.

## 🧠 Key Scientific Findings

- **Calendar/cyclic concepts (days, months) form a distinct cluster** separated from quantity/space/relation concepts, confirmed above a randomized-token control. This separation gets **sharper with model scale** (Qwen2.5-0.5B → Mistral-7B-v0.1).
- **A clean 2D circular manifold did not emerge** from bare-word probing in either model.
- **Switching to in-context arithmetic prompts** ("The month after March is") + PCA showed a real, mostly-clean loop for **months**, but not for **days-of-week**.
- **A first causal-patching experiment produced a null result**, diagnosed as an intervention-design flaw (patching only the last token position, which the model could route around by re-reading the literal day/month name earlier in the prompt), not evidence against the effect.
- **Fixing that flaw (patching the subject-token position instead) produced a clean, strongly positive result:** swapping in a donor day/month's representation flips the model's answer to match the donor, 83-100% accuracy through the early-to-mid layers, collapsing sharply to 0% past a well-defined depth (layer ~20/32 for days, ~17/32 for months), while a random-direction control never once succeeds, at any layer. This confirms the day/month identity at that position is causally used by the model, not just correlational structure - though it doesn't by itself prove that identity is encoded as a 2D circle specifically.
- **Testing that stronger claim directly: months trace the cleanest 2D loop in the whole project at this same position, and it still isn't the causal code.** A synthetic vector built by rotating a base month's own activation within that discovered 2D plane, by the exact angle matching a real calendar offset, produces the correct answer 0/60 times, identical to rotating by a deliberately wrong angle (also 0/60). The 2D plane only captures ~39% of the variance at that position; a clean-looking loop is not the same as a causally sufficient code.
- **Extending the same idea to a full-rank subspace recovers the effect almost completely.** Using n-1 principal components (~100% of the variance among the items' own vectors) and a single rotation generator fit once via orthogonal Procrustes, `R^offset` applied to a base item's own vector, with no real donor data at all, matches or slightly exceeds the real-donor patch (100% vs 100% for days, 85% vs 82% for months), while a matched fractional-power control collapses to 8-14%, near chance. The code is genuinely rotational, just not two-dimensional: the 2D failure above was a dimensionality problem, not evidence against a geometric encoding.
- **Scaling the probe set from 77 to 500 tokens across 25 categories confirms the calendar-cluster finding and refines it.** The categories that separate cleanly turn out to be small, closed, enumerable named sets in general, not specifically cyclic/temporal ones: continents, zodiac signs, playing-card ranks, and planets each form their own distinguishable region in Mistral-7B too. Playing-card ranks even separate from ordinary cardinal numbers despite sharing the same words (`Two` vs `two`). Categories chosen to have no expected structure (animals, colors, body parts, weather, emotions) collapse into one undifferentiated blob - a second negative control that emerged from the data itself rather than being designed in.
- **The rotation mechanism generalizes partially, and the gap tracks subject-name token length almost monotonically.** Repeating the real-donor + full-rank-rotation recipe on the other small closed sets found above: playing-card ranks reach 82%/73% accuracy, close to the days/months level, while zodiac signs and planets reach only 33-42%/17-33%. Average subject-name token length predicts the ranking cleanly (1.00 for days/months, 1.15 for cards, 2.00-2.25 for planets/zodiac): patching a single token position leaves any *other* sub-token of a multi-token name unpatched and still visible to downstream attention, diluting the effect for longer, more-fragmented names - the same leakage mechanism diagnosed in the first causal-patching attempt, now showing up as a matter of degree rather than all-or-nothing.

**Bottom line:** the naive version of the "LLMs encode cyclic concepts as literal 2D circles" claim doesn't survive contact with a causal test, but a more general version does - the day/month identity used by the model is genuinely rotational, just distributed across a full-rank subspace rather than compressed into two dimensions. We think the sequence of null results that led there (last-token patching, then 2D-plane rotation) is as informative as the final positive result, since each failure isolated exactly which simplifying assumption was wrong.

## ⚙️ The Computational Pipeline

1. **Activation extraction:** pulls hidden-state activations for single-token probe words from a chosen layer of Qwen2.5-0.5B or Mistral-7B-v0.1 (4-bit). *(`code/extract_activations.py`)*
2. **Similarity graph construction:** builds a cosine-similarity graph over the extracted activations, restricted to single-token probes. *(`code/build_graph.py`)*
3. **Graph-Laplacian diagonalization:** low-lying eigenmodes of the graph Laplacian, plotted in 2D/3D - the core manifold-learning diagnostic used throughout. *(`code/diagonalize_plot.py`)*
4. **Randomized-token control:** repeats the diagnostic on random tokens to establish the null distribution the calendar cluster is compared against. *(`code/control_experiment.py`)*
5. **Category-level probing:** narrows the diagnostic to specific category subsets (e.g. days-of-week vs. months) with a defined item ordering. *(`code/narrow_category.py`)*
6. **Cross-model spectrum comparison:** compares eigenmode spectra between Qwen2.5-0.5B and Mistral-7B-v0.1. *(`code/compare_spectra.py`)*
7. **In-context template extraction + PCA:** re-extracts activations from arithmetic-style prompts ("The month after March is") rather than bare words, then fits PCA. *(`code/check_task.py`, `code/template_extract.py`, `code/template_pca_plot.py`)*
8. **Causal patching, attempt 1 (last-token position):** activation patching at the final token position - the null result that motivated attempt 2. *(`code/causal_patch.py`)*
9. **Causal patching, attempt 2 (subject-token position):** activation patching at the subject-token position, across all layers - the positive result establishing causal use of day/month identity. *(`code/causal_patch_v2.py`, `code/plot_causal_trace.py`)*
10. **Rotation patching, attempt 3 (2D plane):** synthetic activations built by rotating within the discovered 2D loop by the calendar-matched angle - a null result. *(`code/rotation_patch.py`, `code/plot_rotation_pca.py`)*
11. **Rotation patching, attempt 4 (full-rank subspace):** the same idea generalized to an n-1-dimensional rotation generator fit via orthogonal Procrustes - the positive result that closes the investigation. *(`code/rotation_patch_highdim.py`, `code/plot_rotation_summary.py`)*
12. **Generalization test:** repeats the real-donor + full-rank-rotation recipe on zodiac signs, planets, and playing-card ranks - the other small closed sets surfaced by the 500-token scale-up - to test whether the mechanism found for days/months holds more broadly. *(`code/check_task_newsets.py`, `code/newsets_causal.py`, `code/plot_generalization.py`)*

## 📁 Repository Structure

- `main.tex` / `main.pdf` : Manuscript and LaTeX source.
- `code/` : Every script above, in the order the pipeline runs, plus `probe_tokens.json` (the probe word/category lists).
- `data/` : Intermediate `.npz` artifacts at each pipeline phase (activations, graphs, eigenmodes, PCA/rotation results).
- `figures/` : All plots referenced in the manuscript.
- `requirements.txt` : Pinned dependency versions.

## 🔁 Reproducing

```bash
pip install -r requirements.txt
```

Requires Python 3.12+ and an NVIDIA GPU (8GB+ VRAM tested).

Pipeline order (small model, Qwen2.5-0.5B):
```bash
python code/extract_activations.py                      # Phase 1
python code/build_graph.py --single-token-only           # Phase 2
python code/diagonalize_plot.py                          # Phase 3
python code/control_experiment.py                        # g=0 control
python code/narrow_category.py --category days_of_week,months \
    --order Monday Tuesday Wednesday Thursday Friday Saturday Sunday \
    --order January February March April May June July August September October November December
```

Bigger model (Mistral-7B-v0.1, 4-bit, needs the gated-free public checkpoint):
```bash
python code/extract_activations.py --model mistralai/Mistral-7B-v0.1 \
    --layer 16 --load-in-4bit --out activations_mistral.npz
python code/build_graph.py --activations activations_mistral.npz \
    --single-token-only --out graph_mistral.npz
python code/diagonalize_plot.py --graph graph_mistral.npz \
    --out-prefix eigenmodes_mistral \
    --title "Graph Laplacian low-lying eigenmodes (Mistral-7B-v0.1, layer 16)"
```

Full 500-token probe set, either model (swap `--model`/`--layer`/`--load-in-4bit` as above):
```bash
python code/extract_activations.py --out activations_500.npz
python code/build_graph.py --activations activations_500.npz \
    --single-token-only --out graph_500.npz
python code/diagonalize_plot.py --graph graph_500.npz --out-prefix eigenmodes_500 \
    --title "Graph Laplacian low-lying eigenmodes (Qwen2.5-0.5B, layer 12, 500 tokens)"
```

Template-based / causal-patching experiments:
```bash
python code/check_task.py            # sanity check: does the model solve the task?
python code/template_extract.py --layer 16
python code/template_pca_plot.py
python code/causal_patch.py          # attempt 1: last-token patch (null result)
python code/causal_patch_v2.py       # attempt 2: subject-token position, all layers (positive result)
python code/plot_causal_trace.py
python code/rotation_patch.py --layer 12   # attempt 3: rotate within the discovered 2D loop (null result)
python code/plot_rotation_pca.py
python code/rotation_patch_highdim.py --layer 12   # attempt 4: full-rank rotation generator (positive result)
python code/plot_rotation_summary.py
```

Generalization test (zodiac signs, planets, playing-card ranks):
```bash
python code/check_task_newsets.py    # sanity check: does the model solve each successor task?
python code/newsets_causal.py --layer 12
python code/plot_generalization.py
```

Rebuild the PDF:
```bash
tectonic main.tex
```

## ⚠️ A note on interpretation

This repository documents a chain of null results as much as it documents the final positive one. The first causal-patching attempt failed not because the underlying claim was wrong, but because the intervention patched the wrong token position; the first rotation experiment failed not because the code isn't geometric, but because it isn't confined to two dimensions. We've kept both failed attempts in the pipeline (`causal_patch.py`, `rotation_patch.py`) alongside the fixes that superseded them, since the diagnostic value of a replication attempt is in showing which assumptions broke, not just in the headline result at the end.

## License

MIT - see `LICENSE`.
