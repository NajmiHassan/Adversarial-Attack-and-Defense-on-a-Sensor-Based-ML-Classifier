# Report Outline — Adversarial Robustness of a Sensor-Based Classifier (CPS)

Target length: 2–4 pages. Fill the brackets in with your numbers from
`results/results.json` and drop in `results/comparison.png`.

## 1. Problem & Motivation (~0.5 page)
- What cyber-physical systems rely on sensor-based ML classifiers
  (wearables, IoT, autonomous vehicles, industrial monitoring).
- The risk: adversarial perturbations can flip predictions while staying small
  and plausible. Why that's dangerous in a CPS where outputs trigger actions.
- One sentence on your specific setup: HAR classifier as a stand-in for a
  sensor-driven CPS component.

## 2. Background (~0.5 page)
- Adversarial examples in one paragraph (Goodfellow et al.).
- FGSM: single gradient sign step, budget eps.
- PGD: iterative, projected, the standard strong benchmark (Madry et al.).
- Adversarial training as a defense: train on adversarial examples so the model
  learns a flatter, more robust decision boundary.
- Threat model: L-infinity bounded, white-box. State eps, alpha, steps.

## 3. Method (~0.5 page)
- Dataset: UCI HAR, 561 features, 6 activities, train/test split sizes.
- Model: MLP architecture (layers, dropout), optimizer, epochs.
- Attacks: FGSM and PGD, clipped to [-1, 1]. Give the exact hyper-parameters.
- Defense: PGD adversarial training (same attack used during training).
- Evaluation: accuracy on clean, FGSM, and PGD inputs, for both models.

## 4. Results (~1 page)
- Table: clean / FGSM / PGD accuracy for baseline and defended models.
  | Model        | Clean | FGSM | PGD |
  |--------------|-------|------|-----|
  | Baseline     | [ ]   | [ ]  | [ ] |
  | Adv-trained  | [ ]   | [ ]  | [ ] |
- Figure: `comparison.png`.
- Narrate the story:
  - How far did accuracy fall under attack? (the vulnerability)
  - How much did adversarial training recover? (the defense)
  - Did clean accuracy drop slightly after defending? (the robustness/accuracy
    trade-off — an important point to discuss)
- Optional: accuracy-vs-eps curve if you ran a sweep.

## 5. Discussion & Limitations (~0.25 page)
- Engineered features vs. raw signals: perturbing 561 features is a simplified
  threat model; raw-signal attacks would be more realistic.
- White-box assumption; real attackers may have less access.
- Adversarial training defends mainly against the attack it trained on.

## 6. Conclusion (~0.25 page)
- Recap: demonstrated the vulnerability and a working baseline defense in a CPS
  framing. One line on the most interesting result and one on future work.

## Appendix / Repo
- Link to GitHub. Note reproducibility: fixed seed, single command to run.
