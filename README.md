# Adversarial Attack and Defense on a Sensor-Based ML Classifier (CPS Context)

A compact, end-to-end study of adversarial robustness for a sensor-based
classifier, framed in a cyber-physical systems (CPS) setting. We train a model
on human-activity sensor data, attack it with standard adversarial methods
(FGSM, PGD), then defend it with adversarial training and measure how much
accuracy recovers.

**Why it matters (the CPS angle):** wearables, IoT health monitors, and
vehicle-mounted sensors feed ML classifiers that drive real decisions. Small,
deliberately crafted perturbations to a sensor stream can flip a prediction
while staying within physically plausible bounds. This project demonstrates the
vulnerability and a baseline mitigation.

## The pipeline

1. **Baseline** — train an MLP on the UCI HAR dataset (~95% clean accuracy).
2. **Attack** — FGSM and PGD (L-infinity, white-box); measure accuracy drop.
3. **Defense** — PGD adversarial training; measure accuracy recovery.
4. **Compare** — clean vs. attacked vs. defended, as a table and a bar chart.

## Setup

```bash
pip install -r requirements.txt
```

### Get the dataset (one time)

1. Download "Human Activity Recognition Using Smartphones" from the UCI
   repository: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones
2. Unzip until you have a folder named exactly **`UCI HAR Dataset`**
   (it contains `train/` and `test/` subfolders).
3. Put it next to `main.py`, or pass its path with `--data_dir`.

## Run

```bash
python main.py --data_dir "UCI HAR Dataset"
```

Quick smoke test (fewer epochs):

```bash
python main.py --data_dir "UCI HAR Dataset" --epochs 3 --adv_epochs 3
```

Outputs land in `results/`: `results.json` (all metrics + config) and
`comparison.png` (the bar chart for your report).

### Useful flags

| Flag | Meaning | Default |
|------|---------|---------|
| `--eps` | L-inf perturbation budget (in the [-1, 1] feature space) | `0.1` |
| `--alpha` | PGD step size | `0.01` |
| `--steps` | PGD iterations | `20` |
| `--epochs` / `--adv_epochs` | training length for baseline / defended model | `15` |

Try a sweep of `--eps` values (e.g. 0.03, 0.05, 0.1, 0.2) to make a nice
"accuracy vs. epsilon" curve for the write-up.

## File layout

```
adversarial-robustness-cps/
├── data.py            # load UCI HAR (561-feature version)
├── model.py           # small MLP classifier
├── attacks.py         # FGSM + PGD, from scratch
├── main.py            # train -> attack -> defend -> compare
├── requirements.txt
├── report_outline.md  # skeleton for the 2-4 page write-up
└── results/           # generated: metrics + plot
```

## Next steps (good Phase-2 extensions)

- **More realistic threat model:** attack the *raw* inertial signals
  (9 channels x 128 timesteps) with a 1D-CNN, instead of the engineered features.
- **Validate with a library:** reproduce the numbers using the Adversarial
  Robustness Toolbox (ART) and note that they match — strengthens the report.
- **More defenses:** input preprocessing (feature squeezing / smoothing),
  and compare against adversarial training.
- **Robustness curve:** plot accuracy vs. `eps` for clean and defended models.

## References

- Goodfellow et al., *Explaining and Harnessing Adversarial Examples* (FGSM), 2015.
- Madry et al., *Towards Deep Learning Models Resistant to Adversarial Attacks* (PGD / adversarial training), 2018.
- UCI HAR dataset, Anguita et al., 2013.
