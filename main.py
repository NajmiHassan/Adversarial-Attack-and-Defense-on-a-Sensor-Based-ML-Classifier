import os
import json
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from data import load_har
from model import MLP
from attacks import fgsm, pgd

RESULTS_DIR = "results"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(model, loader, epochs, lr, device, attack=None, attack_kwargs=None):
    """
    Train `model`. If `attack` is given, do adversarial training: each batch is
    replaced by adversarial examples generated on the current model (Madry-style).
    """
    attack_kwargs = attack_kwargs or {}
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        total, correct, running_loss = 0, 0, 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)

            if attack is not None:
                xb = attack(model, xb, yb, **attack_kwargs)

            opt.zero_grad()
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            loss.backward()
            opt.step()

            running_loss += loss.item() * yb.size(0)
            correct += (out.argmax(1) == yb).sum().item()
            total += yb.size(0)

        print(f"  epoch {epoch:2d}/{epochs} | "
              f"loss {running_loss / total:.4f} | train acc {correct / total:.4f}")
    return model


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _accuracy_clean(model, X, y, device, batch_size):
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        correct += (model(xb).argmax(1) == yb).sum().item()
    return correct / len(X)


def _accuracy_under_attack(model, X, y, device, batch_size, attack, attack_kwargs):
    """Attacks need gradients, so this is NOT wrapped in no_grad."""
    model.eval()
    correct = 0
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        yb = y[i:i + batch_size].to(device)
        x_adv = attack(model, xb, yb, **attack_kwargs)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == yb).sum().item()
    return correct / len(X)


def evaluate_all(model, X, y, device, batch_size, eps, alpha, steps):
    """Return accuracy on clean / FGSM / PGD inputs."""
    return {
        "clean": _accuracy_clean(model, X, y, device, batch_size),
        "fgsm": _accuracy_under_attack(
            model, X, y, device, batch_size, fgsm, {"eps": eps}),
        "pgd": _accuracy_under_attack(
            model, X, y, device, batch_size, pgd,
            {"eps": eps, "alpha": alpha, "steps": steps}),
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def print_table(baseline, defended):
    def row(name, d):
        return (f"  {name:<18} {d['clean']*100:8.2f} "
                f"{d['fgsm']*100:8.2f} {d['pgd']*100:8.2f}")
    print("\n=== Accuracy (%) ===")
    print(f"  {'model':<18} {'clean':>8} {'FGSM':>8} {'PGD':>8}")
    print("  " + "-" * 44)
    print(row("baseline", baseline))
    print(row("adv-trained", defended))


def save_plot(baseline, defended, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    conditions = ["clean", "fgsm", "pgd"]
    x = np.arange(len(conditions))
    w = 0.35
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(x - w / 2, [baseline[c] * 100 for c in conditions], w, label="baseline")
    ax.bar(x + w / 2, [defended[c] * 100 for c in conditions], w, label="adv-trained")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Clean vs. Attacked vs. Defended")
    ax.set_xticks(x)
    ax.set_xticklabels(["Clean", "FGSM", "PGD"])
    ax.set_ylim(0, 100)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"Saved plot to {path}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Adversarial robustness on HAR.")
    ap.add_argument("--data_dir", default="UCI HAR Dataset",
                    help="Path to the unzipped 'UCI HAR Dataset' folder.")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--adv_epochs", type=int, default=15,
                    help="Epochs for the adversarially-trained (defended) model.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch_size", type=int, default=128)
    # Attack hyper-parameters (in the [-1, 1] feature space).
    ap.add_argument("--eps", type=float, default=0.1, help="L-inf budget.")
    ap.add_argument("--alpha", type=float, default=0.01, help="PGD step size.")
    ap.add_argument("--steps", type=int, default=20, help="PGD iterations.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Data ---
    X_train, y_train, X_test, y_test, class_names = load_har(args.data_dir)
    print(f"Loaded HAR: train={X_train.shape}, test={X_test.shape}, "
          f"classes={len(class_names)}")

    X_train_t = torch.from_numpy(X_train)
    y_train_t = torch.from_numpy(y_train)
    X_test_t = torch.from_numpy(X_test)
    y_test_t = torch.from_numpy(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size, shuffle=True)

    eval_kwargs = dict(device=device, batch_size=args.batch_size,
                       eps=args.eps, alpha=args.alpha, steps=args.steps)

    # --- Phase 1: baseline ---
    print("\n[Phase 1] Training baseline model...")
    baseline_model = MLP(in_features=X_train.shape[1], num_classes=len(class_names))
    train(baseline_model, train_loader, args.epochs, args.lr, device)

    # --- Phase 2: attack the baseline ---
    print("\n[Phase 2] Evaluating baseline (clean + attacks)...")
    baseline_res = evaluate_all(baseline_model, X_test_t, y_test_t, **eval_kwargs)

    # --- Phase 3: adversarial training (defense) ---
    print("\n[Phase 3] Training defended model with PGD adversarial training...")
    defended_model = MLP(in_features=X_train.shape[1], num_classes=len(class_names))
    train(defended_model, train_loader, args.adv_epochs, args.lr, device,
          attack=pgd,
          attack_kwargs={"eps": args.eps, "alpha": args.alpha, "steps": args.steps})

    print("\n[Phase 3] Evaluating defended model (clean + attacks)...")
    defended_res = evaluate_all(defended_model, X_test_t, y_test_t, **eval_kwargs)

    # --- Report ---
    print_table(baseline_res, defended_res)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {
        "config": vars(args),
        "baseline": baseline_res,
        "defended": defended_res,
    }
    with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {os.path.join(RESULTS_DIR, 'results.json')}")
    save_plot(baseline_res, defended_res,
              os.path.join(RESULTS_DIR, "comparison.png"))


if __name__ == "__main__":
    main()
