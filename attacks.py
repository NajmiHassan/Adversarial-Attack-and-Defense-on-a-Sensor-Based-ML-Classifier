import torch
import torch.nn.functional as F

CLIP_MIN, CLIP_MAX = -1.0, 1.0


def fgsm(model, x, y, eps):
    """Fast Gradient Sign Method. Returns adversarial examples (detached)."""
    x = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    x_adv = x + eps * grad.sign()
    return torch.clamp(x_adv, CLIP_MIN, CLIP_MAX).detach()


def pgd(model, x, y, eps, alpha, steps):
    """Projected Gradient Descent (L-inf) with a random start."""
    x_orig = x.clone().detach()
    # Random start inside the eps-ball.
    x_adv = x_orig + torch.empty_like(x_orig).uniform_(-eps, eps)
    x_adv = torch.clamp(x_adv, CLIP_MIN, CLIP_MAX)

    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        loss = F.cross_entropy(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()
        # Project back into the eps-ball around the original, then into [-1, 1].
        x_adv = torch.min(torch.max(x_adv, x_orig - eps), x_orig + eps)
        x_adv = torch.clamp(x_adv, CLIP_MIN, CLIP_MAX)

    return x_adv.detach()
