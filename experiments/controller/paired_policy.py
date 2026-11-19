"""A conditional, single-slot skill policy on top of a frozen MemSkill Controller."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class PairedPolicy(nn.Module):
    """Keep the other K-1 actions and their positions fixed on every decision."""

    def __init__(self, native, target):
        super().__init__()
        self.native = native
        self.target = target
        self.mode = "explore"
        self.weights = None
        self.decisions = []

    def forward(self, state, candidates, deterministic=False, new_op_mask=None, *, record_noise=None, replay_noise=None):
        with torch.no_grad():
            h = self.native.encode_state(state[None])
            logits = self.native.get_action_logits(h, self.native.encode_ops(candidates[None]))[0]
            logits = self.native._apply_new_action_bias(logits, new_op_mask)
            k = self.native.action_top_k
            if not 1 <= k < logits.numel():
                raise ValueError("A single-slot alternative needs an unused candidate")
            plain = logits.topk(k).indices.tolist()
            if self.target in plain:
                slot = plain.index(self.target)
                unused = logits.clone()
                unused[plain] = -torch.inf
                plain[slot] = int(unused.argmax())
            else:
                slot = k - 1
            grounded = plain.copy()
            grounded[slot] = self.target
            offset = float(logits[self.target] - logits[plain[slot]])
            features = torch.cat((state.new_ones(1), F.normalize(state, dim=0)))
            logit = offset if self.weights is None else offset + float(features @ self.weights.to(features))
            prob = float(torch.sigmoid(state.new_tensor(logit)))
            if self.mode == "explore":
                if deterministic:
                    raise ValueError("Exploration is for sampled training rollouts")
                coin = float(torch.rand(())) if replay_noise is None else replay_noise["uniform"]
                use = coin < 0.5
                prob = 0.5
            else:
                if not deterministic:
                    raise ValueError("Deployment uses greedy selection")
                use = self.mode == "always" or logit >= 0
                if self.mode == "always":
                    prob = 1.0
            selected, other = (grounded, plain) if use else (plain, grounded)
            if record_noise is not None:
                record_noise.update(uniform=coin, paired_alternative=other)
            if replay_noise is None:
                self.decisions.append({"use": use, "prob": prob, "base_logit": offset, "plain": plain,
                                       "grounded": grounded, "state": state.detach().cpu().tolist()})
            probability = prob if use else 1.0 - prob
            return selected, math.log(max(probability, 1e-30)), float(self.native.get_value(h).item())


def fit_policy(rows, credit_weight):
    """Fit a regularized logistic actor on balanced exploration, using the sampled action."""
    states = torch.tensor([r["state"] for r in rows], dtype=torch.float64)
    x = torch.cat((torch.ones_like(states[:, :1]), F.normalize(states, dim=1)), dim=1)
    offset = torch.tensor([r["base_logit"] for r in rows], dtype=x.dtype)
    sign = torch.tensor([1 if r["use"] else -1 for r in rows], dtype=x.dtype)
    epi = {r["episode"]: r["terminal"] for r in rows}
    if len(epi) < 2:
        raise ValueError("Leave-one-episode-out baseline needs multiple episodes")
    total = sum(epi.values())
    terminal = torch.tensor([r["terminal"] - (total-r["terminal"])/(len(epi)-1) for r in rows], dtype=x.dtype)
    credit = torch.tensor([max(-0.5, min(0.5, r["delta"])) for r in rows], dtype=x.dtype)
    active = credit != 0
    theta = torch.zeros(x.shape[1], dtype=x.dtype, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=100, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        logit = offset + x @ theta
        terminal_loss = (terminal.abs() * F.softplus(-terminal.sign() * sign * logit)).mean()
        loss = terminal_loss + 0.01 * theta.square().sum()
        if credit_weight and active.any():
            local = credit[active]
            loss = loss + credit_weight * (local.abs() * F.softplus(-local.sign() * sign[active] * logit[active])).mean()
        loss.backward()
        return loss

    opt.step(closure)
    return theta.detach().float()


def match_training_probability(rows, terminal, credit):
    """Shift only the terminal intercept to match the credit policy's mean training probability."""
    states = torch.tensor([row["state"] for row in rows], dtype=torch.float64)
    features = torch.cat((torch.ones_like(states[:, :1]), F.normalize(states, dim=1)), dim=1)
    offset = torch.tensor([row["base_logit"] for row in rows], dtype=states.dtype)
    base = offset + features @ terminal.double()
    extra = features @ (credit.double() - terminal.double())
    target = (base + extra).sigmoid().mean()
    # A common shift at either extreme bounds the mean of the state-dependent shifts.
    lo, hi = float(extra.min()), float(extra.max())
    for _ in range(60):
        mid = (lo + hi) / 2
        if (base + mid).sigmoid().mean() < target:
            lo = mid
        else:
            hi = mid
    result = terminal.clone()
    result[0] += (lo + hi) / 2
    return result
