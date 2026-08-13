"""Auditable PPO/GAE update with explicit finite checks."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PPOStats:
    policy_loss: float
    value_loss: float
    entropy: float
    kl: float
    grad_norm: float


def ppo_update(model: nn.Module, optimizer: torch.optim.Optimizer, states: torch.Tensor, actions: torch.Tensor, old_logprob: torch.Tensor, returns: torch.Tensor, advantages: torch.Tensor, clip: float, entropy_coef: float, value_coef: float, *, action_temperature: float = 1.0) -> PPOStats:
    if not all(torch.isfinite(x).all() for x in (states, old_logprob, returns, advantages)):
        raise FloatingPointError("non-finite PPO inputs")
    if not 0.0 < action_temperature:
        raise ValueError("action_temperature must be positive")
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    logits, values = model(states)
    distribution = torch.distributions.Categorical(logits=logits / action_temperature)
    logprob = distribution.log_prob(actions)
    ratio = (logprob - old_logprob).exp()
    clipped = torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
    policy_loss = -torch.minimum(ratio * advantages, clipped * advantages).mean()
    value_loss = torch.nn.functional.mse_loss(values, returns)
    entropy = distribution.entropy().mean()
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite PPO loss")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
    if not torch.isfinite(torch.tensor(grad_norm)):
        raise FloatingPointError("non-finite gradient norm")
    optimizer.step()
    return PPOStats(float(policy_loss.detach()), float(value_loss.detach()), float(entropy.detach()), float((old_logprob - logprob).mean().detach()), grad_norm)
