"""Requested recurrent SAC network definitions (1x900 LSTM, 3x400 MLPs)."""
import math
from typing import Sequence
import torch
from torch import nn

from .config import ObservationConfig


class RecurrentEncoder(nn.Module):
    def __init__(self, observation_dim: int = ObservationConfig().dimension, hidden_size: int = 900, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(observation_dim, hidden_size, layers, batch_first=True)

    def forward(self, observation_sequence: torch.Tensor, lengths=None) -> torch.Tensor:
        if observation_sequence.ndim not in (2, 3) or observation_sequence.shape[-1] != self.lstm.input_size:
            raise ValueError("LSTM observation dimension mismatch")
        if observation_sequence.ndim == 2:
            observation_sequence = observation_sequence.unsqueeze(1)
        if lengths is None:
            output, _ = self.lstm(observation_sequence)
            return output[:, -1]
        # Lengths are host metadata. Never read accelerator tensors to branch.
        if isinstance(lengths, torch.Tensor):
            if lengths.device.type != "cpu":
                raise ValueError("Pass real sequence lengths as CPU metadata")
            values = lengths.numpy()
        else:
            import numpy as np
            values = np.asarray(lengths)
        if values.shape != (observation_sequence.shape[0],) or (values < 1).any() or (values > observation_sequence.shape[1]).any():
            raise ValueError("Invalid real sequence lengths")
        if self.lstm.bidirectional:
            raise ValueError("Causal padded batching requires a unidirectional LSTM")
        # Later right-padding cannot affect an earlier causal output or its gradient.
        # Only the output at the final real frame contributes to the losses.
        output, _ = self.lstm(observation_sequence)
        last = torch.as_tensor(values.copy(), dtype=torch.long, device=output.device) - 1
        return output.gather(1, last[:, None, None].expand(-1, 1, output.shape[-1])).squeeze(1)

    def step(self, observation, history=None, window_size=8):
        """Return latent and a bounded real-frame window, never persistent h/c."""
        if observation.ndim != 1 or observation.shape[0] != self.lstm.input_size:
            raise ValueError("Online LSTM observation dimension mismatch")
        if window_size < 1:
            raise ValueError("History window must be positive")
        if history is None:
            window = observation.unsqueeze(0)
        else:
            if not isinstance(history, torch.Tensor) or history.ndim != 2 or history.shape[-1] != self.lstm.input_size:
                raise ValueError("History must be a real-frame window, not legacy LSTM h/c")
            window = torch.cat((history.to(observation), observation.unsqueeze(0)), dim=0)
        window = window[-window_size:]
        latent = self.forward(window.unsqueeze(0))
        return latent, window.detach()


def mlp(input_dim: int, hidden: Sequence[int], output_dim: int) -> nn.Sequential:
    layers = []
    previous = input_dim
    for width in hidden:
        layers.extend((nn.Linear(previous, width), nn.ReLU()))
        previous = width
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class GaussianActor(nn.Module):
    def __init__(self, latent_dim=900, action_dim=3, hidden=(400, 400, 400)):
        super().__init__()
        self.trunk = mlp(latent_dim, hidden, 2 * action_dim)
        self.action_dim = action_dim

    def forward(self, latent):
        mean, log_std = self.trunk(latent).split(self.action_dim, dim=-1)
        return mean, log_std.clamp(-20.0, 2.0)

    def sample(self, latent):
        mean, log_std = self(latent)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std, validate_args=False)
        pre_tanh = normal.rsample()
        action = torch.tanh(pre_tanh)
        log_prob = normal.log_prob(pre_tanh) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)


class TwinCritic(nn.Module):
    def __init__(self, latent_dim=900, action_dim=3, hidden=(400, 400, 400)):
        super().__init__()
        self.q1 = mlp(latent_dim + action_dim, hidden, 1)
        self.q2 = mlp(latent_dim + action_dim, hidden, 1)

    def forward(self, latent, action):
        features = torch.cat((latent, action), dim=-1)
        return self.q1(features), self.q2(features)


class SACNetworks(nn.Module):
    def __init__(self, observation_dim=ObservationConfig().dimension, action_dim=3, lstm_nodes=900, lstm_layers=1, hidden=(400, 400, 400), critic_hidden=None):
        super().__init__()
        self.encoder = RecurrentEncoder(observation_dim, lstm_nodes, lstm_layers)
        self.actor = GaussianActor(lstm_nodes, action_dim, hidden)
        self.critics = TwinCritic(lstm_nodes, action_dim, hidden if critic_hidden is None else critic_hidden)

    def policy(self, observation_sequence, lengths=None):
        return self.actor.sample(self.encoder(observation_sequence, lengths))
