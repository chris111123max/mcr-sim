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
        lengths = torch.as_tensor(lengths, dtype=torch.long, device=observation_sequence.device)
        if lengths.shape != (observation_sequence.shape[0],) or torch.any(lengths < 1) or torch.any(lengths > observation_sequence.shape[1]):
            raise ValueError("Invalid real sequence lengths")
        # Right-padding is storage only. Run each length group on real frames;
        # neither padding nor a repeated first frame enters the recurrent state.
        encoded = observation_sequence.new_zeros((observation_sequence.shape[0], self.lstm.hidden_size))
        for length in torch.unique(lengths).tolist():
            rows = torch.nonzero(lengths == length, as_tuple=False).flatten()
            sequence = observation_sequence.index_select(0, rows)[:, :length]
            output, _ = self.lstm(sequence)  # zero initial h/c for every window
            encoded = encoded.index_copy(0, rows, output[:, -1])
        return encoded

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
        normal = torch.distributions.Normal(mean, std)
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
    def __init__(self, observation_dim=ObservationConfig().dimension, action_dim=3, lstm_nodes=900, lstm_layers=1, hidden=(400, 400, 400)):
        super().__init__()
        self.encoder = RecurrentEncoder(observation_dim, lstm_nodes, lstm_layers)
        self.actor = GaussianActor(lstm_nodes, action_dim, hidden)
        self.critics = TwinCritic(lstm_nodes, action_dim, hidden)

    def policy(self, observation_sequence, lengths=None):
        return self.actor.sample(self.encoder(observation_sequence, lengths))
