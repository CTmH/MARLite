"""Graph communication with recorded availability and differentiable graph weights."""

import torch
from torch import nn

from .fusion import CommunicationOutput, gaussian, draw_latent


class GraphCommunication(nn.Module):
    """Static adjacency or a learned graph, followed by a graph model.

    Adjacency axes are receiver,sender. Learned weights are recomputed inside
    autograd; recorded availability can restrict them, never replace them.
    """

    def __init__(self, graph_model, *, learned_graph=None, normalize=True,
                 probabilistic=False, latent_dim=None, sampled_policy=False):
        super().__init__()
        if sampled_policy and (not probabilistic or latent_dim is None):
            raise ValueError("Sampled graph latents need a Gaussian width")
        self.graph_model, self.learned_graph = graph_model, learned_graph
        self.normalize = normalize
        self.probabilistic, self.latent_dim, self.sampled_policy = probabilistic, latent_dim, sampled_policy

    def sample_context(self, alive, generator):
        if self.sampled_policy:
            return {"latent_noise": torch.randn(*alive.shape, self.latent_dim, generator=generator)}
        return {}

    def forward(self, messages, alive, context, *, queries=None):
        leading, n = alive.shape[:-1], alive.shape[-1]
        flat = messages.reshape(-1, n, messages.shape[-1]).masked_fill(~alive.reshape(-1, n, 1), 0)
        live = alive.reshape(-1, n)
        valid = live[:, :, None] & live[:, None, :]
        external = context.get("adjacency")
        if external is not None:
            if external.shape != (*alive.shape, n) or not torch.isfinite(external).all() or (external < 0).any():
                raise ValueError("Adjacency must be finite nonnegative ...,N,N")
            external = external.reshape(-1, n, n).to(flat)
        if self.learned_graph is None:
            if external is None:
                raise ValueError("Static graph communication requires recorded adjacency")
            adjacency = external
        else:
            learned = self.learned_graph(flat, live)
            adjacency = learned[0] * learned[1] if isinstance(learned, tuple) else learned
            if external is not None:
                adjacency = adjacency * external
        adjacency = adjacency.masked_fill(~valid, 0)
        if self.normalize:
            adjacency = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1e-8)
        result = self.graph_model(flat, adjacency).reshape(*alive.shape, -1)
        diagnostics = {"adjacency": adjacency.reshape(*leading, n, n),
                       "record.adjacency": adjacency.reshape(*leading, n, n)}
        if self.probabilistic:
            mean, log_var = gaussian(result)
            if self.latent_dim is not None and mean.shape[-1] != self.latent_dim:
                raise ValueError("Graph Gaussian width mismatch")
            result = draw_latent(mean, log_var, context, self.sampled_policy)
            diagnostics.update(agent_mu=mean, agent_log_var=log_var)
        result = result.masked_fill(~alive[..., None], 0)
        diagnostics["local_state_estimates"] = result
        return CommunicationOutput(result, diagnostics)


class DenseGraphModel(nn.Module):
    """Matrix graph layers; a component list controls all widths and activations."""

    def __init__(self, layers, output):
        super().__init__()
        self.layers, self.output = nn.ModuleList(layers), output

    def forward(self, tokens, adjacency):
        for layer in self.layers:
            tokens = layer(adjacency @ tokens)
        return self.output(tokens)


class GeometricGraphModel(nn.Module):
    """Apply an installed PyG model to a block-diagonal batched graph."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens, adjacency):
        b, n, _ = tokens.shape
        graph, receiver, sender = (adjacency != 0).nonzero(as_tuple=True)
        edges = torch.stack((graph * n + sender, graph * n + receiver))
        return self.model(tokens.flatten(0, 1), edges).reshape(b, n, -1)
