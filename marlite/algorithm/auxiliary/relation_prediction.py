import torch
from torch import nn
from torch.nn import functional as F

from marlite.algorithm.model import ModelConfig


class RelationPredictionHead(nn.Module):
    """Predict one binary label per group/query pair from training-only queries."""

    def __init__(self, model):
        super().__init__()
        self.model = ModelConfig(**model).get_model()

    def forward(self, context, queries):
        context = context.unsqueeze(-2).expand(*queries.shape[:-1], -1)
        logits = self.model(torch.cat((context, queries), dim=-1))
        if logits.shape != (*queries.shape[:-1], 1):
            raise ValueError("Auxiliary prediction model must output one logit per query")
        return logits.squeeze(-1)

    def loss(self, result, batch, label_provider):
        context = result["relation_context"]
        valid = result["group_valid"]
        targets = label_provider(batch, valid, context.device)
        queries = targets["relation_queries"].to(context)
        labels = targets["relation_labels"].to(context)
        mask = targets["relation_mask"].to(device=context.device, dtype=torch.bool)
        if (
            queries.shape[:2] != valid.shape
            or queries.shape[:-1] != labels.shape
            or mask.shape != labels.shape
        ):
            raise ValueError("Auxiliary targets must match the group and query axes")

        mask = mask & valid.unsqueeze(-1)
        logits = self(context, queries)
        valid_labels = labels[mask]
        if not torch.all((valid_labels >= 0) & (valid_labels <= 1)):
            raise ValueError("Unmasked auxiliary labels must be finite and in [0,1]")
        loss = F.binary_cross_entropy_with_logits(
            logits[mask], valid_labels, reduction="sum"
        )
        return loss / mask.sum().clamp_min(1)
