"""A configurable, training-only binary query predictor."""

import torch
from torch import nn


class QueryPredictionHead(nn.Module):
    """Predict a binary training-only query from group context and query features."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, context, queries):
        expanded = context.unsqueeze(-2).expand(*queries.shape[:-1], -1)
        logits = self.model(torch.cat((expanded, queries), -1))
        if logits.shape != (*queries.shape[:-1], 1):
            raise ValueError("Query predictor must output one scalar per query")
        return logits.squeeze(-1)

