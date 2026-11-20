"""QMPD group attention and scalar codecs as ordinary communication components."""

import torch
from torch import nn

from .fusion import CommunicationOutput


class GroupSetCommunication(nn.Module):
    """Two-level set aggregation followed by an individual query readout.

    The injected fusion module produces shared group slots. The reader queries
    those slots using each receiver's local history representation. Group slots
    describe the current decision frame, not an additional temporal memory.
    """

    def __init__(self, fusion, reader, *, header_bytes=0):
        super().__init__()
        if type(header_bytes) is not int or header_bytes < 0:
            raise ValueError("Header length must be a nonnegative integer")
        self.fusion, self.reader, self.header_bytes = fusion, reader, header_bytes

    def forward(self, messages, alive, context, *, queries=None):
        groups = context.get("groups")
        if groups is None or groups.dtype != torch.long or groups.shape != alive.shape:
            raise ValueError("Group attention requires recorded int64 groups")
        if queries is None or queries.shape != messages.shape:
            raise ValueError("Group reader queries must match sender feature axes/width")
        leading, n = alive.shape[:-1], alive.shape[-1]
        tokens = messages.reshape(-1, n, messages.shape[-1])
        live, gid = alive.reshape(-1, n), groups.reshape(-1, n)
        memory, valid = self.fusion(tokens, gid, live)
        selected = gid.masked_fill(~live, 0)
        rows = torch.arange(tokens.shape[0], device=tokens.device)[:, None]
        per_agent = memory[rows, selected].flatten(0, 1)
        query = queries.reshape(-1, 1, queries.shape[-1])
        read, _ = self.reader(query, per_agent, per_agent, need_weights=False)
        features = read.reshape(*alive.shape, -1).masked_fill(~alive[..., None], 0)
        memory = memory.reshape(*leading, *memory.shape[1:])
        valid = valid.reshape(*leading, -1)
        size = messages.shape[-1] * messages.element_size() + self.header_bytes
        return CommunicationOutput(
            features,
            {
                "group_tokens": memory,
                "group_valid": valid,
                "groups": groups,
                "relation_context": memory.mean(-2),
                "decoded_messages": messages,
                "record.communication_bytes": alive.to(torch.long) * size,
            },
        )


class QuantizedCommunication(nn.Module):
    """Local hard encoding, then the same aggregation/readout as a raw teacher.

    Neither the codec nor the sender representation receives group IDs or other
    agents' features. Recorded codes are unpacked integers; pack() defines the
    payload bytes. The header budget is a simulation convention, not measured
    radio traffic or a latency model.
    """

    def __init__(self, codec, aggregation, *, header_bytes=0):
        super().__init__()
        if type(header_bytes) is not int or header_bytes < 0:
            raise ValueError("Header length must be a nonnegative integer")
        self.codec, self.aggregation, self.header_bytes = codec, aggregation, header_bytes

    def forward(self, messages, alive, context, *, queries=None):
        decoded, codes = self.codec(messages)
        if decoded.shape != messages.shape:
            raise ValueError("Codec decoder must restore sender feature width")
        received = context.get("received_codes")
        if received is not None:
            replace = context.get("replace_senders", alive)
            if received.shape != codes.shape or received.dtype != torch.uint8:
                raise ValueError("Received codes must match uint8 sender code axes")
            if replace.shape != alive.shape or replace.dtype != torch.bool:
                raise ValueError("Code replacement mask must be bool over sender axes")
            replace = replace & alive
            # Unused/dead packet slots are not received information and need not
            # contain legal quantizer levels. Validate only the replaced slots.
            received = received.masked_fill(~replace[..., None], 0)
            received_features = self.codec.decode(received)
            codes = torch.where(replace[..., None], received, codes)
            decoded = torch.where(replace[..., None], received_features, decoded)
        codes = codes.masked_fill(~alive[..., None], 0)
        decoded = decoded.masked_fill(~alive[..., None], 0)
        result = self.aggregation(decoded, alive, context, queries=queries)
        size = self.codec.message_dim * self.codec.bits // 8 + self.header_bytes
        result.diagnostics.update(
            {
                "message_codes": codes,
                "record.message_codes": codes,
                "record.communication_bytes": alive.to(torch.long) * size,
            }
        )
        return result
