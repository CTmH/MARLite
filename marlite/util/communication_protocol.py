"""Bounded message envelopes and exact retransmission deduplication.

Transport is synchronous in MARLite; this utility does not simulate radio delays.
"""

from dataclasses import dataclass
import struct

_HEADER = struct.Struct("<BHHHBB")


@dataclass(frozen=True)
class MessagePacket:
    sender: int
    step: int
    group_version: int
    x: int
    y: int
    payload: bytes

    def encode(self):
        fields = (self.sender, self.step, self.group_version, self.x, self.y)
        limits = (65535, 65535, 65535, 255, 255)
        if any(
            type(v) is not int or not 0 <= v <= limit
            for v, limit in zip(fields, limits)
        ):
            raise ValueError("Message header exceeds version-1 field limits")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("Message payload must be nonempty bytes")
        return _HEADER.pack(1, *fields) + self.payload

    @classmethod
    def decode(cls, data, payload_bytes):
        if payload_bytes < 1 or len(data) != _HEADER.size + payload_bytes:
            raise ValueError("Unexpected message length")
        version, *fields = _HEADER.unpack(data[: _HEADER.size])
        if version != 1:
            raise ValueError("Unknown message protocol version")
        return cls(*fields, data[_HEADER.size :])


def snapshot_packets(packets, step, group_version, expected_senders):
    """Return one exact leaf message per sender; reject mixed/incomplete snapshots."""
    unique = {}
    expected = set(expected_senders)
    for packet in packets:
        packet.encode()  # Validate bounded fields, including directly constructed packets.
        if packet.step != step or packet.group_version != group_version:
            raise ValueError("Stale or mismatched snapshot")
        if packet.sender not in expected:
            raise ValueError("Sender is outside this group snapshot")
        if packet.sender in unique and unique[packet.sender] != packet:
            raise ValueError("Conflicting messages for the same sender event")
        unique[packet.sender] = packet
    if set(unique) != expected:
        raise ValueError("Incomplete group snapshot")
    return [unique[sender] for sender in sorted(expected)]
