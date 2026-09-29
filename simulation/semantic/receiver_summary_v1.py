"""Compact causal receiver covariance summary for the controlled V4 task.

This is an experimental application payload, not a standards CPM.  It carries
three target identities and quantized receiver variances; the ns-3 transport
adds its existing 20-byte SeqTsSizeHeader.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct


HEADER = struct.Struct("<BBHI")  # version, format, cache version, generation ms
ENTRY = struct.Struct("<BH")    # target id, variance in 1e-4 m^2 units
TARGETS = 3
VERSION = 3
FORMAT = 1


@dataclass(frozen=True)
class ReceiverSummary:
    cache_version: int
    generation_ms: int
    variances: tuple[float, float, float]


def encode_summary(summary: ReceiverSummary) -> bytes:
    if not (0 <= summary.cache_version <= 65535 and
            0 <= summary.generation_ms <= 2**32 - 1 and
            len(summary.variances) == TARGETS):
        raise ValueError("Invalid summary identity")
    payload = bytearray(HEADER.pack(VERSION, FORMAT, summary.cache_version,
                                    summary.generation_ms))
    for target_id, variance in enumerate(summary.variances):
        if not math.isfinite(variance) or not 0 <= variance <= 6.5535:
            raise ValueError("Receiver variance outside quantizer range")
        payload.extend(ENTRY.pack(target_id, round(variance * 10000)))
    return bytes(payload)


def decode_summary(raw: bytes) -> ReceiverSummary:
    if len(raw) != HEADER.size + TARGETS * ENTRY.size:
        raise ValueError("Receiver summary length mismatch")
    version, fmt, cache_version, generation_ms = HEADER.unpack_from(raw)
    if (version, fmt) != (VERSION, FORMAT):
        raise ValueError("Receiver summary version mismatch")
    variances = []
    for expected_id in range(TARGETS):
        target_id, quantized = ENTRY.unpack_from(raw, HEADER.size + expected_id * ENTRY.size)
        if target_id != expected_id:
            raise ValueError("Receiver summary target order mismatch")
        variances.append(quantized / 10000)
    return ReceiverSummary(cache_version, generation_ms, tuple(variances))
