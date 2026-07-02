"""Compute-device selection for inference backends.

v1 is CPU-only (``docs/architecture.md`` §5), but the *shape* of device
selection exists from day one so GPU support later is additive: call sites
construct a :class:`DeviceConfig` from configuration and pass it through to
backend constructors without ever branching on the device themselves. When
GPU support lands, :class:`DeviceType` gains a member and each backend's
device→runtime mapping gains an entry — nothing in the surrounding pipeline
changes.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["DeviceConfig", "DeviceType"]


class DeviceType(Enum):
    """Kind of compute device a backend should run on.

    v1 defines only :attr:`CPU`. GPU variants (e.g. ``DirectML``, ``CUDA``)
    are added here later as new members — an additive change, never a
    replacement of this enum.
    """

    CPU = "cpu"


@dataclass(frozen=True, slots=True)
class DeviceConfig:
    """Immutable description of the device an inference backend runs on.

    Every backend constructor accepts one of these instead of hardcoding
    CPU internally. Today it is a single field; GPU support may add fields
    (device index, memory limits) without breaking existing call sites
    because construction goes through :meth:`cpu` / :meth:`from_mapping`.

    Attributes:
        type: The compute device kind. Defaults to CPU, the only v1 option.
    """

    type: DeviceType = DeviceType.CPU

    @classmethod
    def cpu(cls) -> DeviceConfig:
        """Return the CPU device config — the v1 default everywhere."""
        return cls(type=DeviceType.CPU)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> DeviceConfig:
        """Build a config from a ``config.yaml``-style mapping.

        Accepts the documented shape ``{"type": "cpu"}``; a missing
        ``type`` key defaults to CPU. Matching is case-insensitive.

        Args:
            data: Mapping with an optional ``type`` key naming the device.

        Returns:
            The resolved :class:`DeviceConfig`.

        Raises:
            ValueError: If ``type`` names an unknown device — the message
                lists the supported values so users can fix their config.
        """
        raw = data.get("type", DeviceType.CPU.value)
        try:
            device_type = DeviceType(str(raw).lower())
        except ValueError:
            supported = ", ".join(member.value for member in DeviceType)
            raise ValueError(f"Unknown device type {raw!r} (supported: {supported})") from None
        return cls(type=device_type)
