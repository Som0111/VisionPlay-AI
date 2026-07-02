"""The ``Frame`` value object every vision module passes around.

A :class:`Frame` is a thin wrapper over one captured image: the pixel array
plus identifying metadata (sequence number, capture timestamp, dimensions,
color format) and an initially empty ``results`` slot that inference
backends fill in downstream (``docs/architecture.md`` §4). It carries no
processing logic — transformations belong to pipeline stages and app
``processor.py`` modules, not to the frame itself.

Frames are metadata-immutable: every field is fixed at construction except
the ``results`` mapping, which the pipeline populates in place as backends
run (the dict is mutable; the binding is not).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["ColorFormat", "Frame"]

#: Pixel array type for frames: HxW (grayscale) or HxWxC, dtype uint8.
ImageArray = npt.NDArray[np.uint8]


class ColorFormat(Enum):
    """Channel layout of a frame's pixel array.

    OpenCV capture yields BGR, MediaPipe expects RGB — carrying the format
    on the frame makes conversions explicit instead of guessed.
    """

    BGR = "bgr"
    RGB = "rgb"
    GRAY = "gray"


@dataclass(frozen=True, slots=True)
class Frame:
    """One captured camera image with metadata and an inference-results slot.

    Attributes:
        frame_id: Monotonic sequence number assigned by the capture source.
        timestamp: Capture time as Unix seconds (``time.time()`` domain).
        image: Pixel data, shape ``(height, width, channels)`` — or
            ``(height, width)`` for :attr:`ColorFormat.GRAY` — dtype uint8.
        width: Image width in pixels. Must match ``image.shape[1]``.
        height: Image height in pixels. Must match ``image.shape[0]``.
        color_format: Channel layout of ``image``. Defaults to BGR, the
            OpenCV capture native order.
        results: Inference outputs keyed by backend name (e.g.
            ``"mediapipe.pose"``); empty at capture, filled by the pipeline.
    """

    frame_id: int
    timestamp: float
    image: ImageArray
    width: int
    height: int
    color_format: ColorFormat = ColorFormat.BGR
    results: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject metadata that contradicts the pixel array.

        Raises:
            ValueError: If ``width``/``height`` disagree with ``image.shape``
                — stale dimensions are worse than none.
        """
        h, w = self.image.shape[:2]
        if (self.height, self.width) != (h, w):
            raise ValueError(
                f"Frame metadata ({self.width}x{self.height}) does not match "
                f"image shape ({w}x{h})"
            )

    @classmethod
    def from_image(
        cls,
        frame_id: int,
        timestamp: float,
        image: ImageArray,
        color_format: ColorFormat = ColorFormat.BGR,
    ) -> Frame:
        """Build a frame with width/height derived from the array itself.

        Preferred constructor for capture sources — it makes a
        metadata/array mismatch impossible.

        Args:
            frame_id: Monotonic sequence number.
            timestamp: Capture time (Unix seconds).
            image: Pixel array, ``(H, W[, C])`` uint8.
            color_format: Channel layout of ``image``.
        """
        h, w = image.shape[:2]
        return cls(
            frame_id=frame_id,
            timestamp=timestamp,
            image=image,
            width=w,
            height=h,
            color_format=color_format,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        """The underlying array's shape, ``(height, width[, channels])``."""
        return self.image.shape

    @property
    def size(self) -> tuple[int, int]:
        """Image dimensions as ``(width, height)`` — OpenCV argument order."""
        return (self.width, self.height)
