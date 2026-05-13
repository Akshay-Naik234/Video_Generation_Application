"""Render performance optimizations for the video generation pipeline.

Provides utilities to reduce video generation time without sacrificing
visible quality:

- Batch image pre-loading via ThreadPoolExecutor
- Pre-allocated frame buffer (avoids per-frame malloc)
- Smart resize strategy (skip resize when source matches target)

Safety:
- Thread pool is bounded and properly shut down
- No global state — all state is instance-scoped
"""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
from PIL import Image as PILImage

logger = logging.getLogger(__name__)


class RenderOptimizer:
    """Optimizes rendering performance across the video pipeline.

    Usage:
        optimizer = RenderOptimizer(resolution=(1920, 1080))
        # Pre-load images in parallel
        images = optimizer.parallel_load_images(image_paths)
        # Use pre-allocated buffer for frame operations
        buffer = optimizer.get_frame_buffer()
        # Clean up when done
        optimizer.cleanup()
    """

    MAX_WORKERS = None  # Set in __init__ based on CPU count

    def __init__(
        self,
        resolution: Tuple[int, int] = (1920, 1080),
        max_workers: Optional[int] = None,
    ):
        self.width, self.height = resolution
        self.resolution = resolution
        # Bound thread pool to avoid over-subscription
        cpu_count = os.cpu_count() or 4
        self.MAX_WORKERS = max_workers or min(cpu_count, 6)
        # Pre-allocated frame buffer for in-place operations
        self._frame_buffer: Optional[np.ndarray] = None
        self._pool: Optional[ThreadPoolExecutor] = None

    def get_frame_buffer(self) -> np.ndarray:
        """Return a pre-allocated frame buffer for zero-copy frame building.

        The buffer is allocated once and reused across frames to avoid
        per-frame memory allocation overhead. Caller should copy the
        buffer content before the next frame if needed.
        """
        if self._frame_buffer is None:
            self._frame_buffer = np.zeros(
                (self.height, self.width, 3), dtype=np.uint8
            )
        return self._frame_buffer

    def _get_pool(self) -> ThreadPoolExecutor:
        """Lazily create and return the thread pool."""
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.MAX_WORKERS)
        return self._pool

    def parallel_load_images(
        self,
        image_paths: List[Path],
        target_size: Optional[Tuple[int, int]] = None,
    ) -> List[Optional[PILImage.Image]]:
        """Load and optionally resize images in parallel using ThreadPoolExecutor.

        Returns images in the same order as input paths. Failed loads return
        None (logged as warnings, never crash the pipeline).

        Args:
            image_paths: List of image file paths.
            target_size: Optional (width, height) to resize during load.
                         Resizing during load is faster than loading full-res
                         then resizing later, as it skips decoding unused pixels.
        """
        pool = self._get_pool()
        results: List[Optional[PILImage.Image]] = [None] * len(image_paths)

        def _load_one(idx: int, path: Path) -> Tuple[int, Optional[PILImage.Image]]:
            try:
                img = PILImage.open(path)
                img.load()
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                if target_size and img.size != target_size:
                    img = img.resize(target_size, PILImage.LANCZOS)
                return idx, img
            except Exception as e:
                logger.warning("Failed to load image %s: %s", path, e)
                return idx, None

        futures = {
            pool.submit(_load_one, i, p): i
            for i, p in enumerate(image_paths)
        }

        for future in as_completed(futures):
            idx, img = future.result()
            results[idx] = img

        loaded = sum(1 for r in results if r is not None)
        logger.info("Parallel loaded %d/%d images (%d workers)",
                     loaded, len(image_paths), self.MAX_WORKERS)
        return results

    def smart_resize(
        self,
        image: np.ndarray,
        target_w: int,
        target_h: int,
        fast_mode: bool = False,
    ) -> np.ndarray:
        """Resize only when source dimensions differ from target.

        Skips the resize entirely if source already matches target, saving
        the LANCZOS/INTER_AREA computation. Uses cv2 when available for
        GPU-friendly resize, falls back to PIL.
        """
        h, w = image.shape[:2]
        if w == target_w and h == target_h:
            return image

        try:
            import cv2
            interp = cv2.INTER_AREA if fast_mode else cv2.INTER_LANCZOS4
            return cv2.resize(image, (target_w, target_h), interpolation=interp)
        except ImportError:
            pil_img = PILImage.fromarray(image)
            resample = PILImage.LANCZOS
            return np.array(pil_img.resize((target_w, target_h), resample))

    def cleanup(self) -> None:
        """Release resources and shut down thread pool.

        Safe to call multiple times. After cleanup, the optimizer can still
        be reused — pool will be lazily recreated on next use.
        """
        self._frame_buffer = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        logger.debug("RenderOptimizer: cleanup complete")

    def __del__(self):
        """Ensure cleanup on garbage collection."""
        try:
            self.cleanup()
        except Exception:
            pass
