"""AI-enhanced visual effects for cinematic video generation.

This module provides optional AI-powered effects that dramatically improve
video quality when available. All features gracefully fall back to numpy-based
alternatives when AI dependencies (torch, opencv) are not installed.

Tier 1 (numpy only - always available):
    - Pseudo-depth estimation via edge detection + gradient analysis
    - Depth-of-field bokeh blur using estimated depth
    - Subject center detection for smart zoom targeting
    - Dynamic weather/atmosphere overlays (rain, dust, embers, light particles)

Tier 2 (requires torch - optional, best quality):
    - MiDaS AI depth estimation for accurate depth maps
    - True 2.5D parallax Ken Burns using AI depth
    - High-quality depth-of-field with accurate subject isolation

All features are FREE and open source. No paid tools or API keys required.
"""

import math
from typing import Tuple, Optional

import numpy as np
from PIL import Image as PILImage, ImageFilter

# --- Dependency detection ---
_HAS_TORCH = False
_HAS_CV2 = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    pass

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    pass


class DepthEstimator:
    """Estimates depth maps from single images.

    Uses MiDaS (via torch.hub) when PyTorch is available for high-quality
    AI depth estimation. Falls back to a numpy-based pseudo-depth approach
    using edge detection and vertical gradient heuristics when torch is
    not installed.

    The depth maps are used for:
    - 2.5D parallax Ken Burns effects (foreground moves faster than background)
    - Depth-of-field cinematic blur (sharp subject, blurred background)
    - Smart zoom targeting (zoom toward the detected subject center)
    """

    def __init__(self, use_ai: bool = True):
        """Initialize the depth estimator.

        Args:
            use_ai: If True and torch is available, use MiDaS AI model.
                    If False or torch unavailable, use numpy pseudo-depth.
        """
        self._model = None
        self._transform = None
        self._device = None
        self._use_ai = use_ai and _HAS_TORCH
        self._model_loaded = False

    def _load_model(self) -> bool:
        """Lazy-load MiDaS model on first use. Returns True if successful."""
        if self._model_loaded:
            return self._model is not None

        self._model_loaded = True

        if not self._use_ai:
            return False

        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Use MiDaS small for speed — good enough for parallax effects
            model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
            model.to(device)
            model.eval()

            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
            transform = midas_transforms.small_transform

            self._model = model
            self._transform = transform
            self._device = device
            print("  AI Depth: MiDaS small model loaded successfully")
            return True
        except Exception as e:
            print(f"  AI Depth: MiDaS failed to load ({e}), using numpy fallback")
            self._model = None
            return False

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth map from an RGB image.

        Args:
            image: RGB image as numpy array (H, W, 3), uint8.

        Returns:
            Depth map as float32 array (H, W), normalized to [0, 1].
            Higher values = closer to camera (foreground).
        """
        if self._use_ai and self._load_model():
            return self._estimate_depth_ai(image)
        return self._estimate_depth_numpy(image)

    def _estimate_depth_ai(self, image: np.ndarray) -> np.ndarray:
        """Estimate depth using MiDaS AI model."""
        h, w = image.shape[:2]

        input_batch = self._transform(image).to(self._device)

        with torch.no_grad():
            prediction = self._model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(h, w),
                mode='bicubic',
                align_corners=False,
            ).squeeze()

        depth = prediction.cpu().numpy()

        # MiDaS outputs inverse depth (closer = higher value)
        # Normalize to [0, 1]
        depth_min = depth.min()
        depth_max = depth.max()
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)

        return depth.astype(np.float32)

    def _estimate_depth_numpy(self, image: np.ndarray) -> np.ndarray:
        """Estimate pseudo-depth using numpy heuristics.

        Combines three cues that approximate depth in typical photographs:
        1. Vertical gradient: objects lower in frame tend to be closer
        2. Edge density: areas with more detail tend to be closer (in focus)
        3. Brightness: darker areas often recede in photographs

        This is approximate but produces visually convincing parallax for
        most biography/documentary style images (portraits, landscapes, etc.).
        """
        h, w = image.shape[:2]

        # Convert to grayscale
        gray = np.mean(image.astype(np.float32), axis=2)

        # Cue 1: Vertical gradient (lower = closer, 40% weight)
        vertical = np.linspace(0.0, 1.0, h).reshape(h, 1)
        vertical = np.broadcast_to(vertical, (h, w))

        # Cue 2: Edge density (more edges = more detail = closer, 35% weight)
        # Use simple Sobel-like gradient magnitude
        gy = np.zeros_like(gray)
        gx = np.zeros_like(gray)
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        edges = np.sqrt(gx ** 2 + gy ** 2)
        # Smooth the edge map to create regions rather than hard edges
        # Use a simple box blur (average pooling)
        kernel_size = max(h, w) // 30
        if kernel_size > 1:
            edges = self._box_blur(edges, kernel_size)
        # Normalize
        e_max = edges.max()
        if e_max > 0:
            edges = edges / e_max

        # Cue 3: Center-weighted brightness (brighter center = subject, 25% weight)
        # Create a center-weighted map
        cy, cx = h / 2.0, w / 2.0
        yy, xx = np.ogrid[:h, :w]
        center_dist = np.sqrt(((yy - cy) / cy) ** 2 + ((xx - cx) / cx) ** 2)
        center_weight = 1.0 - np.clip(center_dist / 1.5, 0, 1)

        # Brightness as depth cue (normalized)
        brightness = gray / 255.0
        bright_center = brightness * center_weight

        # Combine cues
        depth = 0.40 * vertical + 0.35 * edges + 0.25 * bright_center

        # Normalize to [0, 1]
        d_min = depth.min()
        d_max = depth.max()
        if d_max - d_min > 0:
            depth = (depth - d_min) / (d_max - d_min)

        return depth.astype(np.float32)

    @staticmethod
    def _box_blur(arr: np.ndarray, size: int) -> np.ndarray:
        """Simple box blur using cumulative sum (fast, no scipy needed)."""
        if size < 2:
            return arr
        h, w = arr.shape
        # Pad array
        padded = np.pad(arr, size, mode='edge')
        # Cumulative sum approach for box blur
        cumsum = np.cumsum(np.cumsum(padded, axis=0), axis=1)
        # Extract blurred values using integral image
        result = (
            cumsum[2 * size:, 2 * size:]
            - cumsum[:-2 * size, 2 * size:]
            - cumsum[2 * size:, :-2 * size]
            + cumsum[:-2 * size, :-2 * size]
        )
        area = (2 * size) ** 2
        result = result / area
        # Trim to original size
        return result[:h, :w]


class ParallaxEngine:
    """Creates 2.5D parallax Ken Burns effects using depth maps.

    Instead of the flat Ken Burns effect where the entire image moves uniformly,
    this engine uses depth information to move foreground elements faster than
    background elements, creating a convincing 3D parallax illusion.

    This is the technique used by top creators like Shivanshu Agrawal and
    professional documentary channels to make still images feel alive.
    """

    def __init__(self, resolution: Tuple[int, int]):
        self.width, self.height = resolution

    def create_parallax_frame(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        progress: float,
        movement_type: str = 'zoom_in',
        intensity: float = 1.0,
        offset_x: float = 0.0,
        offset_y: float = 0.0,
    ) -> np.ndarray:
        """Create a single parallax frame with depth-aware movement.

        Args:
            image: RGB image (H, W, 3), uint8.
            depth_map: Depth map (H, W), float32, [0,1]. Higher = closer.
            progress: Animation progress [0, 1].
            movement_type: Type of movement (zoom_in, pan_left, etc.).
            intensity: Movement intensity multiplier.
            offset_x: Additional horizontal offset (e.g., micro-shake).
            offset_y: Additional vertical offset (e.g., micro-shake).

        Returns:
            Parallax-displaced frame (H, W, 3), uint8.
        """
        h, w = image.shape[:2]

        # Calculate base displacement based on movement type
        dx, dy, scale = self._get_displacement(movement_type, progress, intensity)

        # Apply micro-shake offsets from human-feel system
        dx += offset_x
        dy += offset_y

        # Create per-pixel displacement based on depth
        # Foreground (depth ~1.0) gets MORE displacement (moves more)
        # Background (depth ~0.0) gets LESS displacement (moves less)
        # This creates the parallax illusion
        foreground_factor = 1.5  # Foreground moves 1.5x base
        background_factor = 0.3  # Background moves 0.3x base

        depth_factor = background_factor + (foreground_factor - background_factor) * depth_map

        # Create displacement maps
        # Base grid
        yy, xx = np.mgrid[:h, :w].astype(np.float32)

        # Apply depth-weighted displacement
        map_x = xx + dx * w * depth_factor
        map_y = yy + dy * h * depth_factor

        # Apply depth-weighted scale
        # Zoom toward center with depth-aware scaling
        cx, cy = w / 2.0, h / 2.0
        if abs(scale - 1.0) > 0.001:
            scale_depth = 1.0 + (scale - 1.0) * depth_factor
            map_x = cx + (map_x - cx) / scale_depth
            map_y = cy + (map_y - cy) / scale_depth

        # Remap using numpy (bilinear interpolation)
        result = self._remap_bilinear(image, map_x, map_y)

        return result

    def _get_displacement(
        self, movement_type: str, progress: float, intensity: float
    ) -> Tuple[float, float, float]:
        """Calculate base displacement for the movement type.

        Returns:
            (dx, dy, scale) where dx/dy are fractional offsets and scale is zoom.
        """
        t = progress
        amt = 0.03 * intensity  # Base displacement amount

        if movement_type == 'zoom_in':
            return 0, 0, 1.0 + 0.15 * intensity * t
        elif movement_type == 'zoom_out':
            return 0, 0, 1.0 + 0.15 * intensity * (1.0 - t)
        elif movement_type == 'pan_left':
            return -amt * t, 0, 1.0
        elif movement_type == 'pan_right':
            return amt * t, 0, 1.0
        elif movement_type == 'pan_up':
            return 0, -amt * t, 1.0
        elif movement_type == 'pan_down':
            return 0, amt * t, 1.0
        elif movement_type == 'push_in':
            return 0, -0.005 * t, 1.0 + 0.12 * intensity * t
        elif movement_type == 'pull_out':
            return 0, 0.005 * t, 1.0 + 0.12 * intensity * (1.0 - t)
        elif movement_type in ('gentle_drift', 'float_drift'):
            dx = amt * math.sin(t * math.pi * 2)
            dy = amt * 0.5 * math.cos(t * math.pi * 1.3)
            return dx, dy, 1.0 + 0.03 * math.sin(t * math.pi * 1.5)
        elif movement_type == 'diagonal_tl_br':
            return amt * 0.7 * t, amt * 0.7 * t, 1.0 + 0.08 * intensity * t
        elif movement_type == 'diagonal_tr_bl':
            return -amt * 0.7 * t, amt * 0.7 * t, 1.0 + 0.08 * intensity * t
        else:
            # Default: subtle zoom in
            return 0, 0, 1.0 + 0.08 * intensity * t

    @staticmethod
    def _remap_bilinear(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray) -> np.ndarray:
        """Bilinear interpolation remap using pure numpy.

        This replaces cv2.remap for environments without OpenCV.
        """
        h, w = image.shape[:2]

        # Clamp coordinates
        map_x = np.clip(map_x, 0, w - 1.001)
        map_y = np.clip(map_y, 0, h - 1.001)

        # Integer and fractional parts
        x0 = map_x.astype(np.int32)
        y0 = map_y.astype(np.int32)
        x1 = np.minimum(x0 + 1, w - 1)
        y1 = np.minimum(y0 + 1, h - 1)

        fx = (map_x - x0).astype(np.float32)
        fy = (map_y - y0).astype(np.float32)

        # Bilinear interpolation for each channel
        if image.ndim == 3:
            fx = fx[:, :, np.newaxis]
            fy = fy[:, :, np.newaxis]

        val = (
            image[y0, x0] * (1 - fx) * (1 - fy)
            + image[y0, x1] * fx * (1 - fy)
            + image[y1, x0] * (1 - fx) * fy
            + image[y1, x1] * fx * fy
        )

        return np.clip(val, 0, 255).astype(np.uint8)


class DepthOfFieldEffect:
    """Applies cinematic depth-of-field blur using depth maps.

    Creates a professional bokeh-like effect where the subject (foreground)
    stays sharp while the background is smoothly blurred. This mimics the
    look of expensive cinema lenses and is used by top creators to make
    AI-generated images look more cinematic.
    """

    def apply(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        blur_strength: float = 8.0,
        focus_point: float = 0.7,
    ) -> np.ndarray:
        """Apply depth-of-field blur.

        Args:
            image: RGB image (H, W, 3), uint8.
            depth_map: Depth map (H, W), float32, [0,1].
            blur_strength: Maximum blur radius in pixels.
            focus_point: Depth value that should be in focus (0-1).
                        Higher = focus on foreground.

        Returns:
            Blurred image with depth-of-field effect.
        """
        pil_img = PILImage.fromarray(image)

        # Create multiple blur levels for smooth bokeh gradient
        blur_levels = 4
        blurred_images = []
        for i in range(blur_levels):
            radius = blur_strength * (i + 1) / blur_levels
            blurred = pil_img.filter(ImageFilter.GaussianBlur(radius=radius))
            blurred_images.append(np.array(blurred))

        # Calculate blur amount per pixel based on distance from focus point
        focus_distance = np.abs(depth_map - focus_point)
        # Normalize to [0, 1] where 0 = in focus, 1 = max blur
        max_dist = max(focus_point, 1.0 - focus_point)
        if max_dist > 0:
            blur_amount = np.clip(focus_distance / max_dist, 0, 1)
        else:
            blur_amount = np.zeros_like(depth_map)

        # Ease the blur for more natural look
        blur_amount = blur_amount ** 1.5  # Power curve for smoother falloff

        # Blend between sharp and blurred based on blur amount
        result = image.astype(np.float32)
        for i, blurred in enumerate(blurred_images):
            level_start = i / blur_levels
            level_end = (i + 1) / blur_levels
            # How much of this blur level to apply
            mask = np.clip((blur_amount - level_start) / (level_end - level_start), 0, 1)
            mask = mask[:, :, np.newaxis]
            if i == 0:
                result = image.astype(np.float32) * (1 - mask) + blurred.astype(np.float32) * mask
            else:
                result = result * (1 - mask) + blurred.astype(np.float32) * mask

        return np.clip(result, 0, 255).astype(np.uint8)


class SubjectDetector:
    """Detects the main subject position in an image using depth analysis.

    Used to target Ken Burns zoom/pan movements toward the actual subject
    instead of always targeting the frame center. This makes movements
    feel intentional and cinematic rather than mechanical.
    """

    def detect_subject_center(
        self, depth_map: np.ndarray, image: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """Find the center of the main subject using depth map.

        Args:
            depth_map: Depth map (H, W), float32, [0,1].
            image: Optional RGB image for additional analysis.

        Returns:
            (cx, cy) as fractions of image dimensions [0, 1].
        """
        h, w = depth_map.shape

        # The subject is typically the closest object (highest depth values)
        # Threshold at the 75th percentile to isolate foreground
        threshold = np.percentile(depth_map, 75)
        foreground_mask = (depth_map >= threshold).astype(np.float32)

        # Weight by depth value (closer = more weight)
        weighted = foreground_mask * depth_map

        total_weight = weighted.sum()
        if total_weight < 1e-6:
            return 0.5, 0.5  # Fallback to center

        yy, xx = np.mgrid[:h, :w]
        cx = (xx * weighted).sum() / total_weight / w
        cy = (yy * weighted).sum() / total_weight / h

        # Clamp to reasonable range (don't target extreme edges)
        cx = max(0.2, min(0.8, cx))
        cy = max(0.2, min(0.8, cy))

        return float(cx), float(cy)


class WeatherEffects:
    """Dynamic weather and atmosphere particle overlays.

    Creates section-aware atmospheric effects that enhance the emotional
    tone of each biography section:
    - Rain particles for THE_FALL (sadness, loss)
    - Dust/embers for COLD_OPEN and THE_CONFLICT (tension, drama)
    - Light particles/fireflies for LEGACY (hope, remembrance)
    - Snow/ash for devastating moments

    All effects are pure numpy - no additional dependencies required.
    """

    # Section-to-weather mapping
    SECTION_WEATHER = {
        'COLD_OPEN': 'dust',
        'EARLY_LIFE': None,
        'THE_SPARK': 'light_particles',
        'THE_RISE': None,
        'THE_CONFLICT': 'embers',
        'THE_CLIMAX': 'embers',
        'THE_FALL': 'rain',
        'LEGACY': 'light_particles',
        'CTA': None,
    }

    # Emotion overrides
    EMOTION_WEATHER = {
        'devastation': 'rain',
        'darkness': 'rain',
        'tension': 'dust',
        'nostalgia': 'light_particles',
        'hope': 'light_particles',
        'triumph': 'light_particles',
    }

    def create_weather_frame(
        self,
        width: int,
        height: int,
        t: float,
        duration: float,
        weather_type: str,
        intensity: float = 0.5,
    ) -> np.ndarray:
        """Create a single weather overlay frame for time t (on-the-fly, O(1) memory).

        Uses stateless particle position calculations so any frame can be computed
        independently without pre-generating all frames. This avoids OOM crashes
        for long sections (e.g., 120s at 30fps would have needed ~28 GB).

        Args:
            width: Frame width.
            height: Frame height.
            t: Current time in seconds.
            duration: Total duration in seconds.
            weather_type: One of 'rain', 'dust', 'embers', 'light_particles'.
            intensity: Effect intensity [0, 1].

        Returns:
            Single RGBA numpy array (H, W, 4), uint8.
        """
        progress = t / duration if duration > 0 else 0
        progress = max(0.0, min(1.0, progress))

        if weather_type == 'rain':
            return self._create_rain_frame(width, height, t, progress, intensity)
        elif weather_type == 'dust':
            return self._create_dust_frame(width, height, t, progress, intensity)
        elif weather_type == 'embers':
            return self._create_ember_frame(width, height, t, progress, intensity)
        elif weather_type == 'light_particles':
            return self._create_light_particle_frame(width, height, t, progress, intensity)
        else:
            return np.zeros((height, width, 4), dtype=np.uint8)

    def get_weather_for_section(
        self, section: str, emotional_tone: str = ''
    ) -> Optional[str]:
        """Determine weather type based on section and emotion.

        Args:
            section: Story section name.
            emotional_tone: Emotional tone string.

        Returns:
            Weather type string or None if no weather effect.
        """
        # Emotion overrides section
        for key, weather in self.EMOTION_WEATHER.items():
            if key in emotional_tone.lower():
                return weather

        return self.SECTION_WEATHER.get(section)

    # --- Stateless single-frame generators (O(1) memory) ---

    def _create_rain_frame(
        self, w: int, h: int, t: float, progress: float, intensity: float
    ) -> np.ndarray:
        """Create a single rain frame at time t (stateless)."""
        rng = np.random.default_rng(42)
        num_drops = int(200 * intensity)

        # Pre-generate particle initial state (deterministic from seed)
        drops_x0 = rng.uniform(0, w, num_drops)
        drops_y0 = rng.uniform(-h, 0, num_drops)
        drops_speed = rng.uniform(15, 35, num_drops)
        drops_length = rng.uniform(10, 30, num_drops).astype(int)
        drops_alpha = rng.uniform(0.15, 0.45, num_drops) * intensity

        frame = np.zeros((h, w, 4), dtype=np.uint8)
        # Compute positions at time t using modular arithmetic (stateless)
        cycle_height = h + h * 0.3  # Total travel distance before reset
        for i in range(num_drops):
            travel = drops_speed[i] * t * 30  # 30 = reference fps for motion speed
            y = int((drops_y0[i] + travel) % cycle_height - h * 0.3)
            x = int(drops_x0[i] + np.sin(t * 2 + i * 0.1) * 2) % w
            length = drops_length[i]
            alpha = int(drops_alpha[i] * 255)

            if 0 <= y < h:
                y_end = min(y + length, h)
                if y_end > y and 0 <= x < w:
                    frame[y:y_end, x, 0] = 180
                    frame[y:y_end, x, 1] = 200
                    frame[y:y_end, x, 2] = 230
                    frame[y:y_end, x, 3] = alpha

        return frame

    def _create_dust_frame(
        self, w: int, h: int, t: float, progress: float, intensity: float
    ) -> np.ndarray:
        """Create a single dust frame at time t (stateless)."""
        rng = np.random.default_rng(123)
        num_particles = int(80 * intensity)

        px = rng.uniform(0, w, num_particles)
        py = rng.uniform(0, h, num_particles)
        sizes = rng.integers(2, 5, num_particles)
        alphas = rng.uniform(0.1, 0.35, num_particles) * intensity
        phases = rng.uniform(0, 2 * np.pi, num_particles)

        frame = np.zeros((h, w, 4), dtype=np.uint8)
        for i in range(num_particles):
            x = int(px[i] + np.sin(progress * 2 * np.pi + phases[i]) * 10) % w
            y = int(py[i] + progress * (-20 + i % 10)) % h  # Slow drift
            s = sizes[i]
            alpha = int(alphas[i] * 255 * (0.5 + 0.5 * np.sin(progress * np.pi * 3 + phases[i])))
            alpha = max(0, min(255, alpha))

            y1, y2 = max(0, y - s), min(h, y + s)
            x1, x2 = max(0, x - s), min(w, x + s)
            if y2 > y1 and x2 > x1:
                frame[y1:y2, x1:x2, 0] = 220
                frame[y1:y2, x1:x2, 1] = 200
                frame[y1:y2, x1:x2, 2] = 170
                frame[y1:y2, x1:x2, 3] = np.maximum(frame[y1:y2, x1:x2, 3], alpha)

        return frame

    def _create_ember_frame(
        self, w: int, h: int, t: float, progress: float, intensity: float
    ) -> np.ndarray:
        """Create a single ember frame at time t (stateless)."""
        rng = np.random.default_rng(456)
        num_embers = int(50 * intensity)

        ex = rng.uniform(0, w, num_embers)
        ey0 = rng.uniform(h * 0.5, h * 1.2, num_embers)
        e_speed = rng.uniform(2, 8, num_embers)
        e_sizes = rng.integers(1, 4, num_embers)
        e_alpha = rng.uniform(0.2, 0.6, num_embers) * intensity
        e_phase = rng.uniform(0, 2 * np.pi, num_embers)
        e_red = rng.integers(200, 255, num_embers)
        e_green = rng.integers(80, 180, num_embers)
        e_blue = rng.integers(20, 60, num_embers)

        frame = np.zeros((h, w, 4), dtype=np.uint8)
        cycle_height = h * 1.2 + 20  # Total travel before reset
        for i in range(num_embers):
            travel = e_speed[i] * t * 30  # 30 = reference fps
            y = int((ey0[i] - travel) % cycle_height)
            if y > h:
                y = int(h * 1.2 - (y - h))  # Wrap from bottom
            x = int(ex[i] + np.sin(progress * 4 * np.pi + e_phase[i]) * 15) % w
            s = e_sizes[i]

            life = 1.0 - max(0, (h * 0.5 - y) / (h * 0.5)) if y < h * 0.5 else 1.0
            alpha = int(e_alpha[i] * 255 * life)
            alpha = max(0, min(255, alpha))

            if 0 <= y < h:
                y1, y2 = max(0, y - s), min(h, y + s)
                x1, x2 = max(0, x - s), min(w, x + s)
                if y2 > y1 and x2 > x1:
                    frame[y1:y2, x1:x2, 0] = e_red[i]
                    frame[y1:y2, x1:x2, 1] = e_green[i]
                    frame[y1:y2, x1:x2, 2] = e_blue[i]
                    frame[y1:y2, x1:x2, 3] = np.maximum(frame[y1:y2, x1:x2, 3], alpha)

        return frame

    def _create_light_particle_frame(
        self, w: int, h: int, t: float, progress: float, intensity: float
    ) -> np.ndarray:
        """Create a single light particle frame at time t (stateless)."""
        rng = np.random.default_rng(789)
        num_particles = int(40 * intensity)

        lx = rng.uniform(0, w, num_particles)
        ly = rng.uniform(0, h, num_particles)
        l_sizes = rng.integers(2, 6, num_particles)
        l_phase = rng.uniform(0, 2 * np.pi, num_particles)
        l_speed = rng.uniform(0.3, 1.5, num_particles)
        l_alpha_base = rng.uniform(0.15, 0.45, num_particles) * intensity

        frame = np.zeros((h, w, 4), dtype=np.uint8)
        for i in range(num_particles):
            x = int(lx[i] + np.sin(progress * 2 * np.pi * l_speed[i] + l_phase[i]) * 20) % w
            y = int(ly[i] + np.cos(progress * 2 * np.pi * l_speed[i] * 0.7 + l_phase[i]) * 15) % h
            s = l_sizes[i]

            pulse = 0.5 + 0.5 * np.sin(progress * 4 * np.pi + l_phase[i])
            alpha = int(l_alpha_base[i] * 255 * pulse)
            alpha = max(0, min(255, alpha))

            y1, y2 = max(0, y - s), min(h, y + s)
            x1, x2 = max(0, x - s), min(w, x + s)
            if y2 > y1 and x2 > x1:
                frame[y1:y2, x1:x2, 0] = 255
                frame[y1:y2, x1:x2, 1] = 240
                frame[y1:y2, x1:x2, 2] = 180
                frame[y1:y2, x1:x2, 3] = np.maximum(frame[y1:y2, x1:x2, 3], alpha)

        return frame


def get_ai_status() -> dict:
    """Return the current status of AI dependencies.

    Returns:
        Dictionary with availability status of each AI backend.
    """
    return {
        'torch_available': _HAS_TORCH,
        'opencv_available': _HAS_CV2,
        'depth_backend': 'MiDaS (AI)' if _HAS_TORCH else 'numpy (heuristic)',
        'parallax_quality': 'high' if _HAS_TORCH else 'basic',
    }
