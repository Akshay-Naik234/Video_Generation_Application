"""Stunning text overlay system for cinematic biography videos.

Creates eye-catching, professional text overlays with:
- Text shadows and outlines for readability on any background
- Gradient backgrounds with frosted glass feel
- Golden accent elements that pop visually
- Cross-platform font discovery (Linux, macOS, Windows)
- Multiple overlay types: year stamps, location stamps, lower thirds,
  quote cards, info cards, and slide-in text

Designed to make viewers immediately understand context (dates, locations,
names, key facts) while looking polished and cinematic.
"""

import os
import math
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Optional, List


class TextOverlayEngine:
    """Creates stunning, eye-catching text overlays for biography videos.

    All overlays return RGBA numpy arrays at the configured resolution.
    Text is rendered with shadows, outlines, and accent elements for
    maximum readability and visual impact over any video frame.

    Supports:
    - Lower thirds (name + title bar with gradient background and accent stripe)
    - Quote cards (centered stylized quotes with large quotation marks)
    - Info cards (date, location, key fact overlays with icon accents)
    - Year stamps (large cinematic year display with glow effect)
    - Location stamps (city/country with accent bar and pin dot)
    - Slide-in overlays (animated text that slides from the side)
    """

    # Font size defaults (relative to 1080p, scaled by resolution)
    FONT_SIZES = {
        'lower_third_name': 44,
        'lower_third_title': 28,
        'chapter_title': 58,
        'chapter_subtitle': 32,
        'quote_text': 38,
        'quote_attribution': 26,
        'quote_mark': 96,
        'info_card': 32,
        'year_stamp': 80,
        'year_stamp_label': 24,
        'location_stamp': 30,
        'location_stamp_sub': 22,
        'slide_in': 32,
    }

    # Cross-platform font paths: tried in order until one works.
    _FONT_PATHS_REGULAR = [
        # Linux
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSans.ttf',
        # macOS
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/SFNSText.ttf',
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/Library/Fonts/Arial.ttf',
        # Windows
        'C:/Windows/Fonts/arial.ttf',
        'C:/Windows/Fonts/segoeui.ttf',
        'C:/Windows/Fonts/calibri.ttf',
    ]
    _FONT_PATHS_BOLD = [
        # Linux
        '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',
        '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf',
        '/usr/share/fonts/truetype/freefont/FreeSansBold.ttf',
        # macOS
        '/System/Library/Fonts/Helvetica.ttc',
        '/System/Library/Fonts/SFNSText-Bold.ttf',
        '/System/Library/Fonts/Supplemental/Arial Bold.ttf',
        '/Library/Fonts/Arial Bold.ttf',
        # Windows
        'C:/Windows/Fonts/arialbd.ttf',
        'C:/Windows/Fonts/segoeuib.ttf',
        'C:/Windows/Fonts/calibrib.ttf',
    ]

    # Cache resolved font paths (class-level, searched once per process)
    _resolved_regular: Optional[str] = None
    _resolved_bold: Optional[str] = None

    # Default accent color (golden)
    DEFAULT_ACCENT_COLOR = (218, 165, 32)

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080),
                 accent_color: Tuple[int, int, int] = (218, 165, 32)):
        self.width, self.height = resolution
        self.scale = self.height / 1080.0
        self.accent_color = accent_color

    # ---- Font discovery ----

    @classmethod
    def _find_font_path(cls, bold: bool = False) -> Optional[str]:
        """Find the first available font file on this platform."""
        if bold and cls._resolved_bold is not None:
            return cls._resolved_bold
        if not bold and cls._resolved_regular is not None:
            return cls._resolved_regular

        candidates = cls._FONT_PATHS_BOLD if bold else cls._FONT_PATHS_REGULAR
        for path in candidates:
            if os.path.isfile(path):
                if bold:
                    cls._resolved_bold = path
                else:
                    cls._resolved_regular = path
                return path

        # Fallback: try regular paths for bold if no bold font found
        if bold:
            for path in cls._FONT_PATHS_REGULAR:
                if os.path.isfile(path):
                    cls._resolved_bold = path
                    return path

        # Dynamic fallback via matplotlib font_manager (finds any system font)
        try:
            from matplotlib import font_manager as fm
            prop = fm.FontProperties(weight='bold' if bold else 'normal')
            found = fm.findfont(prop, fallback_to_default=True)
            if found and os.path.isfile(found):
                if bold:
                    cls._resolved_bold = found
                else:
                    cls._resolved_regular = found
                return found
        except ImportError:
            pass
        return None

    def _get_font(self, size_key: str, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get a font at the appropriate size for the resolution.

        Searches common font locations across Linux, macOS, and Windows.
        Caches the resolved font path so subsequent calls are instant.
        """
        base_size = self.FONT_SIZES.get(size_key, 30)
        scaled_size = max(14, int(base_size * self.scale))
        font_path = self._find_font_path(bold=bold)
        if font_path:
            try:
                return ImageFont.truetype(font_path, scaled_size)
            except (OSError, IOError):
                pass
        # Last resort: PIL default with size (Pillow 10+ supports size arg)
        try:
            return ImageFont.load_default(size=scaled_size)
        except TypeError:
            return ImageFont.load_default()

    def _get_font_at_size(self, size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get a font at an explicit pixel size."""
        scaled_size = max(14, int(size * self.scale))
        font_path = self._find_font_path(bold=bold)
        if font_path:
            try:
                return ImageFont.truetype(font_path, scaled_size)
            except (OSError, IOError):
                pass
        try:
            return ImageFont.load_default(size=scaled_size)
        except TypeError:
            return ImageFont.load_default()

    # ---- Text wrapping / safe margin helpers ----

    # Safe margin percentage (title-safe area) — text stays within this zone.
    _SAFE_MARGIN_PCT = 0.05  # 5% on each side

    def _safe_rect(self) -> Tuple[int, int, int, int]:
        """Return (left, top, right, bottom) of the title-safe area."""
        mx = int(self.width * self._SAFE_MARGIN_PCT)
        my = int(self.height * self._SAFE_MARGIN_PCT)
        return mx, my, self.width - mx, self.height - my

    def _wrap_text(
        self,
        text: str,
        font: ImageFont.FreeTypeFont,
        max_width: int,
        draw: Optional[ImageDraw.ImageDraw] = None,
    ) -> List[str]:
        """Word-wrap *text* so no line exceeds *max_width* pixels."""
        if draw is None:
            tmp = PILImage.new('RGBA', (1, 1))
            draw = ImageDraw.Draw(tmp)
        words = text.split()
        lines: List[str] = []
        current: List[str] = []
        for word in words:
            test_line = ' '.join(current + [word])
            bbox = draw.textbbox((0, 0), test_line, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current.append(word)
            else:
                lines.append(' '.join(current))
                current = [word]
        if current:
            lines.append(' '.join(current))
        return lines or [text]

    def _clamp_position(
        self, x: int, y: int, obj_w: int, obj_h: int
    ) -> Tuple[int, int]:
        """Clamp (x, y) so the object stays within the title-safe area."""
        sl, st, sr, sb = self._safe_rect()
        x = max(sl, min(x, sr - obj_w))
        y = max(st, min(y, sb - obj_h))
        return x, y

    # ---- Drawing helpers ----

    def _draw_text_with_shadow(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: Tuple[int, int, int, int] = (255, 255, 255, 255),
        shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        shadow_offset: int = 10,
    ) -> None:
        """Draw text with a multi-layer drop shadow and outline for readability."""
        x, y = position
        offset = max(6, int(shadow_offset * self.scale))
        sw = max(2, int(3 * self.scale))
        sf = (0, 0, 0, 220)
        # Multi-layer shadow for stronger depth on dark backgrounds
        for dx, dy, alpha in [(offset + 2, offset + 2, 80), (offset + 1, offset + 1, 140), (offset, offset, 255)]:
            sc = (shadow_color[0], shadow_color[1], shadow_color[2], alpha)
            draw.text((x + dx, y + dy), text, font=font, fill=sc)
        # Main text with stroke
        try:
            draw.text((x, y), text, font=font, fill=fill,
                      stroke_width=sw, stroke_fill=sf)
        except TypeError:
            draw.text((x, y), text, font=font, fill=fill)

    def _draw_text_with_outline(
        self,
        draw: ImageDraw.ImageDraw,
        position: Tuple[int, int],
        text: str,
        font: ImageFont.FreeTypeFont,
        fill: Tuple[int, int, int, int] = (255, 255, 255, 255),
        outline_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        outline_width: int = 10,
        shadow: bool = True,
        shadow_color: Tuple[int, int, int, int] = (0, 0, 0, 255),
        shadow_offset: int = 10,
    ) -> None:
        """Draw text with outline stroke and optional drop shadow.

        Uses Pillow's native stroke_width for fast single-call outline
        rendering instead of an O(n²) multi-direction loop.
        """
        x, y = position
        ow = max(3, int(outline_width * self.scale))

        if shadow:
            so = max(6, int(shadow_offset * self.scale))
            for dx, dy, alpha_div in [(so + 2, so + 2, 3), (so + 1, so + 1, 2),
                                      (so - 1, so - 1, 2), (so, so, 1)]:
                sc = (shadow_color[0], shadow_color[1], shadow_color[2],
                      shadow_color[3] // max(alpha_div, 1))
                draw.text((x + dx, y + dy), text, font=font, fill=sc)

        # Native stroke_width is orders of magnitude faster than the O(n²) loop
        try:
            draw.text(
                (x, y), text, font=font, fill=fill,
                stroke_width=ow, stroke_fill=outline_color,
            )
        except TypeError:
            # Fallback for older Pillow without stroke_width
            for dx in range(-ow, ow + 1, max(1, ow // 2)):
                for dy in range(-ow, ow + 1, max(1, ow // 2)):
                    if dx == 0 and dy == 0:
                        continue
                    draw.text((x + dx, y + dy), text, font=font, fill=outline_color)
            draw.text((x, y), text, font=font, fill=fill)

    def _create_gradient_bar_fast(
        self,
        width: int,
        height: int,
        color: Tuple[int, int, int] = (0, 0, 0),
        alpha_start: int = 220,
        alpha_end: int = 0,
        direction: str = 'right',
    ) -> PILImage.Image:
        """Fast vectorized gradient bar creation using numpy."""
        arr = np.zeros((height, width, 4), dtype=np.uint8)
        arr[:, :, 0] = color[0]
        arr[:, :, 1] = color[1]
        arr[:, :, 2] = color[2]

        if direction == 'right':
            alpha_row = np.linspace(alpha_start, alpha_end, width, dtype=np.uint8)
            arr[:, :, 3] = alpha_row[np.newaxis, :]
        elif direction == 'left':
            alpha_row = np.linspace(alpha_end, alpha_start, width, dtype=np.uint8)
            arr[:, :, 3] = alpha_row[np.newaxis, :]
        elif direction == 'down':
            alpha_col = np.linspace(alpha_start, alpha_end, height, dtype=np.uint8)
            arr[:, :, 3] = alpha_col[:, np.newaxis]
        elif direction == 'up':
            alpha_col = np.linspace(alpha_end, alpha_start, height, dtype=np.uint8)
            arr[:, :, 3] = alpha_col[:, np.newaxis]

        return PILImage.fromarray(arr, 'RGBA')

    # ---- Overlay types ----

    def create_lower_third(
        self,
        name: str,
        title: str = '',
        accent_color: Tuple[int, int, int] = (218, 165, 32),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a stunning lower-third overlay with gradient background.

        Features:
        - Gradient background that fades right-to-transparent (cinematic look)
        - Bright accent stripe on the left edge
        - Name in bold with text outline for readability
        - Title in lighter weight below
        - Drop shadow on all text

        Returns an RGBA numpy array sized to full resolution.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        name_font = self._get_font('lower_third_name', bold=True)
        title_font = self._get_font('lower_third_title')

        sl, st, sr, sb = self._safe_rect()
        safe_w = sr - sl

        # Wrap name/title if they exceed safe width
        pad_x = int(36 * self.scale)
        accent_w = int(6 * self.scale)
        max_text_w = safe_w - pad_x * 3 - accent_w - int(120 * self.scale)
        name_lines = self._wrap_text(name, name_font, max_text_w, draw)
        name = '\n'.join(name_lines)

        # Measure text
        name_bbox = draw.textbbox((0, 0), name, font=name_font)
        name_w = name_bbox[2] - name_bbox[0]
        name_h = name_bbox[3] - name_bbox[1]

        title_h = 0
        title_w = 0
        if title:
            title_lines = self._wrap_text(title, title_font, max_text_w, draw)
            title = '\n'.join(title_lines)
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_w = title_bbox[2] - title_bbox[0]
            title_h = title_bbox[3] - title_bbox[1]

        # Bar dimensions
        pad_y = int(16 * self.scale)
        content_w = max(name_w, title_w)
        bar_w = content_w + pad_x * 3 + accent_w
        # Extend gradient beyond text for fade-out effect
        gradient_w = min(bar_w + int(120 * self.scale), self.width - int(40 * self.scale))
        bar_h = name_h + title_h + pad_y * (3 if title else 2) + (int(10 * self.scale) if title else 0)

        # Position: bottom-left with margin, clamped to safe area
        margin_x = int(80 * self.scale)
        margin_y = int(100 * self.scale)
        bar_x = margin_x
        bar_y = self.height - margin_y - bar_h
        bar_x, bar_y = self._clamp_position(bar_x, bar_y, gradient_w, bar_h)

        # Gradient background (fades right)
        gradient = self._create_gradient_bar_fast(
            gradient_w, bar_h,
            color=(12, 12, 18), alpha_start=255, alpha_end=60, direction='right'
        )
        overlay.paste(gradient, (bar_x, bar_y), gradient)

        # Accent stripe (bright, full opacity)
        accent = PILImage.new('RGBA', (accent_w, bar_h), (*accent_color, 255))
        overlay.paste(accent, (bar_x, bar_y), accent)

        # Thin accent line at bottom of bar
        line_h = max(1, int(2 * self.scale))
        accent_line = PILImage.new('RGBA', (bar_w, line_h), (*accent_color, 180))
        overlay.paste(accent_line, (bar_x, bar_y + bar_h - line_h), accent_line)

        # Name text with outline + shadow
        text_x = bar_x + accent_w + int(18 * self.scale)
        text_y = bar_y + pad_y
        self._draw_text_with_outline(
            draw, (text_x, text_y), name, name_font,
            fill=(*text_color, 255),
            outline_color=(0, 0, 0, 255), outline_width=6,
            shadow=True, shadow_color=(0, 0, 0, 250), shadow_offset=8
        )

        # Title text with shadow (lighter, no outline)
        if title:
            title_y = text_y + name_h + int(10 * self.scale)
            self._draw_text_with_shadow(
                draw, (text_x, title_y), title, title_font,
                fill=(*accent_color, 255),
                shadow_color=(0, 0, 0, 250), shadow_offset=6
            )

        return np.array(overlay)

    def create_year_stamp(
        self,
        year: str,
        label: str = '',
        position: str = 'top_right',
        accent_color: Tuple[int, int, int] = (218, 165, 32),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a large, eye-catching year stamp with glow effect.

        Features:
        - Large bold year number in accent gold color
        - Subtle glow/bloom behind the year text
        - Optional label below (e.g., city name)
        - Frosted dark background card
        - Accent line on top

        Returns an RGBA numpy array sized to full resolution.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        year_font = self._get_font('year_stamp', bold=True)
        label_font = self._get_font('year_stamp_label')

        # Measure year text
        year_bbox = draw.textbbox((0, 0), year, font=year_font)
        year_w = year_bbox[2] - year_bbox[0]
        year_h = year_bbox[3] - year_bbox[1]

        label_h = 0
        label_w = 0
        if label:
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_w = label_bbox[2] - label_bbox[0]
            label_h = label_bbox[3] - label_bbox[1]

        # Card dimensions
        pad_x = int(32 * self.scale)
        pad_y = int(20 * self.scale)
        content_w = max(year_w, label_w)
        card_w = content_w + pad_x * 2
        card_h = year_h + pad_y * 2 + (label_h + int(8 * self.scale) if label else 0)

        margin = int(50 * self.scale)
        if position == 'top_right':
            card_x = self.width - margin - card_w
            card_y = margin
        elif position == 'top_left':
            card_x = margin
            card_y = margin
        else:
            card_x = self.width - margin - card_w
            card_y = margin

        # Glow effect: draw year text on a separate layer, blur it, then composite
        glow_layer = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        glow_draw = ImageDraw.Draw(glow_layer)
        glow_year_x = card_x + (card_w - year_w) // 2
        glow_year_y = card_y + pad_y
        # Draw glow text (accent color, larger for bloom)
        glow_draw.text(
            (glow_year_x, glow_year_y), year, font=year_font,
            fill=(*accent_color, 100)
        )
        glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=int(12 * self.scale)))

        # Composite glow onto overlay
        overlay = PILImage.alpha_composite(overlay, glow_layer)
        draw = ImageDraw.Draw(overlay)

        # Dark frosted background card
        card_bg = PILImage.new('RGBA', (card_w, card_h), (8, 8, 14, 200))
        overlay.paste(card_bg, (card_x, card_y), card_bg)

        # Accent line on top of card
        line_h = max(1, int(3 * self.scale))
        accent_line = PILImage.new('RGBA', (card_w, line_h), (*accent_color, 240))
        overlay.paste(accent_line, (card_x, card_y), accent_line)

        # Year text with outline (centered in card)
        year_x = card_x + (card_w - year_w) // 2
        year_y = card_y + pad_y
        self._draw_text_with_outline(
            draw, (year_x, year_y), year, year_font,
            fill=(*accent_color, 255),
            outline_color=(0, 0, 0, 255), outline_width=6,
            shadow=True, shadow_color=(0, 0, 0, 250), shadow_offset=8
        )

        # Label below year (centered, white, stronger)
        if label:
            label_x = card_x + (card_w - label_w) // 2
            label_y = year_y + year_h + int(8 * self.scale)
            self._draw_text_with_shadow(
                draw, (label_x, label_y), label, label_font,
                fill=(*text_color, 255),
                shadow_color=(0, 0, 0, 250), shadow_offset=6
            )

        return np.array(overlay)

    def create_location_stamp(
        self,
        location: str,
        sub_text: str = '',
        accent_color: Tuple[int, int, int] = (218, 165, 32),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a location stamp with accent bar and pin dot.

        Features:
        - Gradient background fading right
        - Accent bar on left edge with pin dot
        - Location name in bold with outline
        - Optional sub-text (e.g., country)

        Returns an RGBA numpy array sized to full resolution.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        loc_font = self._get_font('location_stamp', bold=True)
        sub_font = self._get_font('location_stamp_sub')

        loc_bbox = draw.textbbox((0, 0), location, font=loc_font)
        loc_w = loc_bbox[2] - loc_bbox[0]
        loc_h = loc_bbox[3] - loc_bbox[1]

        sub_h = 0
        sub_w = 0
        if sub_text:
            sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
            sub_w = sub_bbox[2] - sub_bbox[0]
            sub_h = sub_bbox[3] - sub_bbox[1]

        pad_x = int(28 * self.scale)
        pad_y = int(12 * self.scale)
        accent_w = int(5 * self.scale)
        content_w = max(loc_w, sub_w)
        card_w = content_w + pad_x * 2 + accent_w + int(14 * self.scale)
        card_h = loc_h + pad_y * 2 + (sub_h + int(6 * self.scale) if sub_text else 0)
        gradient_w = min(card_w + int(80 * self.scale), self.width // 2)

        margin_x = int(80 * self.scale)
        margin_y = int(80 * self.scale)
        card_x = margin_x
        card_y = self.height - margin_y - card_h

        # Gradient background (more opaque for readability)
        gradient = self._create_gradient_bar_fast(
            gradient_w, card_h,
            color=(10, 10, 16), alpha_start=255, alpha_end=60, direction='right'
        )
        overlay.paste(gradient, (card_x, card_y), gradient)

        # Accent bar
        accent_bar = PILImage.new('RGBA', (accent_w, card_h), (*accent_color, 250))
        overlay.paste(accent_bar, (card_x, card_y), accent_bar)

        # Pin dot above accent bar
        dot_r = int(6 * self.scale)
        dot_cx = card_x + accent_w // 2
        dot_cy = card_y - dot_r - int(4 * self.scale)
        draw.ellipse(
            [dot_cx - dot_r, dot_cy - dot_r, dot_cx + dot_r, dot_cy + dot_r],
            fill=(*accent_color, 255)
        )
        # Inner highlight dot
        inner_r = max(1, dot_r // 2)
        draw.ellipse(
            [dot_cx - inner_r, dot_cy - inner_r, dot_cx + inner_r, dot_cy + inner_r],
            fill=(255, 255, 255, 180)
        )

        # Location text with outline
        text_x = card_x + accent_w + int(14 * self.scale)
        text_y = card_y + pad_y
        self._draw_text_with_outline(
            draw, (text_x, text_y), location, loc_font,
            fill=(*text_color, 255),
            outline_color=(0, 0, 0, 255), outline_width=6,
            shadow=True, shadow_color=(0, 0, 0, 250), shadow_offset=8
        )

        # Sub-text
        if sub_text:
            sub_y = text_y + loc_h + int(6 * self.scale)
            self._draw_text_with_shadow(
                draw, (text_x, sub_y), sub_text, sub_font,
                fill=(*text_color, 255),
                shadow_color=(0, 0, 0, 250), shadow_offset=6
            )

        return np.array(overlay)

    def create_info_card(
        self,
        text: str,
        position: str = 'bottom_right',
        bg_color: Tuple[int, int, int] = (12, 12, 18),
        text_color: Tuple[int, int, int] = (240, 240, 240),
        accent_color: Tuple[int, int, int] = (218, 165, 32),
        icon_text: str = '',
    ) -> np.ndarray:
        """Create a stylish info card for dates, locations, or key facts.

        Features:
        - Gradient background with accent border
        - Text with drop shadow
        - Optional icon/emoji prefix

        Returns an RGBA numpy array sized to full resolution.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = self._get_font('info_card', bold=True)
        display_text = f"{icon_text}  {text}".strip() if icon_text else text

        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pad_x = int(28 * self.scale)
        pad_y = int(14 * self.scale)
        card_w = text_w + pad_x * 2
        card_h = text_h + pad_y * 2

        margin = int(60 * self.scale)
        if position == 'bottom_right':
            card_x = self.width - margin - card_w
            card_y = self.height - margin - card_h
        elif position == 'bottom_left':
            card_x = margin
            card_y = self.height - margin - card_h
        elif position == 'top_right':
            card_x = self.width - margin - card_w
            card_y = margin
        else:
            card_x = margin
            card_y = margin

        # Card background
        card_bg = PILImage.new('RGBA', (card_w, card_h), (*bg_color, 200))
        overlay.paste(card_bg, (card_x, card_y), card_bg)

        # Accent border on left
        border_w = max(1, int(3 * self.scale))
        border = PILImage.new('RGBA', (border_w, card_h), (*accent_color, 230))
        overlay.paste(border, (card_x, card_y), border)

        # Accent border on bottom
        bottom_border = PILImage.new('RGBA', (card_w, max(1, int(2 * self.scale))), (*accent_color, 140))
        overlay.paste(bottom_border, (card_x, card_y + card_h - max(1, int(2 * self.scale))), bottom_border)

        # Text with outline for max readability
        self._draw_text_with_outline(
            draw, (card_x + pad_x, card_y + pad_y),
            display_text, font,
            fill=(*text_color, 255),
            outline_color=(0, 0, 0, 255), outline_width=6,
            shadow=True, shadow_color=(0, 0, 0, 250), shadow_offset=8
        )

        return np.array(overlay)

    def create_quote_card(
        self,
        quote: str,
        attribution: str = '',
        accent_color: Tuple[int, int, int] = (218, 165, 32),
        max_width_ratio: float = 0.65,
    ) -> np.ndarray:
        """Create a stunning centered quote overlay.

        Features:
        - Large decorative quotation marks in accent color
        - Word-wrapped quote text centered on screen
        - Gradient backdrop band across the middle
        - Attribution line in accent color
        - All text with outlines for readability

        Returns an RGBA numpy array sized to full resolution.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        quote_font = self._get_font('quote_text', bold=True)
        attr_font = self._get_font('quote_attribution')
        mark_font = self._get_font_at_size(96, bold=True)

        max_width = int(self.width * max_width_ratio)

        # Word-wrap the quote
        words = quote.split()
        lines: List[str] = []
        current_line = ''
        for word in words:
            test_line = f'{current_line} {word}'.strip()
            test_bbox = draw.textbbox((0, 0), test_line, font=quote_font)
            if test_bbox[2] - test_bbox[0] <= max_width:
                current_line = test_line
            else:
                if current_line:
                    lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)

        # Calculate total height
        line_spacing = int(12 * self.scale)
        line_heights: List[int] = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=quote_font)
            line_heights.append(bbox[3] - bbox[1])

        total_h = sum(line_heights) + line_spacing * max(0, len(lines) - 1)
        mark_h = int(60 * self.scale)  # Space for quotation marks
        attr_h = 0
        if attribution:
            attr_bbox = draw.textbbox((0, 0), f"-- {attribution}", font=attr_font)
            attr_h = attr_bbox[3] - attr_bbox[1]
            total_h += int(24 * self.scale) + attr_h

        # Gradient backdrop band
        bg_pad = int(50 * self.scale)
        bg_h = total_h + bg_pad * 2 + mark_h
        bg_y = (self.height - bg_h) // 2
        bg = self._create_gradient_bar_fast(
            self.width, bg_h,
            color=(0, 0, 0), alpha_start=10, alpha_end=10, direction='down'
        )
        # Override with uniform opaque black to prevent ghosting from layers below
        bg_arr = np.array(bg)
        bg_arr[:, :, 3] = 255
        bg = PILImage.fromarray(bg_arr, 'RGBA')
        overlay.paste(bg, (0, bg_y), bg)

        # Opening quotation mark (large, accent color, top-left of quote area)
        open_mark = "\u201C"
        mark_bbox = draw.textbbox((0, 0), open_mark, font=mark_font)
        mark_w_actual = mark_bbox[2] - mark_bbox[0]
        quote_area_x = (self.width - max_width) // 2
        mark_x = quote_area_x - int(10 * self.scale)
        mark_y = bg_y + bg_pad - int(20 * self.scale)
        self._draw_text_with_shadow(
            draw, (mark_x, mark_y), open_mark, mark_font,
            fill=(*accent_color, 200),
            shadow_color=(0, 0, 0, 120), shadow_offset=4
        )

        # Draw quote lines centered with outline
        y = bg_y + bg_pad + mark_h
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=quote_font)
            line_w = bbox[2] - bbox[0]
            x = (self.width - line_w) // 2
            self._draw_text_with_outline(
                draw, (x, y), line, quote_font,
                fill=(255, 255, 255, 255),
                outline_color=(0, 0, 0, 255), outline_width=6,
                shadow=True, shadow_color=(0, 0, 0, 250), shadow_offset=8
            )
            y += line_heights[i] + line_spacing

        # Accent divider line before attribution
        if attribution:
            line_w = int(60 * self.scale)
            line_h = max(1, int(2 * self.scale))
            line_x = (self.width - line_w) // 2
            line_y = y + int(8 * self.scale)
            accent_line = PILImage.new('RGBA', (line_w, line_h), (*accent_color, 200))
            overlay.paste(accent_line, (line_x, line_y), accent_line)

            # Attribution
            attr_text = f"-- {attribution}"
            attr_bbox = draw.textbbox((0, 0), attr_text, font=attr_font)
            attr_w = attr_bbox[2] - attr_bbox[0]
            attr_x = (self.width - attr_w) // 2
            attr_y = line_y + line_h + int(10 * self.scale)
            self._draw_text_with_shadow(
                draw, (attr_x, attr_y), attr_text, attr_font,
                fill=(*accent_color, 240),
                shadow_color=(0, 0, 0, 100), shadow_offset=2
            )

        return np.array(overlay)

    def create_slide_in_overlay(
        self,
        text: str,
        position: str = 'bottom_left',
        slide_from: str = 'left',
        progress: float = 1.0,
        accent_color: Tuple[int, int, int] = (218, 165, 32),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a single frame of an animated slide-in text overlay.

        Features:
        - Gradient background with accent stripe
        - Text with outline for readability
        - Smooth slide animation based on progress (0.0 = off-screen, 1.0 = at rest)

        Returns an RGBA numpy array sized to full resolution.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = self._get_font('slide_in', bold=True)

        # Wrap text within half the screen width
        pad_x = int(28 * self.scale)
        accent_w = int(5 * self.scale)
        max_card_text_w = self.width // 2 - pad_x * 2 - accent_w - int(94 * self.scale)
        wrapped_lines = self._wrap_text(text, font, max_card_text_w, draw)
        text = '\n'.join(wrapped_lines)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_w = text_bbox[2] - text_bbox[0]
        text_h = text_bbox[3] - text_bbox[1]

        pad_y = int(14 * self.scale)
        card_w = text_w + pad_x * 2 + accent_w + int(14 * self.scale)
        card_h = text_h + pad_y * 2
        gradient_w = min(card_w + int(80 * self.scale), self.width // 2)

        margin = int(80 * self.scale)

        # Final resting position
        if position == 'bottom_right':
            rest_x = self.width - margin - card_w
            rest_y = self.height - margin - card_h
        else:  # bottom_left
            rest_x = margin
            rest_y = self.height - margin - card_h

        # Apply slide offset
        slide_distance = card_w + margin + int(40 * self.scale)
        remaining = 1.0 - min(1.0, max(0.0, progress))

        if slide_from == 'left':
            card_x = int(rest_x - slide_distance * remaining)
        elif slide_from == 'right':
            card_x = int(rest_x + slide_distance * remaining)
        elif slide_from == 'bottom':
            card_x = rest_x
            rest_y = int(rest_y + (card_h + margin) * remaining)
        else:
            card_x = int(rest_x - slide_distance * remaining)

        card_y = rest_y

        # Skip if fully off-screen
        if card_x + card_w < 0 or card_x > self.width:
            return np.array(overlay)

        # Gradient background
        gradient = self._create_gradient_bar_fast(
            gradient_w, card_h,
            color=(10, 10, 16), alpha_start=210, alpha_end=0, direction='right'
        )
        paste_x = max(0, card_x)
        overlay.paste(gradient, (paste_x, max(0, card_y)), gradient)

        # Accent bar
        accent_bar = PILImage.new('RGBA', (accent_w, card_h), (*accent_color, 250))
        overlay.paste(accent_bar, (max(0, card_x), max(0, card_y)), accent_bar)

        # Text with outline
        text_x = card_x + accent_w + int(14 * self.scale) + pad_x
        text_y = card_y + pad_y
        if 0 < text_x < self.width:
            self._draw_text_with_outline(
                draw, (text_x, text_y), text, font,
                fill=(*text_color, 250),
                outline_color=(0, 0, 0, 200), outline_width=2,
                shadow=True, shadow_color=(0, 0, 0, 120), shadow_offset=3
            )

        return np.array(overlay)

    def create_date_location_stamp(
        self,
        date_text: str,
        location_text: str = '',
        accent_color: Tuple[int, int, int] = (218, 165, 32),
    ) -> np.ndarray:
        """Create a combined date + location stamp in the bottom-right corner.

        Features:
        - Date in bold accent color, location in white
        - Vertical divider between date and location
        - Frosted dark background
        - Drop shadows on all text

        Returns an RGBA numpy array sized to full resolution.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        date_font = self._get_font('location_stamp', bold=True)
        loc_font = self._get_font('location_stamp_sub')

        date_bbox = draw.textbbox((0, 0), date_text, font=date_font)
        date_w = date_bbox[2] - date_bbox[0]
        date_h = date_bbox[3] - date_bbox[1]

        loc_w = 0
        loc_h = 0
        if location_text:
            loc_bbox = draw.textbbox((0, 0), location_text, font=loc_font)
            loc_w = loc_bbox[2] - loc_bbox[0]
            loc_h = loc_bbox[3] - loc_bbox[1]

        pad_x = int(24 * self.scale)
        pad_y = int(12 * self.scale)
        divider_w = int(2 * self.scale) if location_text else 0
        gap = int(18 * self.scale) if location_text else 0
        card_w = date_w + loc_w + pad_x * 2 + divider_w + gap * 2
        card_h = date_h + pad_y * 2

        margin = int(60 * self.scale)
        card_x = self.width - margin - card_w
        card_y = self.height - margin - card_h

        # Background
        card_bg = PILImage.new('RGBA', (card_w, card_h), (8, 8, 14, 200))
        overlay.paste(card_bg, (card_x, card_y), card_bg)

        # Accent border top
        border_h = max(1, int(2 * self.scale))
        accent_border = PILImage.new('RGBA', (card_w, border_h), (*accent_color, 200))
        overlay.paste(accent_border, (card_x, card_y), accent_border)

        # Date text (accent color, bold)
        date_x = card_x + pad_x
        date_y = card_y + pad_y
        self._draw_text_with_shadow(
            draw, (date_x, date_y), date_text, date_font,
            fill=(*accent_color, 255),
            shadow_color=(0, 0, 0, 140), shadow_offset=2
        )

        if location_text:
            # Divider
            div_x = date_x + date_w + gap
            div_y1 = card_y + int(8 * self.scale)
            div_y2 = card_y + card_h - int(8 * self.scale)
            draw.line(
                [(div_x, div_y1), (div_x, div_y2)],
                fill=(*accent_color, 130), width=divider_w
            )

            # Location text
            loc_x = div_x + divider_w + gap
            # Vertically center location relative to date
            loc_y = card_y + pad_y + (date_h - loc_h) // 2
            self._draw_text_with_shadow(
                draw, (loc_x, loc_y), location_text, loc_font,
                fill=(240, 240, 240, 220),
                shadow_color=(0, 0, 0, 100), shadow_offset=2
            )

        return np.array(overlay)

    # ---- Text classification helpers ----

    @staticmethod
    def classify_overlay_text(text: str) -> str:
        """Classify overlay text into a display type based on its content.

        Returns one of: 'year_stamp', 'location', 'date_location',
        'quote', 'lower_third', 'info_card'.

        Classification rules:
        - Pure year (e.g., "1884"): year_stamp
        - Year with location (e.g., "1943 | NYC"): date_location
        - Quoted text or text with em-dash attribution: quote
        - Text with " -- " name pattern: lower_third
        - Everything else: info_card (slide-in)
        """
        import re as _re
        stripped = text.strip()

        # Pure year: 3-4 digit number
        if _re.match(r'^\d{3,4}$', stripped):
            return 'year_stamp'

        # Year | Location or Year - Location
        if _re.match(r'^\d{3,4}\s*[\||\-\u2013\u2014]\s*.+', stripped):
            return 'date_location'

        # Quoted text (curly or straight quotes)
        if _re.match(r'^[\u201C\u201D\u2018\u2019"\']', stripped):
            return 'quote'

        # Attribution pattern: "quote text" -- Person Name
        if '\u2014' in stripped or '\u2013' in stripped or ' -- ' in stripped:
            # Check if it looks like "quote -- attribution"
            parts = _re.split(r'\s*[\u2014\u2013]\s*|\s+--\s+', stripped, maxsplit=1)
            if len(parts) == 2:
                quote_part = parts[0].strip()
                attr_part = parts[1].strip()
                # If the first part is quoted or the attribution looks like a name
                if (quote_part.startswith(('"', "'", '\u201C', '\u2018')) or
                        _re.match(r'^[A-Z][a-z]', attr_part)):
                    return 'quote'
                # Otherwise it's a name -- title pattern (lower third)
                return 'lower_third'

        # Default: info card (works for dates, facts, short context text)
        return 'info_card'

    def render_overlay_text(self, text: str,
                             accent_color: Optional[Tuple[int, int, int]] = None
                             ) -> np.ndarray:
        """Auto-classify and render overlay text into the appropriate visual style.

        This is the main entry point for the orchestrator. Given any overlay_text
        string from the duration config JSON, it determines the best visual
        representation and renders it.

        Args:
            text: The overlay text from image_display_duration.json.
            accent_color: Optional override for the accent color. If None,
                          uses the instance's accent_color (set at init).
        """
        import re as _re
        overlay_type = self.classify_overlay_text(text)
        stripped = text.strip()
        ac = accent_color or self.accent_color

        if overlay_type == 'year_stamp':
            return self.create_year_stamp(stripped, accent_color=ac)

        elif overlay_type == 'date_location':
            parts = _re.split(r'\s*[\||\-\u2013\u2014]\s*', stripped, maxsplit=1)
            date_part = parts[0].strip()
            loc_part = parts[1].strip() if len(parts) > 1 else ''
            if _re.match(r'^\d{3,4}$', date_part):
                return self.create_year_stamp(date_part, label=loc_part, accent_color=ac)
            return self.create_date_location_stamp(date_part, loc_part, accent_color=ac)

        elif overlay_type == 'quote':
            parts = _re.split(r'\s*[\u2014\u2013]\s*|\s+--\s+', stripped, maxsplit=1)
            quote_text = parts[0].strip()
            attribution = parts[1].strip() if len(parts) > 1 else ''
            # Strip quote marks
            quote_text = _re.sub(r'^[\u201C\u201D\u2018\u2019"\']+|[\u201C\u201D\u2018\u2019"\']+$', '', quote_text).strip()
            return self.create_quote_card(quote_text, attribution, accent_color=ac)

        elif overlay_type == 'lower_third':
            parts = _re.split(r'\s*[\u2014\u2013]\s*|\s+--\s+', stripped, maxsplit=1)
            name = parts[0].strip()
            title = parts[1].strip() if len(parts) > 1 else ''
            # Strip any quotes from name
            name = _re.sub(r'^[\u201C\u201D\u2018\u2019"\']+|[\u201C\u201D\u2018\u2019"\']+$', '', name).strip()
            return self.create_lower_third(name, title, accent_color=ac)

        else:
            # info_card / default
            return self.create_slide_in_overlay(stripped, progress=1.0, accent_color=ac)
