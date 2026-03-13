"""Text overlay system for cinematic lower thirds, quotes, and info cards.

Top biography channels (Biographics, The People Profiles, Biography) use text
overlays extensively: names, dates, locations, key quotes, and chapter titles.
This module adds that capability to the video generation pipeline.
"""

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from typing import Tuple, Optional


class TextOverlayEngine:
    """Creates professional text overlays for biography videos.
    
    Supports:
    - Lower thirds (name + title bar at bottom)
    - Chapter titles (centered section headers with fade)
    - Quote cards (stylized quote text with attribution)
    - Info cards (date, location, key fact overlays)
    """

    # Font size defaults (relative to 1080p, scaled by resolution)
    FONT_SIZES = {
        'lower_third_name': 42,
        'lower_third_title': 28,
        'chapter_title': 56,
        'chapter_subtitle': 32,
        'quote_text': 36,
        'quote_attribution': 24,
        'info_card': 30,
    }

    # Section-to-chapter-title mapping for automatic chapter cards
    SECTION_TITLES = {
        'COLD_OPEN': '',
        'EARLY_LIFE': 'The Early Years',
        'THE_SPARK': 'The Turning Point',
        'THE_RISE': 'The Rise',
        'THE_CONFLICT': 'The Struggle',
        'THE_CLIMAX': 'The Climax',
        'THE_FALL': 'The Fall',
        'LEGACY': 'The Legacy',
        'CTA': '',
    }

    def __init__(self, resolution: Tuple[int, int] = (1920, 1080)):
        self.width, self.height = resolution
        self.scale = self.height / 1080.0

    def _get_font(self, size_key: str, bold: bool = False) -> ImageFont.FreeTypeFont:
        """Get a font at the appropriate size for the resolution."""
        base_size = self.FONT_SIZES.get(size_key, 30)
        scaled_size = int(base_size * self.scale)
        try:
            if bold:
                return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", scaled_size)
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", scaled_size)
        except (OSError, IOError):
            return ImageFont.load_default()

    def create_lower_third(
        self,
        name: str,
        title: str = '',
        bar_color: Tuple[int, int, int] = (20, 20, 20),
        accent_color: Tuple[int, int, int] = (200, 170, 80),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a professional lower-third overlay with name and optional title.
        
        Returns an RGBA numpy array (transparent background) sized to full resolution.
        The lower third appears in the bottom-left area of the frame.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        name_font = self._get_font('lower_third_name', bold=True)
        title_font = self._get_font('lower_third_title')

        # Measure text
        name_bbox = draw.textbbox((0, 0), name, font=name_font)
        name_width = name_bbox[2] - name_bbox[0]
        name_height = name_bbox[3] - name_bbox[1]

        title_height = 0
        title_width = 0
        if title:
            title_bbox = draw.textbbox((0, 0), title, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
            title_height = title_bbox[3] - title_bbox[1]

        # Calculate bar dimensions
        padding_x = int(30 * self.scale)
        padding_y = int(12 * self.scale)
        accent_width = int(5 * self.scale)
        bar_width = max(name_width, title_width) + padding_x * 2 + accent_width + int(20 * self.scale)
        bar_height = name_height + title_height + padding_y * (3 if title else 2) + (int(8 * self.scale) if title else 0)

        # Position: bottom-left with margin
        margin_x = int(80 * self.scale)
        margin_y = int(100 * self.scale)
        bar_x = margin_x
        bar_y = self.height - margin_y - bar_height

        # Draw semi-transparent background bar
        bar_bg = PILImage.new('RGBA', (bar_width, bar_height), (*bar_color, 200))
        overlay.paste(bar_bg, (bar_x, bar_y), bar_bg)

        # Draw accent stripe on left edge
        accent_rect = PILImage.new('RGBA', (accent_width, bar_height), (*accent_color, 240))
        overlay.paste(accent_rect, (bar_x, bar_y), accent_rect)

        # Draw name text
        text_x = bar_x + accent_width + padding_x
        text_y = bar_y + padding_y
        draw.text((text_x, text_y), name, font=name_font, fill=(*text_color, 255))

        # Draw title text (dimmer)
        if title:
            title_y = text_y + name_height + int(8 * self.scale)
            dimmer_color = (text_color[0], text_color[1], text_color[2], 180)
            draw.text((text_x, title_y), title, font=title_font, fill=dimmer_color)

        return np.array(overlay)

    def create_chapter_card(
        self,
        section: str,
        custom_title: str = '',
        bg_color: Tuple[int, int, int] = (0, 0, 0),
        text_color: Tuple[int, int, int] = (255, 255, 255),
        accent_color: Tuple[int, int, int] = (200, 170, 80),
    ) -> Optional[np.ndarray]:
        """Create a centered chapter title card for section transitions.
        
        Returns an RGBA numpy array or None if the section has no title.
        """
        title = custom_title or self.SECTION_TITLES.get(section, '')
        if not title:
            return None

        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        title_font = self._get_font('chapter_title', bold=True)

        # Measure text
        title_bbox = draw.textbbox((0, 0), title, font=title_font)
        title_width = title_bbox[2] - title_bbox[0]
        title_height = title_bbox[3] - title_bbox[1]

        # Center position
        text_x = (self.width - title_width) // 2
        text_y = (self.height - title_height) // 2

        # Draw semi-transparent background band
        band_padding_y = int(40 * self.scale)
        band_height = title_height + band_padding_y * 2
        band_y = text_y - band_padding_y
        band = PILImage.new('RGBA', (self.width, band_height), (*bg_color, 180))
        overlay.paste(band, (0, band_y), band)

        # Draw accent line above title
        line_width = int(60 * self.scale)
        line_height = int(3 * self.scale)
        line_x = (self.width - line_width) // 2
        line_y = band_y + int(10 * self.scale)
        accent_line = PILImage.new('RGBA', (line_width, line_height), (*accent_color, 220))
        overlay.paste(accent_line, (line_x, line_y), accent_line)

        # Draw title
        draw.text((text_x, text_y), title, font=title_font, fill=(*text_color, 240))

        return np.array(overlay)

    def create_info_card(
        self,
        text: str,
        position: str = 'bottom_right',
        bg_color: Tuple[int, int, int] = (20, 20, 20),
        text_color: Tuple[int, int, int] = (230, 230, 230),
        icon_text: str = '',
    ) -> np.ndarray:
        """Create an info card overlay for dates, locations, or key facts.
        
        Args:
            text: The info text to display (e.g., "New York, 1952" or "$2.3 Million")
            position: Where to place the card ('bottom_right', 'bottom_left', 'top_right')
            bg_color: Background color of the card
            text_color: Text color
            icon_text: Optional icon/emoji prefix (e.g., "📍" for location)
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = self._get_font('info_card')
        display_text = f"{icon_text} {text}".strip() if icon_text else text

        # Measure text
        text_bbox = draw.textbbox((0, 0), display_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding_x = int(24 * self.scale)
        padding_y = int(12 * self.scale)
        card_width = text_width + padding_x * 2
        card_height = text_height + padding_y * 2

        margin = int(60 * self.scale)

        if position == 'bottom_right':
            card_x = self.width - margin - card_width
            card_y = self.height - margin - card_height
        elif position == 'bottom_left':
            card_x = margin
            card_y = self.height - margin - card_height
        elif position == 'top_right':
            card_x = self.width - margin - card_width
            card_y = margin
        else:
            card_x = margin
            card_y = margin

        # Draw card background
        card_bg = PILImage.new('RGBA', (card_width, card_height), (*bg_color, 190))
        overlay.paste(card_bg, (card_x, card_y), card_bg)

        # Draw text
        draw.text(
            (card_x + padding_x, card_y + padding_y),
            display_text,
            font=font,
            fill=(*text_color, 240)
        )

        return np.array(overlay)

    def create_quote_card(
        self,
        quote: str,
        attribution: str = '',
        max_width_ratio: float = 0.6,
    ) -> np.ndarray:
        """Create a centered quote overlay with optional attribution.
        
        Wraps long quotes to fit within max_width_ratio of the screen.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        quote_font = self._get_font('quote_text')
        attr_font = self._get_font('quote_attribution')

        max_width = int(self.width * max_width_ratio)

        # Word-wrap the quote
        words = quote.split()
        lines = []
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

        # Add quote marks
        if lines:
            lines[0] = f'"{lines[0]}'
            lines[-1] = f'{lines[-1]}"'

        # Calculate total height
        line_spacing = int(8 * self.scale)
        line_heights = []
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=quote_font)
            line_heights.append(bbox[3] - bbox[1])

        total_height = sum(line_heights) + line_spacing * (len(lines) - 1)
        if attribution:
            attr_bbox = draw.textbbox((0, 0), f"— {attribution}", font=attr_font)
            attr_height = attr_bbox[3] - attr_bbox[1]
            total_height += int(20 * self.scale) + attr_height

        # Draw semi-transparent background
        bg_padding = int(40 * self.scale)
        bg_height = total_height + bg_padding * 2
        bg_y = (self.height - bg_height) // 2
        bg = PILImage.new('RGBA', (self.width, bg_height), (0, 0, 0, 160))
        overlay.paste(bg, (0, bg_y), bg)

        # Draw quote lines centered
        y = bg_y + bg_padding
        for i, line in enumerate(lines):
            bbox = draw.textbbox((0, 0), line, font=quote_font)
            line_width = bbox[2] - bbox[0]
            x = (self.width - line_width) // 2
            draw.text((x, y), line, font=quote_font, fill=(255, 255, 255, 230))
            y += line_heights[i] + line_spacing

        # Draw attribution
        if attribution:
            attr_text = f"— {attribution}"
            attr_bbox = draw.textbbox((0, 0), attr_text, font=attr_font)
            attr_width = attr_bbox[2] - attr_bbox[0]
            attr_x = (self.width - attr_width) // 2
            attr_y = y + int(12 * self.scale)
            draw.text((attr_x, attr_y), attr_text, font=attr_font, fill=(200, 170, 80, 220))

        return np.array(overlay)
