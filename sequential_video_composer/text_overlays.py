"""Text overlay system for cinematic lower thirds, quotes, info cards, and date stamps.

Top biography channels (Biographics, The People Profiles, MagnatesMedia, Newsthink)
use text overlays extensively: names, dates, locations, key quotes, year stamps,
and chapter titles. This module provides all overlay types needed for professional
biography video production.
"""

import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from typing import Tuple, Optional, List


class TextOverlayEngine:
    """Creates professional text overlays for biography videos.
    
    Supports:
    - Lower thirds (name + title bar at bottom)
    - Chapter titles (centered section headers with fade)
    - Quote cards (stylized quote text with attribution)
    - Info cards (date, location, key fact overlays)
    - Year stamps (large cinematic year display, top-right corner)
    - Location stamps (city/country with map pin icon, bottom-left)
    - Date-location combos (combined date + location, bottom-right)
    - Progress indicators (timeline dot showing story position)
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
        'year_stamp': 72,
        'year_stamp_label': 22,
        'location_stamp': 28,
        'location_stamp_sub': 20,
        'progress_year': 18,
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

    def create_year_stamp(
        self,
        year: str,
        label: str = '',
        position: str = 'top_right',
        accent_color: Tuple[int, int, int] = (200, 170, 80),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a large cinematic year stamp overlay.
        
        Displays a prominent year number (e.g., "1884") in the corner with an
        optional label below (e.g., "New York City"). Used by Biographics and
        The People Profiles to orient viewers in the timeline.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        year_font = self._get_font('year_stamp', bold=True)
        label_font = self._get_font('year_stamp_label')

        # Measure year text
        year_bbox = draw.textbbox((0, 0), year, font=year_font)
        year_width = year_bbox[2] - year_bbox[0]
        year_height = year_bbox[3] - year_bbox[1]

        label_height = 0
        label_width = 0
        if label:
            label_bbox = draw.textbbox((0, 0), label, font=label_font)
            label_width = label_bbox[2] - label_bbox[0]
            label_height = label_bbox[3] - label_bbox[1]

        # Card dimensions
        padding_x = int(28 * self.scale)
        padding_y = int(16 * self.scale)
        card_content_width = max(year_width, label_width)
        card_width = card_content_width + padding_x * 2
        card_height = year_height + padding_y * 2 + (label_height + int(6 * self.scale) if label else 0)

        margin = int(50 * self.scale)

        if position == 'top_right':
            card_x = self.width - margin - card_width
            card_y = margin
        elif position == 'top_left':
            card_x = margin
            card_y = margin
        else:
            card_x = self.width - margin - card_width
            card_y = margin

        # Draw semi-transparent background
        card_bg = PILImage.new('RGBA', (card_width, card_height), (10, 10, 10, 180))
        overlay.paste(card_bg, (card_x, card_y), card_bg)

        # Draw accent line on top of card
        line_height = int(3 * self.scale)
        accent_line = PILImage.new('RGBA', (card_width, line_height), (*accent_color, 220))
        overlay.paste(accent_line, (card_x, card_y), accent_line)

        # Draw year (centered in card)
        year_x = card_x + (card_width - year_width) // 2
        year_y = card_y + padding_y
        draw.text((year_x, year_y), year, font=year_font, fill=(*accent_color, 255))

        # Draw label below year (centered, dimmer)
        if label:
            label_x = card_x + (card_width - label_width) // 2
            label_y = year_y + year_height + int(6 * self.scale)
            draw.text((label_x, label_y), label, font=label_font, fill=(*text_color, 190))

        return np.array(overlay)

    def create_location_stamp(
        self,
        location: str,
        sub_text: str = '',
        accent_color: Tuple[int, int, int] = (200, 170, 80),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a location stamp overlay with a pin-style accent.
        
        Shows location name (e.g., "New York City") with an optional subtitle
        (e.g., "United States") in the bottom-left corner. Uses a vertical
        accent bar to mimic map pin styling.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        loc_font = self._get_font('location_stamp', bold=True)
        sub_font = self._get_font('location_stamp_sub')

        # Measure text
        loc_bbox = draw.textbbox((0, 0), location, font=loc_font)
        loc_width = loc_bbox[2] - loc_bbox[0]
        loc_height = loc_bbox[3] - loc_bbox[1]

        sub_height = 0
        sub_width = 0
        if sub_text:
            sub_bbox = draw.textbbox((0, 0), sub_text, font=sub_font)
            sub_width = sub_bbox[2] - sub_bbox[0]
            sub_height = sub_bbox[3] - sub_bbox[1]

        # Card dimensions
        padding_x = int(24 * self.scale)
        padding_y = int(10 * self.scale)
        accent_width = int(4 * self.scale)
        content_width = max(loc_width, sub_width)
        card_width = content_width + padding_x * 2 + accent_width + int(12 * self.scale)
        card_height = loc_height + padding_y * 2 + (sub_height + int(4 * self.scale) if sub_text else 0)

        # Position: bottom-left
        margin_x = int(80 * self.scale)
        margin_y = int(80 * self.scale)
        card_x = margin_x
        card_y = self.height - margin_y - card_height

        # Draw background
        card_bg = PILImage.new('RGBA', (card_width, card_height), (15, 15, 15, 185))
        overlay.paste(card_bg, (card_x, card_y), card_bg)

        # Draw accent bar on left
        accent_bar = PILImage.new('RGBA', (accent_width, card_height), (*accent_color, 230))
        overlay.paste(accent_bar, (card_x, card_y), accent_bar)

        # Draw pin dot above accent bar
        dot_radius = int(5 * self.scale)
        dot_x = card_x + accent_width // 2
        dot_y = card_y - dot_radius - int(4 * self.scale)
        draw.ellipse(
            [dot_x - dot_radius, dot_y - dot_radius, dot_x + dot_radius, dot_y + dot_radius],
            fill=(*accent_color, 240)
        )

        # Draw location text
        text_x = card_x + accent_width + int(12 * self.scale)
        text_y = card_y + padding_y
        draw.text((text_x, text_y), location, font=loc_font, fill=(*text_color, 250))

        # Draw sub-text (dimmer)
        if sub_text:
            sub_y = text_y + loc_height + int(4 * self.scale)
            draw.text((text_x, sub_y), sub_text, font=sub_font, fill=(*text_color, 160))

        return np.array(overlay)

    def create_date_location_stamp(
        self,
        date_text: str,
        location_text: str = '',
        accent_color: Tuple[int, int, int] = (200, 170, 80),
    ) -> np.ndarray:
        """Create a combined date + location stamp in the bottom-right corner.
        
        Shows date on the left side and location on the right side of a single
        card, separated by a vertical divider. Used for scene-setting context.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        date_font = self._get_font('location_stamp', bold=True)
        loc_font = self._get_font('location_stamp_sub')

        # Measure text
        date_bbox = draw.textbbox((0, 0), date_text, font=date_font)
        date_width = date_bbox[2] - date_bbox[0]
        date_height = date_bbox[3] - date_bbox[1]

        loc_width = 0
        if location_text:
            loc_bbox = draw.textbbox((0, 0), location_text, font=loc_font)
            loc_width = loc_bbox[2] - loc_bbox[0]

        # Card dimensions
        padding_x = int(20 * self.scale)
        padding_y = int(10 * self.scale)
        divider_width = int(2 * self.scale) if location_text else 0
        gap = int(16 * self.scale) if location_text else 0
        card_width = date_width + loc_width + padding_x * 2 + divider_width + gap * 2
        card_height = date_height + padding_y * 2

        # Position: bottom-right
        margin = int(60 * self.scale)
        card_x = self.width - margin - card_width
        card_y = self.height - margin - card_height

        # Draw background
        card_bg = PILImage.new('RGBA', (card_width, card_height), (10, 10, 10, 185))
        overlay.paste(card_bg, (card_x, card_y), card_bg)

        # Draw date text (accent color)
        date_x = card_x + padding_x
        date_y = card_y + padding_y
        draw.text((date_x, date_y), date_text, font=date_font, fill=(*accent_color, 250))

        if location_text:
            # Draw divider
            div_x = date_x + date_width + gap
            div_y1 = card_y + int(6 * self.scale)
            div_y2 = card_y + card_height - int(6 * self.scale)
            draw.line([(div_x, div_y1), (div_x, div_y2)], fill=(*accent_color, 120), width=divider_width)

            # Draw location text (white, dimmer)
            loc_x = div_x + divider_width + gap
            loc_y = card_y + padding_y + (date_height - (draw.textbbox((0, 0), location_text, font=loc_font)[3] - draw.textbbox((0, 0), location_text, font=loc_font)[1])) // 2
            draw.text((loc_x, loc_y), location_text, font=loc_font, fill=(230, 230, 230, 210))

        return np.array(overlay)

    def create_progress_indicator(
        self,
        sections: List[str],
        current_section: str,
        accent_color: Tuple[int, int, int] = (200, 170, 80),
    ) -> np.ndarray:
        """Create a minimal timeline progress indicator showing story position.
        
        Renders a horizontal line with dots for each section at the top of the
        frame. The current section's dot is highlighted and labeled. This gives
        viewers a sense of where they are in the biography arc.
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        label_font = self._get_font('progress_year')

        # Filter out empty-title sections (COLD_OPEN, CTA)
        visible_sections = [s for s in sections if self.SECTION_TITLES.get(s, '')]
        if not visible_sections or current_section not in visible_sections:
            return np.array(overlay)

        # Dimensions
        total_width = int(self.width * 0.35)
        start_x = self.width - int(60 * self.scale) - total_width
        y_pos = int(40 * self.scale)
        dot_radius = int(5 * self.scale)
        active_radius = int(7 * self.scale)

        n = len(visible_sections)
        if n < 2:
            return np.array(overlay)

        spacing = total_width // (n - 1)

        # Draw connecting line (dim)
        line_y = y_pos + active_radius
        draw.line(
            [(start_x, line_y), (start_x + (n - 1) * spacing, line_y)],
            fill=(255, 255, 255, 60),
            width=int(2 * self.scale)
        )

        current_idx = visible_sections.index(current_section) if current_section in visible_sections else -1

        for i, section in enumerate(visible_sections):
            cx = start_x + i * spacing
            cy = line_y
            is_current = (i == current_idx)
            is_past = (i < current_idx)

            if is_current:
                # Active dot (larger, accent color)
                draw.ellipse(
                    [cx - active_radius, cy - active_radius, cx + active_radius, cy + active_radius],
                    fill=(*accent_color, 240)
                )
                # Label below dot
                title = self.SECTION_TITLES.get(section, section)
                if title:
                    label_bbox = draw.textbbox((0, 0), title, font=label_font)
                    label_width = label_bbox[2] - label_bbox[0]
                    label_x = cx - label_width // 2
                    label_y = cy + active_radius + int(6 * self.scale)
                    draw.text((label_x, label_y), title, font=label_font, fill=(*accent_color, 220))
            elif is_past:
                # Past dot (smaller, accent color, dimmer)
                draw.ellipse(
                    [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                    fill=(*accent_color, 140)
                )
            else:
                # Future dot (smaller, white, dim)
                draw.ellipse(
                    [cx - dot_radius, cy - dot_radius, cx + dot_radius, cy + dot_radius],
                    fill=(255, 255, 255, 70)
                )

        return np.array(overlay)

    def create_animated_counter_frames(
        self,
        target_number: int,
        prefix: str = '',
        suffix: str = '',
        num_frames: int = 30,
        accent_color: Tuple[int, int, int] = (200, 170, 80),
    ) -> List[np.ndarray]:
        """Create frames for an animated number counter (Shivanshu-style).
        
        Returns a list of RGBA numpy arrays that count from 0 up to target_number.
        Each frame shows the number at a different stage of the count-up animation
        with an ease-out curve so the counting slows down near the target.
        
        Args:
            target_number: The final number to count up to (e.g., 1000000)
            prefix: Text before the number (e.g., "$")
            suffix: Text after the number (e.g., " million")
            num_frames: How many frames in the animation
            accent_color: Color for the number text
        """
        frames = []
        font = self._get_font('year_stamp', bold=True)

        for i in range(num_frames):
            progress = i / max(1, num_frames - 1)
            # Ease-out cubic: fast start, slow finish
            eased = 1.0 - (1.0 - progress) ** 3
            current_number = int(target_number * eased)

            display_text = f"{prefix}{current_number:,}{suffix}"

            overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            text_bbox = draw.textbbox((0, 0), display_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            padding_x = int(40 * self.scale)
            padding_y = int(24 * self.scale)
            card_width = text_width + padding_x * 2
            card_height = text_height + padding_y * 2
            card_x = (self.width - card_width) // 2
            card_y = (self.height - card_height) // 2

            card_bg = PILImage.new('RGBA', (card_width, card_height), (10, 10, 10, 180))
            overlay.paste(card_bg, (card_x, card_y), card_bg)

            text_x = card_x + padding_x
            text_y = card_y + padding_y
            draw.text((text_x, text_y), display_text, font=font, fill=(*accent_color, 255))

            frames.append(np.array(overlay))

        return frames

    def create_slide_in_overlay(
        self,
        text: str,
        position: str = 'bottom_left',
        slide_from: str = 'left',
        progress: float = 1.0,
        accent_color: Tuple[int, int, int] = (200, 170, 80),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> np.ndarray:
        """Create a single frame of a slide-in text overlay (Shivanshu-style).
        
        The text slides in from the specified direction based on the progress value.
        At progress=0.0 the text is fully off-screen, at progress=1.0 it's at rest.
        
        Args:
            text: Text to display
            position: Final resting position ('bottom_left', 'bottom_right')
            slide_from: Direction text slides from ('left', 'right', 'bottom')
            progress: Animation progress 0.0 to 1.0 (use ease-out for smooth motion)
            accent_color: Accent bar color
            text_color: Text color
        """
        overlay = PILImage.new('RGBA', (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        font = self._get_font('info_card', bold=True)

        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding_x = int(24 * self.scale)
        padding_y = int(12 * self.scale)
        accent_width = int(4 * self.scale)
        card_width = text_width + padding_x * 2 + accent_width + int(12 * self.scale)
        card_height = text_height + padding_y * 2

        margin = int(80 * self.scale)

        # Calculate final resting position
        if position == 'bottom_right':
            rest_x = self.width - margin - card_width
            rest_y = self.height - margin - card_height
        else:  # bottom_left
            rest_x = margin
            rest_y = self.height - margin - card_height

        # Apply slide offset based on progress
        slide_distance = card_width + margin + int(20 * self.scale)
        remaining = 1.0 - min(1.0, max(0.0, progress))

        if slide_from == 'left':
            card_x = int(rest_x - slide_distance * remaining)
        elif slide_from == 'right':
            card_x = int(rest_x + slide_distance * remaining)
        elif slide_from == 'bottom':
            card_x = rest_x
            rest_y = int(rest_y + (card_height + margin) * remaining)
        else:
            card_x = int(rest_x - slide_distance * remaining)

        card_y = rest_y

        # Only draw if at least partially visible
        if card_x + card_width < 0 or card_x > self.width:
            return np.array(overlay)

        # Draw background
        card_bg = PILImage.new('RGBA', (card_width, card_height), (15, 15, 15, 200))
        paste_x = max(0, card_x)
        paste_y = max(0, card_y)
        overlay.paste(card_bg, (paste_x, paste_y), card_bg)

        # Draw accent bar
        accent_bar = PILImage.new('RGBA', (accent_width, card_height), (*accent_color, 240))
        overlay.paste(accent_bar, (max(0, card_x), paste_y), accent_bar)

        # Draw text
        text_x = card_x + accent_width + int(12 * self.scale) + padding_x
        text_y = card_y + padding_y
        if 0 < text_x < self.width:
            draw.text((text_x, text_y), text, font=font, fill=(*text_color, 240))

        return np.array(overlay)
