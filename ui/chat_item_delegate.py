# app/ui/chat_item_delegate.py
import html
import logging
import os
from datetime import datetime
from typing import Optional, Dict, Any, Tuple, TYPE_CHECKING

from PySide6.QtCore import (QModelIndex, QRect, QPoint, QSize, Qt, QObject,
                            QByteArray, QPersistentModelIndex, Slot, Signal)
from PySide6.QtGui import (QPainter, QColor, QFontMetrics, QTextDocument, QPixmap,
                           QFont, QMovie, QPen, QTextOption, QPalette)  # Added QTextOption, QPalette
from PySide6.QtWidgets import QStyledItemDelegate, QStyle, QStyleOptionViewItem, QWidget

try:
    # Corrected paths
    from models.chat_message import ChatMessage, USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE
    from models.message_enums import MessageLoadingState
    from .chat_list_model import ChatMessageRole, LoadingStatusRole  # Assuming it's in the same UI package
    from utils import constants
except ImportError as e_delegate_import:
    logging.getLogger(__name__).critical(f"ChatItemDelegate: Critical import error: {e_delegate_import}", exc_info=True)
    # Fallback types
    ChatMessage = type("ChatMessage", (object,), {})  # type: ignore
    MessageLoadingState = type("MessageLoadingState", (object,), {})  # type: ignore
    Qt = type("Qt", (object,), {"ItemDataRole": type("ItemDataRole", (object,), {"UserRole": 1000})})  # type: ignore
    ChatMessageRole = Qt.ItemDataRole.UserRole + 1  # type: ignore
    LoadingStatusRole = Qt.ItemDataRole.UserRole + 2  # type: ignore
    constants = type("constants", (object,), {})  # type: ignore
    USER_ROLE, MODEL_ROLE, SYSTEM_ROLE, ERROR_ROLE = "user", "model", "system", "error"
    raise

if TYPE_CHECKING:
    from PySide6.QtWidgets import QListView

logger = logging.getLogger(__name__)

# --- Constants for Bubble Styling (pulled from original, ensure they align with constants.py) ---
# These are defined here for clarity within the delegate, but could also be solely in constants.py
BUBBLE_PADDING_V = constants.CHAT_FONT_SIZE + 2 if hasattr(constants, 'CHAT_FONT_SIZE') else 14
BUBBLE_PADDING_H = constants.CHAT_FONT_SIZE + 6 if hasattr(constants, 'CHAT_FONT_SIZE') else 18
BUBBLE_MARGIN_V = constants.CHAT_FONT_SIZE if hasattr(constants,
                                                      'CHAT_FONT_SIZE') else 12  # Vertical margin between bubbles
BUBBLE_MARGIN_H = 16  # Horizontal margin from view edge
BUBBLE_RADIUS = 12
IMAGE_PADDING = 5
MAX_IMAGE_WIDTH = 250
MAX_IMAGE_HEIGHT = 250
MIN_BUBBLE_WIDTH = 60  # Minimum width for a bubble to ensure it looks like a bubble
USER_BUBBLE_INDENT_FACTOR = 0.15  # How much user bubbles are indented from the right (0.0 to 1.0)
TIMESTAMP_PADDING_TOP = 6
TIMESTAMP_HEIGHT = 15  # Approximate height for timestamp text
BUBBLE_MAX_WIDTH_PERCENTAGE = 0.78  # Max width of bubble relative to view width

INDICATOR_SIZE = QSize(16, 16)  # Smaller indicator
INDICATOR_PADDING_X = 8  # Padding from bubble edge
INDICATOR_PADDING_Y = 8  # Padding from bubble top

# Colors (ensure these match your constants.py or theme)
USER_BUBBLE_COLOR = QColor(getattr(constants, 'USER_BUBBLE_COLOR_HEX', "#0a7cff"))
USER_TEXT_COLOR = QColor(getattr(constants, 'USER_TEXT_COLOR_HEX', "#ffffff"))
AI_BUBBLE_COLOR = QColor(getattr(constants, 'AI_BUBBLE_COLOR_HEX', "#3E3E3E"))
AI_TEXT_COLOR = QColor(getattr(constants, 'AI_TEXT_COLOR_HEX', "#E0E0E0"))
SYSTEM_BUBBLE_COLOR = QColor(getattr(constants, 'SYSTEM_BUBBLE_COLOR_HEX', "#5A5A5A"))
SYSTEM_TEXT_COLOR = QColor(getattr(constants, 'SYSTEM_TEXT_COLOR_HEX', "#B0B0B0"))
ERROR_BUBBLE_COLOR = QColor(getattr(constants, 'ERROR_BUBBLE_COLOR_HEX', "#730202"))
ERROR_TEXT_COLOR = QColor(getattr(constants, 'ERROR_TEXT_COLOR_HEX', "#FFCCCC"))

BUBBLE_BORDER_COLOR = QColor(getattr(constants, 'BUBBLE_BORDER_COLOR_HEX', "#2D2D2D"))
TIMESTAMP_COLOR = QColor(getattr(constants, 'TIMESTAMP_COLOR_HEX', "#888888"))
CODE_BLOCK_BG_COLOR = QColor(getattr(constants, 'CODE_BLOCK_BG_COLOR_HEX', "#1E1E1E"))


class ChatItemDelegate(QStyledItemDelegate):
    """
    Custom delegate for painting ChatMessage objects in a QListView.
    Handles different bubble styles, text formatting, image display (placeholder),
    timestamps, and loading indicators.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._font = QFont(getattr(constants, 'CHAT_FONT_FAMILY', "Segoe UI"), getattr(constants, 'CHAT_FONT_SIZE', 10))
        self._font_metrics = QFontMetrics(self._font)
        self._timestamp_font = QFont(self._font.family(), self._font.pointSize() - 2)
        self._timestamp_font_metrics = QFontMetrics(self._timestamp_font)

        self._text_doc_cache: Dict[
            Tuple[str, int, str, str], QTextDocument] = {}  # key: (msg_id, width, role, text_content_hash)
        self._image_pixmap_cache: Dict[str, QPixmap] = {}  # Not fully implemented in paint yet

        self._loading_animation_movie_template: Optional[QMovie] = None
        self._completed_icon_pixmap: Optional[QPixmap] = None
        self._error_icon_pixmap: Optional[QPixmap] = None
        self._active_loading_movies: Dict[QPersistentModelIndex, QMovie] = {}
        self._view_ref: Optional['QListView'] = None  # Specifically QListView for update()

        self._bubble_stylesheet_content: str = self._load_bubble_stylesheet()
        self._init_indicator_assets()
        logger.info("ChatItemDelegate initialized.")

    def _load_bubble_stylesheet(self) -> str:
        try:
            qss_path = getattr(constants, 'BUBBLE_STYLESHEET_PATH', "")
            if qss_path and os.path.exists(qss_path):
                with open(qss_path, "r", encoding="utf-8") as f:
                    content = f.read()
                logger.info(f"ChatItemDelegate: Loaded bubble stylesheet from: {qss_path}")
                return content
            else:
                logger.warning(
                    f"ChatItemDelegate: Bubble stylesheet not found at '{qss_path}'. Using default QTextDocument styles.")
        except Exception as e:
            logger.error(f"ChatItemDelegate: Error loading bubble stylesheet: {e}", exc_info=True)
        return ""  # Fallback to empty string

    def _init_indicator_assets(self):
        try:
            assets_base = getattr(constants, 'ASSETS_PATH', "./assets")
            loading_gif = os.path.join(assets_base, getattr(constants, 'LOADING_GIF_FILENAME', "loading_orb.gif"))
            if os.path.exists(loading_gif):
                self._loading_animation_movie_template = QMovie(loading_gif)
                if self._loading_animation_movie_template.isValid():
                    self._loading_animation_movie_template.setScaledSize(INDICATOR_SIZE)
                else:
                    self._loading_animation_movie_template = None; logger.error(f"Invalid loading GIF: {loading_gif}")
            else:
                logger.warning(f"Loading GIF not found: {loading_gif}")

            completed_png = os.path.join(assets_base, "loading_complete.png")  # Example name
            if os.path.exists(completed_png):
                self._completed_icon_pixmap = QPixmap(completed_png).scaled(INDICATOR_SIZE,
                                                                            Qt.AspectRatioMode.KeepAspectRatio,
                                                                            Qt.TransformationMode.SmoothTransformation)
            else:
                logger.warning(f"Completed icon not found: {completed_png}")

            error_png = os.path.join(assets_base, "loading_error.png")  # Example name
            if os.path.exists(error_png):
                self._error_icon_pixmap = QPixmap(error_png).scaled(INDICATOR_SIZE, Qt.AspectRatioMode.KeepAspectRatio,
                                                                    Qt.TransformationMode.SmoothTransformation)
            else:
                logger.warning(f"Error icon not found: {error_png}")
        except Exception as e:
            logger.exception(f"Error initializing indicator assets: {e}")

    def setView(self, view: 'QListView'):  # Type hint specifically for QListView
        self._view_ref = view

    @Slot(int)  # frame_number (unused)
    def _on_movie_frame_changed(self, frame_number: int):
        if not self._view_ref or not self._active_loading_movies: return
        movie_sender = self.sender()
        if not isinstance(movie_sender, QMovie): return

        for p_index, active_movie in list(self._active_loading_movies.items()):  # Iterate copy for safe removal
            if active_movie == movie_sender:
                if p_index.isValid() and self._view_ref.model() and \
                        self._view_ref.model().data(p_index,
                                                    LoadingStatusRole) == MessageLoadingState.LOADING:  # type: ignore
                    self._view_ref.update(p_index)  # Request repaint for this item
                else:  # State changed or index invalid, remove movie
                    self._remove_active_movie(p_index)
                break

    def _remove_active_movie(self, p_index: QPersistentModelIndex):
        if p_index in self._active_loading_movies:
            movie = self._active_loading_movies.pop(p_index)
            movie.stop()
            try:
                movie.frameChanged.disconnect(self._on_movie_frame_changed)
            except (TypeError, RuntimeError):
                pass  # Ignore if not connected or already disconnected
            # movie.deleteLater() # QMovie is a QObject, let Qt handle deletion if parented, or if explicitly needed.
            # If created with self._view_ref as parent, it's handled.
            # If created with `self` as parent, also handled.
            # If parent is None, then deleteLater is good.
            # For movies created on-the-fly in paint, they might need explicit deletion
            # if not parented and not stored for long.
            # Here, they are stored in _active_loading_movies, so manage their lifecycle.
            if movie.parent() is None: movie.deleteLater()

    def clearCache(self):
        """Clears internal caches. Call when model resets or major changes occur."""
        self._text_doc_cache.clear()
        self._image_pixmap_cache.clear()  # Not used yet, but good practice
        for p_index in list(self._active_loading_movies.keys()):  # Iterate copy
            self._remove_active_movie(p_index)
        logger.debug("ChatItemDelegate: Caches cleared.")

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)  # Ensure smooth text

        message: Optional[ChatMessage] = index.data(ChatMessageRole)  # type: ignore
        if not isinstance(message, ChatMessage):  # type: ignore
            super().paint(painter, option, index)  # Default paint for non-ChatMessage items
            painter.restore()
            return

        loading_status: MessageLoadingState = index.model().data(index, LoadingStatusRole)  # type: ignore
        if not isinstance(loading_status, MessageLoadingState):  # type: ignore
            loading_status = MessageLoadingState.IDLE  # type: ignore

        is_user = (message.role == USER_ROLE)
        bubble_color, _ = self._get_colors_for_role(message.role)  # Text color handled by QTextDocument
        available_width = option.rect.width()

        # Calculate necessary sizes
        content_metrics = self._calculate_content_metrics(message, available_width)
        bubble_rect = self._get_bubble_rect(option.rect, content_metrics["bubble_visual_size"], is_user,
                                            available_width)

        # Draw bubble background
        painter.setPen(QPen(BUBBLE_BORDER_COLOR, 0.8))  # Slightly thicker border
        painter.setBrush(bubble_color)
        painter.drawRoundedRect(bubble_rect, BUBBLE_RADIUS, BUBBLE_RADIUS)

        # Content drawing rectangle (inside padding)
        content_draw_rect = bubble_rect.adjusted(BUBBLE_PADDING_H, BUBBLE_PADDING_V,
                                                 -BUBBLE_PADDING_H, -BUBBLE_PADDING_V)

        # Draw Text using QTextDocument
        if message.text:
            text_doc = self._get_prepared_text_document(message, content_draw_rect.width())
            painter.save()
            painter.translate(content_draw_rect.topLeft())
            # Clip drawing to the content area to prevent overflow if text_doc is larger
            clip_rect = QRect(0, 0, content_draw_rect.width(), content_draw_rect.height())
            text_doc.drawContents(painter, clip_rect)
            painter.restore()

        # Draw Loading/Status Indicator (typically for AI messages)
        if message.role == MODEL_ROLE:
            # Position indicator inside the bubble, top-right (or left for user if needed)
            indicator_x = bubble_rect.right() - INDICATOR_SIZE.width() - INDICATOR_PADDING_X
            indicator_y = bubble_rect.top() + INDICATOR_PADDING_Y  # Align with top padding
            indicator_rect = QRect(QPoint(indicator_x, indicator_y), INDICATOR_SIZE)

            persistent_index = QPersistentModelIndex(index)  # Use persistent index for movies

            if loading_status == MessageLoadingState.LOADING:  # type: ignore
                active_movie = self._active_loading_movies.get(persistent_index)
                if not active_movie and self._loading_animation_movie_template:
                    # Create a new movie instance from the template for this item
                    active_movie = QMovie(self._loading_animation_movie_template.fileName(), QByteArray(),
                                          self._view_ref or self)  # Parent to view or self
                    if active_movie.isValid():
                        active_movie.setScaledSize(INDICATOR_SIZE)
                        active_movie.frameChanged.connect(self._on_movie_frame_changed)
                        self._active_loading_movies[persistent_index] = active_movie
                        active_movie.start()
                    else:
                        active_movie = None  # Failed to create/load

                if active_movie and active_movie.isValid() and active_movie.state() == QMovie.MovieState.Running:
                    painter.drawPixmap(indicator_rect, active_movie.currentPixmap())
                elif self._loading_animation_movie_template:  # Fallback if movie didn't start, draw first frame
                    painter.drawPixmap(indicator_rect, self._loading_animation_movie_template.currentPixmap())


            elif loading_status == MessageLoadingState.COMPLETED:  # type: ignore
                self._remove_active_movie(persistent_index)  # Stop animation if it was running
                if self._completed_icon_pixmap: painter.drawPixmap(indicator_rect, self._completed_icon_pixmap)
            elif loading_status == MessageLoadingState.ERROR:  # type: ignore
                self._remove_active_movie(persistent_index)
                if self._error_icon_pixmap: painter.drawPixmap(indicator_rect, self._error_icon_pixmap)
            else:  # IDLE or other states, ensure movie is removed
                self._remove_active_movie(persistent_index)

        # Draw Timestamp
        formatted_timestamp = self._format_timestamp(message.timestamp)
        if formatted_timestamp:
            # Position timestamp below the bubble
            ts_y_pos = bubble_rect.bottom() + TIMESTAMP_PADDING_TOP
            painter.setFont(self._timestamp_font)
            painter.setPen(TIMESTAMP_COLOR)

            timestamp_width = self._timestamp_font_metrics.horizontalAdvance(formatted_timestamp)
            ts_x_pos = bubble_rect.right() - timestamp_width if is_user else bubble_rect.left()
            # Ensure timestamp doesn't go out of item bounds
            ts_x_pos = max(item_rect.left() + BUBBLE_MARGIN_H,
                           min(ts_x_pos, item_rect.right() - BUBBLE_MARGIN_H - timestamp_width))

            painter.drawText(QPoint(ts_x_pos, ts_y_pos + self._timestamp_font_metrics.ascent()), formatted_timestamp)

        # Item selection highlight (optional, usually selection is disabled for chat views)
        if option.state & QStyle.StateFlag.State_Selected:
            highlight_color = option.palette.color(QPalette.ColorRole.Highlight)  # Use theme's highlight
            highlight_color.setAlpha(60)  # Semi-transparent highlight
            painter.fillRect(option.rect, highlight_color)

        painter.restore()

    def sizeHint(self, option: QStyleOptionViewItem, index: QModelIndex) -> QSize:
        message: Optional[ChatMessage] = index.data(ChatMessageRole)  # type: ignore
        if not isinstance(message, ChatMessage):  # type: ignore
            return super().sizeHint(option, index)

        available_width = option.rect.width()  # Width available for the entire item
        content_metrics = self._calculate_content_metrics(message, available_width)

        bubble_height = content_metrics["bubble_visual_size"].height()
        total_height = bubble_height

        if self._format_timestamp(message.timestamp):
            total_height += TIMESTAMP_PADDING_TOP + TIMESTAMP_HEIGHT

        # Add vertical margins for the entire item
        final_height = total_height + BUBBLE_MARGIN_V * 2

        # Ensure a minimum height for very short messages
        min_text_line_height = self._font_metrics.height() + BUBBLE_PADDING_V * 2  # Bubble padding around one line
        min_item_height = min_text_line_height + (TIMESTAMP_PADDING_TOP + TIMESTAMP_HEIGHT if self._format_timestamp(
            message.timestamp) else 0) + BUBBLE_MARGIN_V * 2

        return QSize(available_width, max(final_height, min_item_height))

    def _get_colors_for_role(self, role: str) -> Tuple[QColor, QColor]:
        """Returns (bubble_color, text_color) for a given message role."""
        if role == USER_ROLE: return USER_BUBBLE_COLOR, USER_TEXT_COLOR
        if role == SYSTEM_ROLE: return SYSTEM_BUBBLE_COLOR, SYSTEM_TEXT_COLOR
        if role == ERROR_ROLE: return ERROR_BUBBLE_COLOR, ERROR_TEXT_COLOR
        return AI_BUBBLE_COLOR, AI_TEXT_COLOR  # Default for MODEL_ROLE and others

    def _get_bubble_rect(self, item_rect: QRect, bubble_visual_size: QSize, is_user: bool,
                         available_item_width: int) -> QRect:
        """Calculates the QRect for the chat bubble itself within the item_rect."""
        bubble_w = bubble_visual_size.width()
        bubble_h = bubble_visual_size.height()

        bubble_y = item_rect.top() + BUBBLE_MARGIN_V
        bubble_x: int
        if is_user:
            # User bubbles align to the right, indented
            user_indent_px = int(available_item_width * USER_BUBBLE_INDENT_FACTOR)
            # Calculate right edge and subtract width, then ensure it's not too far left
            bubble_x = item_rect.right() - BUBBLE_MARGIN_H - bubble_w
            bubble_x = max(bubble_x,
                           item_rect.left() + BUBBLE_MARGIN_H + user_indent_px)  # Ensure indent is respected if bubble is very wide
        else:
            # AI/System bubbles align to the left
            bubble_x = item_rect.left() + BUBBLE_MARGIN_H

        return QRect(bubble_x, bubble_y, bubble_w, bubble_h)

    def _calculate_content_metrics(self, message: ChatMessage, total_view_width: int) -> Dict[str, Any]:
        """Calculates the size needed for the bubble content (text, images)."""
        # Max width for the bubble itself, considering item margins
        max_bubble_width_for_item = total_view_width - (2 * BUBBLE_MARGIN_H)

        # Apply percentage based max width for the bubble
        bubble_render_width_limit = int(max_bubble_width_for_item * BUBBLE_MAX_WIDTH_PERCENTAGE)

        is_user = (message.role == USER_ROLE)
        if is_user:  # User bubbles have an additional indent
            user_indent_px = int(total_view_width * USER_BUBBLE_INDENT_FACTOR)
            bubble_render_width_limit = min(bubble_render_width_limit, max_bubble_width_for_item - user_indent_px)

        # Ensure minimum bubble width
        bubble_render_width_limit = max(bubble_render_width_limit, MIN_BUBBLE_WIDTH + 2 * BUBBLE_PADDING_H)

        # Inner content width constraint (bubble width - horizontal paddings)
        inner_content_width_constraint = bubble_render_width_limit - (2 * BUBBLE_PADDING_H)

        current_content_height_accumulator = 0
        max_content_width_used_by_elements = 0  # Tracks the widest element (text or image)

        # Calculate text size
        if message.text:
            text_doc = self._get_prepared_text_document(message, inner_content_width_constraint)
            # Get ideal size first without hard width constraint for text_doc itself
            text_doc.setTextWidth(-1)  # Let it calculate its ideal width
            ideal_text_size = text_doc.size()

            # Now constrain its width for rendering and get resulting height
            actual_text_render_width = min(int(ideal_text_size.width()), inner_content_width_constraint)
            text_doc.setTextWidth(max(1, actual_text_render_width))  # Ensure at least 1px width
            text_render_height = int(text_doc.size().height())

            current_content_height_accumulator += text_render_height
            max_content_width_used_by_elements = max(max_content_width_used_by_elements, actual_text_render_width)

        # TODO: Add image size calculation here if images are implemented
        # if message.has_images:
        #     for img_part in message.image_parts:
        #         # Calculate image display size, add to current_content_height_accumulator & update max_content_width_used_by_elements
        #         img_height = ...; img_width = ...
        #         current_content_height_accumulator += img_height + IMAGE_PADDING
        #         max_content_width_used_by_elements = max(max_content_width_used_by_elements, img_width)

        # Final bubble visual size (includes internal padding)
        final_bubble_height = current_content_height_accumulator + (2 * BUBBLE_PADDING_V)
        final_bubble_width = max(max_content_width_used_by_elements, MIN_BUBBLE_WIDTH) + (2 * BUBBLE_PADDING_H)
        # Ensure bubble width doesn't exceed its render limit
        final_bubble_width = min(final_bubble_width, bubble_render_width_limit)

        return {
            "bubble_visual_size": QSize(final_bubble_width, final_bubble_height),
            "max_inner_content_width_used": max_content_width_used_by_elements,
            "inner_content_width_constraint": inner_content_width_constraint
        }

    def _get_prepared_text_document(self, message: ChatMessage, width_constraint: int) -> QTextDocument:
        """Prepares or retrieves a cached QTextDocument for the message text."""
        text_content_for_cache = message.text if message.text else ""
        # Using a hash of the text content for part of the cache key to handle content changes better
        text_hash = str(hash(text_content_for_cache))  # Simple hash
        cache_key = (message.id, width_constraint, message.role, text_hash)

        cached_doc = self._text_doc_cache.get(cache_key)
        if cached_doc:
            # Ensure cached doc has the correct width constraint
            if abs(cached_doc.textWidth() - max(1, width_constraint)) > 1:  # Tolerance for float precision
                cached_doc.setTextWidth(max(1, width_constraint))
            return cached_doc

        doc = QTextDocument()
        doc.setDefaultFont(self._font)
        doc.setDocumentMargin(0)  # Padding is handled by the bubble drawing

        _, text_color = self._get_colors_for_role(message.role)

        # Apply base stylesheet for colors, then add HTML content
        # This ensures code blocks and other elements inherit base text color if not specified by Markdown->HTML
        base_style = f"body {{ color: {text_color.name()}; }}"
        final_stylesheet = f"{base_style}\n{self._bubble_stylesheet_content}" if self._bubble_stylesheet_content else base_style

        # For system/error messages, use simpler formatting (no markdown)
        is_markdown_source = not (message.role == SYSTEM_ROLE or message.role == ERROR_ROLE)
        html_content = self._convert_text_to_html(text_content_for_cache, is_markdown_source, text_color)

        # Set stylesheet before HTML for elements to pick up styles
        doc.setDefaultStyleSheet(final_stylesheet)
        doc.setHtml(html_content)

        # Set text width for layout calculation
        doc.setTextWidth(max(1, width_constraint))

        # Cache management: simple FIFO if cache gets too large
        if len(self._text_doc_cache) > 150:  # Adjust cache size as needed
            try:
                self._text_doc_cache.pop(next(iter(self._text_doc_cache)))
            except StopIteration:
                pass  # Cache was empty

        self._text_doc_cache[cache_key] = doc
        return doc

    def _convert_text_to_html(self, text: str, use_markdown: bool, default_text_color: QColor) -> str:
        """Converts text to HTML, using Markdown if specified."""
        if not text: return ""

        if use_markdown:
            try:
                import markdown
                # Configure Markdown extensions
                # 'extra' includes fenced_code, tables, footnotes, etc.
                # 'nl2br' converts newlines to <br>
                # 'sane_lists' improves list parsing
                # 'codehilite' for syntax highlighting if Pygments is used by Markdown lib
                md_extensions = [
                    'extra', 'nl2br', 'sane_lists',
                    'markdown.extensions.codehilite',  # Requires Pygments
                    'markdown.extensions.tables',
                    'markdown.extensions.attr_list'  # For adding attributes to elements
                ]
                extension_configs = {
                    'markdown.extensions.codehilite': {
                        'css_class': 'codehilite',  # Class for <pre> block
                        'linenums': False,  # No line numbers from markdown itself
                        'guess_lang': True,
                        'pygments_style': 'material'  # Pygments theme
                    }
                }
                html_from_md = markdown.markdown(text, extensions=md_extensions, extension_configs=extension_configs)
                # Wrap in body with default text color, Markdown output is usually fragments
                return f"<body style='color: {default_text_color.name()};'>{html_from_md}</body>"
            except ImportError:
                logger.warning("Markdown library not found. Falling back to basic HTML conversion.")
                escaped_text = html.escape(text).replace('\n', '<br/>')
                return f"<body style='color: {default_text_color.name()};'><p>{escaped_text}</p></body>"
            except Exception as e_md:
                logger.error(f"Markdown conversion failed: {e_md}. Using basic HTML conversion.", exc_info=True)
                escaped_text = html.escape(text).replace('\n', '<br/>')
                return f"<body style='color: {default_text_color.name()};'><p>{escaped_text}</p></body>"
        else:  # For SYSTEM or ERROR roles, just escape and use <p> with <br>
            escaped_text = html.escape(text).replace('\n', '<br/>')
            return f"<body style='color: {default_text_color.name()};'><p>{escaped_text}</p></body>"

    def _format_timestamp(self, iso_timestamp: Optional[str]) -> Optional[str]:
        """Formats an ISO timestamp string to HH:MM format."""
        if not iso_timestamp: return None
        try:
            dt_object = datetime.fromisoformat(iso_timestamp)
            return dt_object.strftime("%H:%M")
        except (ValueError, TypeError):  # Catch potential parsing errors
            logger.warning(f"Could not parse timestamp: {iso_timestamp}")
            return None