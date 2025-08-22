"""
Theme Manager for Image Space Application

This module provides theme management functionality including dark mode support
for the Image Space application. Themes are implemented using Qt stylesheets
that can be applied to the entire application or individual widgets.
"""

from PySide6.QtCore import QSettings
from PySide6.QtGui import QPalette, QColor
from typing import Optional, Dict, Any


class ThemeManager:
    """Manages application themes and styling."""
    
    # Theme names
    LIGHT_THEME = "light"
    DARK_THEME = "dark"
    SYSTEM_THEME = "system"
    
    def __init__(self):
        """Initialize the theme manager."""
        self.current_theme = self.LIGHT_THEME
        self.settings = QSettings("ImageSpace", "ImageProcessor")
        
    def get_available_themes(self) -> list:
        """Get list of available theme names."""
        return [self.LIGHT_THEME, self.DARK_THEME]
    
    def load_theme_preference(self) -> str:
        """Load saved theme preference from settings."""
        return self.settings.value("theme", self.LIGHT_THEME)
    
    def save_theme_preference(self, theme_name: str):
        """Save theme preference to settings."""
        self.settings.setValue("theme", theme_name)
        self.current_theme = theme_name
    
    def apply_theme(self, widget, theme_name: Optional[str] = None):
        """
        Apply a theme to a widget (typically the main application or window).
        
        Args:
            widget: The QWidget or QApplication to apply the theme to
            theme_name: Name of the theme to apply (uses saved preference if None)
        """
        if theme_name is None:
            theme_name = self.load_theme_preference()
        
        if theme_name == self.DARK_THEME:
            widget.setStyleSheet(self.get_dark_stylesheet())
        elif theme_name == self.LIGHT_THEME:
            widget.setStyleSheet(self.get_light_stylesheet())
        
        self.current_theme = theme_name
    
    def get_light_stylesheet(self) -> str:
        """
        Get the light theme stylesheet.
        Returns an empty string to use system defaults.
        """
        return ""
    
    def get_dark_stylesheet(self) -> str:
        """
        Get the dark theme stylesheet.
        This provides a comprehensive dark mode theme for the application.
        """
        return """
        /* Main Window and Base Widgets */
        QMainWindow {
            background-color: #1e1e1e;
            color: #e0e0e0;
        }
        
        QWidget {
            background-color: #2b2b2b;
            color: #e0e0e0;
        }
        
        /* Buttons */
        QPushButton {
            background-color: #3c3c3c;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 3px 10px;
            border-radius: 3px;
            min-height: 20px;
            max-height: 32px;
            text-align: center;
        }
        
        QPushButton:hover {
            background-color: #484848;
            border: 1px solid #666;
        }
        
        QPushButton:pressed {
            background-color: #2a2a2a;
            border: 1px solid #444;
        }
        
        QPushButton:disabled {
            background-color: #2b2b2b;
            color: #666;
            border: 1px solid #444;
        }
        
        /* Input Fields */
        QLineEdit, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit {
            background-color: #3c3c3c;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 0px;
            border-radius: 2px;
            selection-background-color: #0078d4;
        }
        
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
            border: 1px solid #0078d4;
        }
        
        QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            background-color: #2b2b2b;
            color: #666;
        }
        
        /* ComboBox */
        QComboBox {
            background-color: #3c3c3c;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 0px;
            border-radius: 2px;
            min-height: 20px;
        }
        
        QComboBox:hover {
            border: 1px solid #666;
        }
        
        QComboBox:disabled {
            background-color: #2b2b2b;
            color: #666;
        }
        
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        
        QComboBox::down-arrow {
            image: none;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 5px solid #e0e0e0;
            margin-right: 5px;
        }
        
        QComboBox QAbstractItemView {
            background-color: #3c3c3c;
            color: #e0e0e0;
            border: 1px solid #555;
            selection-background-color: #0078d4;
        }
        
        /* List Widget */
        QListWidget {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #555;
            alternate-background-color: #323232;
            outline: none;
        }
        
        QListWidget::item {
            padding: 3px;
        }
        
        QListWidget::item:selected {
            background-color: #0078d4;
            color: white;
        }
        
        QListWidget::item:hover {
            background-color: #3c3c3c;
        }
        
        /* Text Edit */
        QTextEdit, QPlainTextEdit {
            background-color: #1e1e1e;
            color: #e0e0e0;
            border: 1px solid #555;
            selection-background-color: #0078d4;
        }
        
        /* Progress Bar */
        QProgressBar {
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-radius: 2px;
            text-align: center;
            color: #e0e0e0;
        }
        
        QProgressBar::chunk {
            background-color: #0078d4;
            border-radius: 2px;
        }
        
        /* Sliders */
        QSlider::groove:horizontal {
            background-color: #3c3c3c;
            height: 6px;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background-color: #0078d4;
            width: 14px;
            height: 14px;
            margin: -4px 0;
            border-radius: 7px;
        }
        
        QSlider::handle:horizontal:hover {
            background-color: #1e90ff;
        }
        
        QSlider::sub-page:horizontal {
            background-color: #0078d4;
            border-radius: 2px;
        }
        
        /* Check Box */
        QCheckBox {
            color: #e0e0e0;
            spacing: 5px;
        }
        
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            margin: 1px;
        }
        
        QCheckBox::indicator:unchecked {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
        }
        
        QCheckBox::indicator:checked {
            background-color: #0078d4;
            border: 1px solid #0078d4;
            border-radius: 3px;
            image: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTAiIGhlaWdodD0iMTAiIHZpZXdCb3g9IjAgMCAxMCAxMCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTggM0w0IDdMMiA1IiBzdHJva2U9IndoaXRlIiBzdHJva2Utd2lkdGg9IjEuNSIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIi8+Cjwvc3ZnPgo=);
        }
        
        QCheckBox::indicator:checked:hover {
            background-color: #1e90ff;
            border: 1px solid #1e90ff;
        }
        
        QCheckBox::indicator:unchecked:hover {
            background-color: #484848;
            border: 1px solid #666;
        }
        
        QCheckBox::indicator:disabled {
            background-color: #2b2b2b;
            border: 1px solid #444;
        }
        
        /* Radio Button */
        QRadioButton {
            color: #e0e0e0;
            spacing: 5px;
        }
        
        QRadioButton::indicator {
            width: 15px;
            height: 15px;
            border-radius: 8px;
        }
        
        QRadioButton::indicator:unchecked {
            background-color: #3c3c3c;
            border: 1px solid #555;
        }
        
        QRadioButton::indicator:checked {
            background-color: #0078d4;
            border: 1px solid #0078d4;
        }
        
        QRadioButton::indicator:checked {
            image: none;
        }
        
        /* Group Box */
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
            color: #e0e0e0;
        }
        
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px 0 5px;
            background-color: #2b2b2b;
        }
        
        /* ToolBox */
        QToolBox {
            background-color: #2b2b2b;
            border: 1px solid #555;
        }
        
        QToolBox::tab {
            background-color: #3c3c3c;
            color: #e0e0e0;
            border: 1px solid #555;
            padding: 5px;
        }
        
        QToolBox::tab:selected {
            background-color: #484848;
            font-weight: bold;
        }
        
        QToolBox::tab:hover {
            background-color: #404040;
        }
        
        /* Menu Bar and Menus */
        QMenuBar {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border-bottom: 1px solid #555;
        }
        
        QMenuBar::item {
            padding: 4px 10px;
            background: transparent;
        }
        
        QMenuBar::item:selected {
            background-color: #3c3c3c;
        }
        
        QMenuBar::item:pressed {
            background-color: #0078d4;
        }
        
        QMenu {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border: 1px solid #555;
        }
        
        QMenu::item {
            padding: 5px 25px;
        }
        
        QMenu::item:selected {
            background-color: #0078d4;
        }
        
        QMenu::separator {
            height: 1px;
            background-color: #555;
            margin: 3px 10px;
        }
        
        /* Status Bar */
        QStatusBar {
            background-color: #2b2b2b;
            color: #e0e0e0;
            border-top: 1px solid #555;
        }
        
        /* Scroll Bars */
        QScrollBar:vertical {
            background-color: #2b2b2b;
            width: 12px;
            border: none;
        }
        
        QScrollBar::handle:vertical {
            background-color: #555;
            min-height: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:vertical:hover {
            background-color: #666;
        }
        
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
            height: 0px;
        }
        
        QScrollBar:horizontal {
            background-color: #2b2b2b;
            height: 12px;
            border: none;
        }
        
        QScrollBar::handle:horizontal {
            background-color: #555;
            min-width: 20px;
            border-radius: 6px;
        }
        
        QScrollBar::handle:horizontal:hover {
            background-color: #666;
        }
        
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
            width: 0px;
        }
        
        /* Tab Widget */
        QTabWidget::pane {
            background-color: #2b2b2b;
            border: 1px solid #555;
        }
        
        QTabBar::tab {
            background-color: #3c3c3c;
            color: #e0e0e0;
            padding: 5px 10px;
            margin-right: 2px;
        }
        
        QTabBar::tab:selected {
            background-color: #2b2b2b;
            border: 1px solid #555;
            border-bottom: none;
        }
        
        QTabBar::tab:hover {
            background-color: #484848;
        }
        
        /* Graphics View - Image Display Area */
        QGraphicsView {
            background-color: #1e1e1e;
            border: 1px solid #555;
        }
        
        /* Labels */
        QLabel {
            color: #e0e0e0;
            background: transparent;
        }
        
        QLabel:disabled {
            color: #666;
        }
        
        /* Frame */
        QFrame {
            color: #e0e0e0;
        }
        
        QFrame[frameShape="4"],  /* QFrame::HLine */
        QFrame[frameShape="5"] { /* QFrame::VLine */
            background-color: #555;
            max-height: 1px;
            border: 2px;
        }
        
        /* ToolTips */
        QToolTip {
            background-color: #3c3c3c;
            color: #e0e0e0;
            border: 1px solid #666;
            padding: 3px;
        }
        
        /* Special styling for server status label */
        QLabel#serverStatusLabel {
            background-color: #2b2b2b;
            padding: 5px;
            border: 1px solid #555;
            border-radius: 3px;
        }
        """
    
    def get_theme_colors(self, theme_name: Optional[str] = None) -> Dict[str, str]:
        """
        Get a dictionary of theme colors for custom drawing operations.
        
        Args:
            theme_name: Name of the theme (uses current theme if None)
            
        Returns:
            Dictionary of color values for various UI elements
        """
        if theme_name is None:
            theme_name = self.current_theme
        
        if theme_name == self.DARK_THEME:
            return {
                'background': '#1e1e1e',
                'surface': '#2b2b2b',
                'surface_variant': '#3c3c3c',
                'primary': '#0078d4',
                'primary_variant': '#1e90ff',
                'text': '#e0e0e0',
                'text_secondary': '#a0a0a0',
                'text_disabled': '#666666',
                'border': '#555555',
                'border_hover': '#666666',
                'error': '#f44336',
                'warning': '#ff9800',
                'success': '#4caf50',
                'info': '#2196f3'
            }
        else:
            # Light theme colors (default system colors)
            return {
                'background': '#ffffff',
                'surface': '#f5f5f5',
                'surface_variant': '#e0e0e0',
                'primary': '#0078d4',
                'primary_variant': '#106ebe',
                'text': '#000000',
                'text_secondary': '#666666',
                'text_disabled': '#999999',
                'border': '#cccccc',
                'border_hover': '#999999',
                'error': '#d32f2f',
                'warning': '#f57c00',
                'success': '#388e3c',
                'info': '#1976d2'
            }
    
    def create_palette(self, theme_name: Optional[str] = None) -> QPalette:
        """
        Create a QPalette for the specified theme.
        This can be used as an alternative to stylesheets.
        
        Args:
            theme_name: Name of the theme (uses current theme if None)
            
        Returns:
            QPalette configured for the theme
        """
        if theme_name is None:
            theme_name = self.current_theme
        
        palette = QPalette()
        
        if theme_name == self.DARK_THEME:
            # Window colors
            palette.setColor(QPalette.Window, QColor(43, 43, 43))
            palette.setColor(QPalette.WindowText, QColor(224, 224, 224))
            
            # Base colors (for input widgets)
            palette.setColor(QPalette.Base, QColor(60, 60, 60))
            palette.setColor(QPalette.AlternateBase, QColor(50, 50, 50))
            palette.setColor(QPalette.Text, QColor(224, 224, 224))
            
            # Button colors
            palette.setColor(QPalette.Button, QColor(60, 60, 60))
            palette.setColor(QPalette.ButtonText, QColor(224, 224, 224))
            
            # Highlight colors
            palette.setColor(QPalette.Highlight, QColor(0, 120, 212))
            palette.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
            
            # Other colors
            palette.setColor(QPalette.BrightText, QColor(255, 255, 255))
            palette.setColor(QPalette.Link, QColor(0, 120, 212))
            palette.setColor(QPalette.LinkVisited, QColor(255, 0, 255))
            
            # Disabled colors
            palette.setColor(QPalette.Disabled, QPalette.WindowText, QColor(102, 102, 102))
            palette.setColor(QPalette.Disabled, QPalette.Text, QColor(102, 102, 102))
            palette.setColor(QPalette.Disabled, QPalette.ButtonText, QColor(102, 102, 102))
        
        return palette


# Global theme manager instance
theme_manager = ThemeManager()