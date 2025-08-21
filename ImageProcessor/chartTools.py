"""
Chart Tools Module

This module contains all functions related to manual chart selection and chart flattening operations.
Extracted from the main image_space.py file for better code organization.

Functions include:
- Manual chart selection with rubber band interface
- Chart flattening with 4-corner perspective correction
- UI helper classes for visual markers
- Chart swatch grid drawing utilities
"""

import functools
import gc

import colour
import numpy as np
import cv2
from PySide6.QtWidgets import (
    QRubberBand, QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsRectItem, QLabel, QGraphicsPixmapItem
)
from PySide6.QtCore import Qt, QRect, QSize
from PySide6.QtGui import QColor, QImage, QPixmap, QPainter
from PySide6.QtSvg import QSvgRenderer


class FixedSizeText(QGraphicsTextItem):
    """Graphics text item that ignores transformations for consistent display."""
    
    def __init__(self, text, x, y, color=QColor('red'), parent=None):
        super().__init__(text, parent)
        self.setPos(x, y)
        self.setDefaultTextColor(color)
        self.setFlag(QGraphicsTextItem.ItemIgnoresTransformations, True)


class FixedSizeEllipse(QGraphicsEllipseItem):
    """Graphics ellipse item that ignores transformations for consistent display."""
    
    def __init__(self, x, y, radius=8, color=QColor('red'), parent=None):
        super().__init__(parent)
        self.setPos(x, y)
        self.radius = radius
        self.color = color
        self.setFlag(QGraphicsEllipseItem.ItemIgnoresTransformations, True)
        # The ellipse is centered on (x, y)
        self.setRect(-radius/2, -radius/2, radius, radius)
    
    def paint(self, painter, option, widget):
        painter.setBrush(self.color)
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(self.rect())


class ChartTools:
    """
    Chart manipulation tools for manual selection and perspective correction.
    
    This class contains methods for:
    - Manual chart region selection using rubber band interface
    - 4-corner perspective flattening with OpenCV
    - ColorChecker swatch grid overlay
    - UI state management for chart tools
    """
    
    @staticmethod
    def create_colored_svg_pixmap(svg_path: str, color: QColor, size: QSize = QSize(16, 16)) -> QPixmap:
        """
        Create a properly antialiased colored SVG pixmap.
        
        Args:
            svg_path: Path to the SVG file
            color: Color to apply to the SVG
            size: Desired size of the output pixmap
            
        Returns:
            QPixmap: Colored SVG rendered as pixmap with proper antialiasing
        """
        # Create SVG renderer
        renderer = QSvgRenderer(svg_path)
        if not renderer.isValid():
            # Fallback to simple circle if SVG fails to load
            pixmap = QPixmap(size)
            pixmap.fill(Qt.transparent)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setBrush(color)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(2, 2, size.width()-4, size.height()-4)
            painter.end()
            return pixmap
        
        # First render SVG in black to create a mask
        mask_pixmap = QPixmap(size)
        mask_pixmap.fill(Qt.transparent)
        painter = QPainter(mask_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        renderer.render(painter)
        painter.end()
        
        # Create final colored pixmap
        colored_pixmap = QPixmap(size)
        colored_pixmap.fill(Qt.transparent)
        painter = QPainter(colored_pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        
        # Draw the colored shape using the SVG as a mask
        painter.setBrush(color)
        painter.setPen(Qt.NoPen)
        painter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        painter.drawPixmap(0, 0, mask_pixmap)
        painter.setCompositionMode(QPainter.CompositionMode_SourceIn)
        painter.fillRect(colored_pixmap.rect(), color)
        
        painter.end()
        return colored_pixmap
    
    @staticmethod
    def exit_manual_mode(func):
        """Decorator that exits manual mode before executing the wrapped function."""
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            if getattr(self, 'manual_selection_mode', False):
                ChartTools.finalize_manual_chart_selection(self, canceled=True)
            result = func(self, *args, **kwargs)
            return result
        return wrapper
    
    @staticmethod
    def finalize_manual_chart_selection(main_window, *, canceled: bool = False):
        """
        Exit manual chart selection (either commit or cancel) and restore UI.

        Args:
            main_window: Reference to the main application window
            canceled: if True, we're aborting manual mode and do NOT commit temp_swatches.
                     if False, this is a real "Finalize Chart" and we copy temp_swatches to chart_swatches.
        """
        # 1) Log & commit (only if finalize, not cancel)
        if canceled:
            main_window.log("[Manual] Manual chart selection canceled.")
        else:
            main_window.log("[Manual] Manual chart selection finalized.")
            # commit the temporary swatches/calibration
            if hasattr(main_window, 'temp_swatches'):
                main_window.chart_swatches = main_window.temp_swatches
            if hasattr(main_window, 'temp_calibration_file'):
                main_window.calibration_file = main_window.temp_calibration_file

        # 2) Reset internal modes & data
        main_window.manual_selection_mode = False
        main_window.flatten_mode = False
        if hasattr(main_window, 'corner_points'):
            main_window.corner_points.clear()
        main_window.flatten_swatch_rects = None

        # 3) Hide any visible rubberband
        if hasattr(main_window, 'rubberBand') and main_window.rubberBand.isVisible():
            main_window.rubberBand.hide()

        # 4) Restore the *original* preview if we have it
        if hasattr(main_window, 'original_preview_pixmap') and main_window.original_preview_pixmap:
            main_window._display_preview(main_window.original_preview_pixmap)
            main_window.showing_chart_preview = False
            main_window.ui.showOriginalImagePushbutton.setText("Show Chart Preview")

        # 5) Disable & reset style of all charttools buttons
        ChartTools.set_chart_tools_enabled(main_window)

        # 6) Hide the toolshelf & instruction overlay
        main_window.ui.detectChartToolshelfFrame.setVisible(False)
        if hasattr(main_window, 'instruction_label') and isinstance(main_window.instruction_label, QLabel):
            main_window.instruction_label.hide()

        # 7) Hide debug frame if present
        main_window.show_debug_frame(False)
    
    @staticmethod
    def set_chart_tools_enabled(main_window, *, manual=False, detect=False, show=False, flatten=False, finalize=False):
        """
        Enable or disable all chart-tools buttons with optional highlight on Manual & Flatten.
        
        Args:
            main_window: Reference to the main application window
            manual: Enable manual selection button
            detect: Enable detect chart button
            show: Enable show original image button
            flatten: Enable flatten chart button
            finalize: Enable finalize chart button
        """
        mapping = {
            'manual':   main_window.ui.manuallySelectChartPushbutton,
            'detect':   main_window.ui.detectChartShelfPushbutton,
            'show':     main_window.ui.showOriginalImagePushbutton,
            'flatten':  main_window.ui.flattenChartImagePushButton,
            'finalize': main_window.ui.finalizeChartPushbutton,
        }
        
        local_vars = locals()
        for key, btn in mapping.items():
            enabled = local_vars[key]
            btn.setEnabled(enabled)
            # highlight Manual & Flatten when active
            if (key == 'manual' and enabled) or (key == 'flatten' and enabled):
                btn.setStyleSheet("background-color: #A5D6A7")
            else:
                btn.setStyleSheet("")
    
    @staticmethod
    def manually_select_chart(main_window):
        """
        Enter manual chart selection mode:
          - Load RAW + thumbnail, reset UI, show instructions, enable only Manual-Select.
        
        Args:
            main_window: Reference to the main application window
        """
        chart_text = main_window.ui.chartPathLineEdit.text().strip()
        if not chart_text:
            main_window.log('[Manual] No chart selected')
            return
        cursor_pixmap = QPixmap(r'resources/icons/crosshair.svg')
        main_window.cursor = cursor_pixmap
        main_window.ui.imagePreviewGraphicsView.setCursor(main_window.cursor)
        
        # Extract actual file path from formatted text like "[Group Name] filename.raw"
        if chart_text.startswith('[') and '] ' in chart_text:
            # Format: "[Group Name] filename.raw" -> get the actual path from chart_groups
            selected_item = main_window.ui.imagesListWidget.currentItem()
            if selected_item:
                meta = selected_item.data(Qt.UserRole)
                if not meta.get('is_group_header', False):
                    path = meta['input_path']
                else:
                    main_window.log('[Manual] Cannot select chart from group header')
                    return
            else:
                main_window.log('[Manual] No image selected')
                return
        else:
            # Direct file path
            path = chart_text
        
        full_fp = main_window.load_raw_image(path)
        if full_fp is None:
            main_window.log(f'[Manual] RAW load failed: {path}')
            return
        main_window.fp_image_array = full_fp

        # build 8-bit thumbnail with proper gamma correction
        # Create a safe copy of full_fp to avoid memory conflicts during color correction
        full_fp_copy = full_fp.copy()
        thumb_corrected = colour.cctf_encoding(full_fp_copy)
        thumb_arr = np.uint8(255 * np.clip(thumb_corrected, 0.0, 1.0))
        
        # Create QImage BEFORE deleting intermediate arrays
        h, w, _ = thumb_arr.shape
        bytes_per_line = w * 3
        qimg = QImage(thumb_arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        main_window.original_preview_pixmap = pixmap
        main_window.current_image_pixmap = pixmap
        main_window._display_preview(pixmap)

        # Disable background RAW loading to prevent conflicts during chart selection
        main_window._clear_preview_state()
        main_window.raw_load_timer.stop()
        
        # Reset modes and hide overlays/debug
        main_window.manual_selection_mode = True
        main_window.flatten_mode = False
        main_window.show_debug_frame(False)
        if hasattr(main_window, 'rubberBand') and main_window.rubberBand.isVisible():
            main_window.rubberBand.hide()

        # Reset and enable only the Manual-Select button
        ChartTools.set_chart_tools_enabled(main_window, manual=True)

        # Show instruction label
        instr = getattr(main_window, 'instruction_label', None)
        if isinstance(instr, QLabel):
            instr.setText("Click and drag box around the colour chart")
            instr.show()
        else:
            main_window.instruction_label = QLabel("Click and drag box around the colour chart")
            main_window.instruction_label.setAlignment(Qt.AlignCenter)
            main_window.ui.verticalLayout_4.insertWidget(
                main_window.ui.verticalLayout_4.indexOf(main_window.ui.imagePreviewGraphicsView),
                main_window.instruction_label
            )
        # Reveal toolshelf
        main_window.ui.detectChartToolshelfFrame.setVisible(True)
    
    @staticmethod
    def on_manual_crop_complete(main_window, rect: QRect):
        """
        Called when the QRect is drawn in manual crop mode
        Enables the buttons for Show Original Image, Flatten Chart Mode and Detect Chart
        
        Args:
            main_window: Reference to the main application window
            rect: The selected rectangle area
        """
        main_window.ui.detectChartShelfPushbutton.setEnabled(True)
        main_window.ui.showOriginalImagePushbutton.setEnabled(True)
        main_window.ui.imagePreviewGraphicsView.unsetCursor()

        css = "background-color: #A5D6A7"
        main_window.ui.flattenChartImagePushButton.setEnabled(True)
        main_window.ui.flattenChartImagePushButton.setStyleSheet(css)
    
    @staticmethod
    def flatten_chart_image(main_window):
        """
        Enter cornerpicking mode to flatten the manually selected chart region;
        on completion, stores the cropped-preview pixmap and enables flattening.
        
        Args:
            main_window: Reference to the main application window
        """
        if not hasattr(main_window, 'cropped_preview_pixmap'):
            main_window.log('[Flatten] No cropped image, select region first')
            return

        main_window.log('[Flatten] Select the 4 corners of the chart')
        main_window.instruction_label.setText('Select the 4 corners of the chart')
        css = "background-color: #A5D6A7"
        main_window.ui.flattenChartImagePushButton.setStyleSheet(css)

        cursor_pixmap = QPixmap(r'resources/icons/crosshair.svg')
        main_window.cursor = cursor_pixmap
        main_window.ui.imagePreviewGraphicsView.setCursor(main_window.cursor)

        main_window.flatten_mode = True
        main_window.corner_points = []

        main_window._display_preview(main_window.cropped_preview_pixmap)
        # enable only Flatten & Finalize
        ChartTools.set_chart_tools_enabled(main_window, flatten=True, finalize=True)
    
    @staticmethod
    def handle_manual_press(main_window, event):
        """
        Handle mouse press event for manual chart selection rubber band.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse press event
            
        Returns:
            bool: True if event was handled
        """
        main_window.origin = event.position().toPoint()
        if not hasattr(main_window, 'rubberBand'):
            main_window.rubberBand = QRubberBand(QRubberBand.Rectangle, main_window.ui.imagePreviewGraphicsView)
        main_window.rubberBand.setGeometry(QRect(main_window.origin, QSize()))
        main_window.rubberBand.show()
        return True
    
    @staticmethod
    def handle_manual_move(main_window, event):
        """
        Handle mouse move event for manual chart selection rubber band.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse move event
            
        Returns:
            bool: True if event was handled
        """
        if hasattr(main_window, 'rubberBand') and main_window.rubberBand.isVisible():
            rect = QRect(main_window.origin, event.position().toPoint()).normalized()
            main_window.rubberBand.setGeometry(rect)
            return True
        return False
    
    @staticmethod
    def handle_manual_release(main_window, event):
        """
        Handle mouse release event for manual chart selection rubber band.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse release event
            
        Returns:
            bool: True if event was handled
        """
        if not hasattr(main_window, 'rubberBand') or not main_window.rubberBand.isVisible():
            return False
            
        # Map rubber-band rect to scene coords
        rect = main_window.rubberBand.geometry()
        tl = main_window.ui.imagePreviewGraphicsView.mapToScene(rect.topLeft())
        br = main_window.ui.imagePreviewGraphicsView.mapToScene(rect.bottomRight())
        x, y = int(tl.x()), int(tl.y())
        w, h = abs(int(br.x() - tl.x())), abs(int(br.y() - tl.y()))

        # Constrain crop region to image boundaries to prevent out-of-bounds access
        img_height, img_width = main_window.fp_image_array.shape[:2]
        pixmap_width = main_window.current_image_pixmap.width()
        pixmap_height = main_window.current_image_pixmap.height()
        
        # Clamp coordinates to valid ranges
        x = max(0, min(x, pixmap_width - 1))
        y = max(0, min(y, pixmap_height - 1))
        w = min(w, pixmap_width - x)
        h = min(h, pixmap_height - y)
        
        # Ensure minimum crop size
        w = max(1, w)
        h = max(1, h)
        
        # For the float array, ensure we don't exceed its bounds
        fp_x = max(0, min(x, img_width - 1))
        fp_y = max(0, min(y, img_height - 1))
        fp_w = min(w, img_width - fp_x)
        fp_h = min(h, img_height - fp_y)
        fp_w = max(1, fp_w)
        fp_h = max(1, fp_h)

        main_window.rubberBand.hide()
        main_window.manual_selection_mode = False

        # Crop data with bounds checking
        main_window.cropped_preview_pixmap = main_window.current_image_pixmap.copy(x, y, w, h)
        main_window.cropped_fp = main_window.fp_image_array[fp_y:fp_y+fp_h, fp_x:fp_x+fp_w, :]

        main_window._display_preview(main_window.cropped_preview_pixmap)
        ChartTools.set_chart_tools_enabled(main_window, detect=True, flatten=True)
        return True
    
    @staticmethod
    def handle_flatten_press(main_window, event):
        """
        Handle mouse press event for chart flattening corner selection.
        
        Args:
            main_window: Reference to the main application window
            event: Mouse press event
            
        Returns:
            bool: True if event was handled
        """
        # Create properly antialiased colored SVG icon
        point_colour = QColor(255, 0, 0)  # Red
        colored_point_pixmap = ChartTools.create_colored_svg_pixmap(
            r'resources/icons/circle-dot.svg', 
            point_colour, 
            QSize(20, 20)  # Larger size for better visibility
        )

        pt = event.position().toPoint()
        sp = main_window.ui.imagePreviewGraphicsView.mapToScene(pt)
        idx = len(main_window.corner_points) + 1
        main_window.corner_points.append((sp.x(), sp.y()))
        main_window.log(f"[Flatten] Point {idx}: ({int(sp.x())}, {int(sp.y())})")

        # Draw UI markers  
        label = FixedSizeText(str(idx), sp.x()+12, sp.y()-8, color=QColor('red'))
        
        # Convert QPixmap to QGraphicsPixmapItem and position it
        point_item = QGraphicsPixmapItem(colored_point_pixmap)
        point_item.setPos(sp.x() - colored_point_pixmap.width()/2, sp.y() - colored_point_pixmap.height()/2)
        
        main_window.previewScene.addItem(point_item)
        main_window.previewScene.addItem(label)

        # Once four points selected, perform flatten
        if idx == 4:
            ChartTools.perform_flatten_transform(main_window)
        return True
    
    @staticmethod
    def perform_flatten_transform(main_window):
        """
        Executes perspective warp and grid overlay after four corner points are set.
        
        Args:
            main_window: Reference to the main application window
        """

        main_window.ui.imagePreviewGraphicsView.unsetCursor()

        PADDING = 600
        pts_src = np.array(main_window.corner_points, dtype=np.float32)
        width = np.linalg.norm(pts_src[1] - pts_src[0])
        height = int(width * 9.0 / 14.0)
        dst_pts = np.array([
            [PADDING, PADDING], [PADDING+width, PADDING],
            [PADDING+width, PADDING+height], [PADDING, PADDING+height]
        ], dtype=np.float32)

        # Extract byte buffer from cropped_preview_pixmap
        qimg = main_window.cropped_preview_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        h0 = qimg.height()
        stride = qimg.bytesPerLine()
        buf = bytes(qimg.constBits())[:h0*stride]
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h0, stride))
        arr = arr[:, :qimg.width()*3].reshape((h0, qimg.width(), 3))

        # Pad and compute warp
        arr_p = cv2.copyMakeBorder(arr, PADDING, PADDING, PADDING, PADDING,
                                  borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
        pts_src_p = pts_src + PADDING
        M = cv2.getPerspectiveTransform(pts_src_p, dst_pts)
        warped = cv2.warpPerspective(arr_p, M, (int(width+2*PADDING), int(height+2*PADDING)))

        # Display warped image
        q2 = QImage(warped.data, warped.shape[1], warped.shape[0], warped.shape[1]*3,
                    QImage.Format_RGB888)
        main_window._display_preview(QPixmap.fromImage(q2))

        # Draw swatch grid and update fp array
        swatch_rect = QRect(PADDING, PADDING, int(width), int(height))
        main_window.flatten_swatch_rects = ChartTools.draw_colorchecker_swatch_grid(
            main_window.previewScene, swatch_rect, n_cols=6, n_rows=4)

        # Transform float-precision data
        fparr = cv2.copyMakeBorder(main_window.cropped_fp, PADDING, PADDING, PADDING, PADDING,
                                   borderType=cv2.BORDER_REFLECT)
        main_window.cropped_fp = cv2.warpPerspective(fparr.astype(np.float32), M,
                                              (warped.shape[1], warped.shape[0]))
        main_window.flatten_mode = False
        main_window.instruction_label.setText(
            "Please Run Detect Chart or Revert image to select new region")
        main_window.ui.finalizeChartPushbutton.setEnabled(True)
        
        # Call preview with error handling to prevent hanging
        try:
            main_window.preview_manual_swatch_correction()
        except Exception as e:
            main_window.log(f"[Flatten] Preview error: {e}")
            # Continue without preview - not critical for functionality
    
    @staticmethod
    def draw_colorchecker_swatch_grid(scene, image_rect, n_cols=6, n_rows=4, color=QColor(0, 255, 0, 90)):
        """
        Draws a 6x4 swatch grid (for ColorChecker) over the image_rect on the provided QGraphicsScene.
        
        Args:
            scene: QGraphicsScene to draw on
            image_rect: QRect defining the image area
            n_cols: Number of columns (default 6 for ColorChecker)
            n_rows: Number of rows (default 4 for ColorChecker)
            color: Color for the swatch overlays
            
        Returns:
            list: List of QRect for each swatch, row-wise
        """
        x0, y0, w, h = image_rect.left(), image_rect.top(), image_rect.width(), image_rect.height()
        swatch_rects = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                sw_x0 = x0 + col * w / n_cols
                sw_y0 = y0 + row * h / n_rows
                sw_x1 = x0 + (col + 1) * w / n_cols
                sw_y1 = y0 + (row + 1) * h / n_rows
                rect = QRect(int(sw_x0), int(sw_y0), int(sw_x1 - sw_x0), int(sw_y1 - sw_y0))
                swatch_rects.append(rect)
                
                item = QGraphicsRectItem(rect)
                item.setBrush(QColor(color))
                item.setPen(QColor(50, 150, 50, 220))
                scene.addItem(item)
                
                # Draw swatch index label (1-based, row-wise)
                label_idx = row * n_cols + col + 1
                label = QGraphicsTextItem(str(label_idx))
                label.setDefaultTextColor(QColor(0, 120, 0))
                label.setPos(rect.left() + 6, rect.top() + 4)
                scene.addItem(label)
                
        return swatch_rects  # list of QRect for each swatch, row-wise