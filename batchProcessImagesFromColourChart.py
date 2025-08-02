#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# To install dependencies, run:
#   pip install colour-checker-detection psutil pillow

import os
import time

import cv2
import numpy as np
import rawpy
import imageio
import colour
import psutil
import tempfile
from PIL import Image
from colour_checker_detection import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    detect_colour_checkers_segmentation)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QListWidgetItem,
    QGraphicsScene, QGraphicsPixmapItem, QProgressBar, QGraphicsTextItem, QLabel, QRubberBand, QGraphicsEllipseItem,
    QPushButton, QRadioButton, QButtonGroup, QHBoxLayout, QSizePolicy, QWidget, QGraphicsRectItem
)
from PySide6.QtCore import (
    QRunnable, QThreadPool, Signal, QObject, QTimer, Qt, QSettings, QEvent, QRect, QSize
)
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter, QIcon
from sympy.codegen.ast import continue_

from scanspaceImageProcessor_UI import Ui_MainWindow

# Code to get taskbar icon visible
import ctypes
scanSpaceImageProcessor = u'mycompany.myproduct.subproduct.version' # arbitrary string
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(scanSpaceImageProcessor)

RAW_EXTENSIONS = ('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')

class WorkerSignals(QObject):
    log = Signal(str)
    preview = Signal(object)
    status = Signal(str, str, float, str)

class ClickableLabel(QLabel):
    clicked = Signal(int)  # Pass the index in the imagesListWidget

    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.index = index

    def mouseReleaseEvent(self, event):
        self.clicked.emit(self.index)
        super().mouseReleaseEvent(event)

# Class for our text in our preview window to be at screen resolution, not image resolution
class FixedSizeText(QGraphicsTextItem):
    def __init__(self, text, x, y, color=QColor('red'), parent=None):
        super().__init__(text, parent)
        self.setPos(x, y)
        self.setDefaultTextColor(color)
        self.setFlag(QGraphicsTextItem.ItemIgnoresTransformations, True)

# Class for our dots in our preview window to be at screen resolution, not image resolution
class FixedSizeEllipse(QGraphicsEllipseItem):
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

class ImageCorrectionWorker(QRunnable):
    def __init__(self, images, swatches, output_folder, signals, jpeg_quality,
                 rename_map=None, name_base='', padding=0, export_masked=False):
        super().__init__()
        self.images = images
        self.swatches = swatches
        self.output_folder = output_folder
        self.signals = signals
        self.jpeg_quality = jpeg_quality
        # File renaming support
        self.rename_map = rename_map or {}
        self.name_base = name_base
        self.padding = padding
        self.export_masked = export_masked
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def run(self):
        os.makedirs(self.output_folder, exist_ok=True)
        D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
        reference = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
        ref_swatches = colour.XYZ_to_RGB(
            colour.xyY_to_XYZ(list(reference.data.values())),
            reference.illuminant, D65,
            colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
        )
        for img_path in self.images:
            local_timer_start = time.time()

            # Build output path
            out_fn = None
            if getattr(self, "use_original_filenames", False):
                # Use original filename with .jpg extension
                out_fn = os.path.splitext(os.path.basename(img_path))[0] + '.jpg'
            else:
                seq = self.rename_map.get(img_path)
                if seq is not None and self.name_base:
                    out_fn = f"{self.name_base}_{seq:0{self.padding}d}.jpg"
                else:
                    out_fn = os.path.splitext(os.path.basename(img_path))[0] + '.jpg'
            out_path = os.path.join(self.output_folder, out_fn)

            # send start signal
            self.signals.status.emit(img_path, 'started', (time.time() - local_timer_start), out_path)
            self.signals.log.emit(f"[Worker] Starting {img_path}")

            try:
                with rawpy.imread(img_path) as raw:
                    rgb = raw.postprocess(
                        output_bps=16,
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=True,
                        output_color=rawpy.ColorSpace.sRGB
                    )
                    img_arr = np.array(rgb, dtype=np.float32) / 65535.0
            except Exception as e:
                self.signals.log.emit(f"[Image Load Error] {img_path}: {e}")
                self.signals.status.emit(img_path, 'error')
                continue

            # Check cancel before processing
            if self.cancelled:
                self.images = []  # clear the image list
                self.signals.log.emit("[Worker] Cancelled by user during processing. Exiting thread.")
                self.signals.status.emit(img_path, 'cancelled')
                return

            
            try:
                corrected = colour.colour_correction(img_arr, self.swatches, ref_swatches)
                multiplier = 1.0
                if hasattr(self, "image_metadata_map"):
                    meta = self.image_metadata_map.get(img_path)
                    if meta is not None:
                        value = meta.get('average_exposure')
                        try:
                            multiplier = float(value)
                        except (TypeError, ValueError):
                            print("no multiplier found")
                            multiplier = 1.0
                img_arr = corrected * multiplier

                corrected = np.clip(img_arr, 0, 1)

                # Export the masked images if flag is set

                if getattr(self, "export_masked", False):
                    lum = 0.2126 * corrected[:, :, 0] + 0.7152 * corrected[:, :, 1] + 0.0722 * corrected[:, :, 2]
                    shadow_limit = getattr(self, "shadow_limit", 0.05)
                    highlight_limit = getattr(self, "highlight_limit", 0.98)
                    shadow_mask = lum <= shadow_limit
                    highlight_mask = lum >= highlight_limit
                    mask = shadow_mask | highlight_mask

                    # Find connected clusters in mask
                    mask_uint8 = mask.astype(np.uint8)
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_uint8, connectivity=8)
                    min_area = 200  # <-- set your minimum area threshold (in pixels)
                    clean_mask = np.zeros_like(mask, dtype=bool)

                    for label in range(1, num_labels):  # 0 is background
                        area = stats[label, cv2.CC_STAT_AREA]
                        if area >= min_area:
                            clean_mask[labels == label] = True

                    corrected[clean_mask] = 0
                    self.signals.log.emit(
                        f"[Export Masked] Masked {np.sum(clean_mask)} pixels (min area={min_area}) "
                        f"with limits shadow<={shadow_limit:.2f}, highlight>={highlight_limit:.2f}"
                    )

                corrected_uint8 = np.uint8(255 * colour.cctf_encoding(corrected))
                imageio.imwrite(out_path, corrected_uint8, quality=self.jpeg_quality)

                self.signals.log.emit(f"[Saved] {out_path}")
                # emit an array directly for preview
                data = [corrected_uint8, out_path]
                self.signals.preview.emit(data)
                self.signals.status.emit(img_path, 'finished', (time.time() - local_timer_start), out_path)

            except Exception as e:
                self.signals.log.emit(f"[Processing Error] {img_path}: {e}")
                self.signals.status.emit(img_path, 'error')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.threadpool = QThreadPool()
        self.settings = QSettings('scanSpace', 'ImageProcessor')
        icon_path = ("./resources/scanSpaceLogo_256px.ico")

        if os.path.exists(icon_path):
            pixmap = QPixmap(str(icon_path))
            app_icon = QIcon(pixmap)
            self.setWindowIcon(app_icon)

        else:
            print(f"Icon file not found at: {icon_path}")

        # Metadata format is: Raw Image Path (Path), Processed Output Path (Path), Processing Status (str), Calibration File (Path), Datatype (Str), Colour Chart (bool)
        # Metadata is stored per-image in the imageListWidget.

        # Restore last-used paths
        rawf = self.settings.value('rawFolder', '')
        outf = self.settings.value('outputFolder', '')
        if rawf:
            self.ui.rawImagesDirectoryLineEdit.setText(rawf)
        if outf:
            self.ui.outputDirectoryLineEdit.setText(outf)

        # Preview scene
        self.previewScene = QGraphicsScene(self)
        self.ui.imagePreviewGraphicsView.setScene(self.previewScene)
        self.ui.imagePreviewGraphicsView.setRenderHint(QPainter.SmoothPixmapTransform)

        # Hide the detect chart controls by default
        self.ui.detectChartToolshelfFrame.setVisible(False)
        # Disable flatten button until a crop region is selected
        self.ui.flattenChartImagePushButton.setEnabled(False)
        self.ui.finalizeChartPushbutton.setEnabled(False)

        # Hide chart debug tools
        self.setup_debug_views()
        self.ui.colourChartDebugToolsFrame.setVisible(False)

        # Calibration metadata path
        self.calibration_file = None
        self.chart_image = None
        self.chart_swatches = None
        self.correct_thumbnails = False
        self.flatten_swatch_rects = None
        self.temp_swatches = []

        # Modes
        self.manual_selection_mode = False
        self.flatten_mode = False
        self.corner_points = []
        self.showing_chart_preview = False
        self.cropped_fp = None
        self.original_preview_pixmap = None
        self.fp_image_array = None

        # thumbnail cache
        self.thumbnail_cache = {}

        # thumnail preview controls
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        # Instruction label
        self.instruction_label = None

        # profiling image processing
        self.total_images = 0
        self.finished_images = 0
        self.global_start = None
        self.processing_active = False
        self.active_workers = []

        # CPU / Memory bars
        for bar in (self.ui.cpuUsageProgressBar, self.ui.memoryUsageProgressBar):
            bar.setRange(0, 100)
            bar.setFormat(bar.objectName().replace('ProgressBar', ' Usage: %p%'))
        self.cpuTimer = QTimer(self)
        self.cpuTimer.timeout.connect(self.update_system_usage)
        self.cpuTimer.start(1000)
        self.ui.cpuUsageProgressBar.setFormat("CPU Usage: %p%")
        self.ui.memoryUsageProgressBar.setFormat("Memory Usage: %p%")

        self.memoryWarningLabel = QLabel('', self)
        self.statusBar().addPermanentWidget(self.memoryWarningLabel)

        # Manual selection RubberBand
        self.rubberBand = QRubberBand(QRubberBand.Rectangle, self.ui.imagePreviewGraphicsView.viewport())
        self.manual_selection_mode = False
        self.showing_chart_preview = False
        self.ui.showOriginalImagePushbutton.setEnabled(False)
        self.ui.detectChartShelfPushbutton.setEnabled(False)

        # Connect signals
        self.ui.browseForChartPushbutton.clicked.connect(self.browse_chart)
        self.ui.browseForImagesPushbutton.clicked.connect(self.browse_images)
        self.ui.browseoutputDirectoryPushbutton.clicked.connect(self.browse_output_directory)
        self.ui.setSelectedAsChartPushbutton.clicked.connect(self.set_selected_as_chart)
        self.ui.processImagesPushbutton.clicked.connect(self.process_images_button_clicked)
        self.ui.imagesListWidget.itemSelectionChanged.connect(self.preview_selected)
        self.ui.previewChartPushbutton.clicked.connect(self.detect_chart)
        self.ui.manuallySelectChartPushbutton.clicked.connect(self.manually_select_chart)
        self.ui.detectChartShelfPushbutton.clicked.connect(lambda: self.detect_chart(input_source=self.ui.chartPathLineEdit.text(), is_npy=False))
        self.ui.revertImagePushbutton.clicked.connect(self.revert_image)
        self.ui.showOriginalImagePushbutton.clicked.connect(self.toggle_chart_preview)
        self.ui.flattenChartImagePushButton.clicked.connect(self.flatten_chart_image)
        self.ui.finalizeChartPushbutton.clicked.connect(self.finalize_manual_chart_selection)
        self.ui.calculateAverageExposurePushbutton.clicked.connect(self.calculate_average_exposure)
        self.ui.removeAverageDataPushbutton.clicked.connect(self.remove_average_exposure_data)
        self.ui.displayDebugExposureDataCheckBox.toggled.connect(
            lambda checked: checked and self.show_exposure_debug_overlay()
        )
        self.ui.highlightLimitSpinBox.valueChanged.connect(
            lambda: self.show_exposure_debug_overlay() if self.ui.displayDebugExposureDataCheckBox.isChecked() else None
        )
        self.ui.shadowLimitSpinBox.valueChanged.connect(
            lambda: self.show_exposure_debug_overlay() if self.ui.displayDebugExposureDataCheckBox.isChecked() else None
        )
        self.ui.nextImagePushbutton.clicked.connect(self.select_next_image)
        self.ui.previousImagePushbutton.clicked.connect(self.select_previous_image)
        self.ui.imagesListWidget.currentRowChanged.connect(self.update_thumbnail_strip)

        # Event filters
        self.ui.thumbnailPreviewFrame.installEventFilter(self)
        # self.ui.thumbnailPreviewDisplayFrame_holder.installEventFilter(self)
        self.ui.imagePreviewGraphicsView.viewport().installEventFilter(self)

        self.current_image_pixmap = None


    def log(self, msg):
        self.ui.logOutputTextEdit.append(msg)

    def process_images_button_clicked(self):
        if not self.processing_active:
            # Start processing
            self.processing_active = True
            self.ui.processImagesPushbutton.setText("Cancel Processing")
            self.ui.processImagesPushbutton.setStyleSheet("background-color: #f77; color: black;")
            self.start_processing()  # Your current method to launch the workers
        else:
            # Cancel
            self.cancel_processing()

    def browse_chart(self):
        default = self.ui.chartPathLineEdit.text() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Chart Image', default,
            filter='RAW Files (*.nef *.cr2 *.cr3 *.dng *.arw *.raw)'
        )
        if path:
            self.ui.chartPathLineEdit.setText(path)
            self.log(f"[Browse] Chart set to {path}")

    def reset_all_state(self):
        """Clear all cached/cached/selection data and UI for a new session."""
        # Clear all metadata
        self.thumbnail_cache.clear()
        self.calibration_file = None
        self.chart_image = None
        self.chart_swatches = None
        self.temp_swatches = []
        self.flatten_swatch_rects = None
        self.cropped_fp = None
        self.original_preview_pixmap = None
        self.current_image_pixmap = None
        self.fp_image_array = None
        self.corrected_preview_pixmap = None
        self.manual_selection_mode = False
        self.flatten_mode = False
        self.corner_points = []
        self.showing_chart_preview = False

        # Clear preview, chart path, and instruction label
        self.previewScene.clear()
        self.ui.chartPathLineEdit.clear()
        if self.instruction_label:
            self.instruction_label.hide()
            self.instruction_label = None

        # Clear debug tools and disable advanced UI actions
        self.ui.colourChartDebugToolsFrame.setVisible(False)
        self.ui.detectChartToolshelfFrame.setVisible(False)
        self.ui.flattenChartImagePushButton.setEnabled(False)
        self.ui.finalizeChartPushbutton.setEnabled(False)
        self.ui.showOriginalImagePushbutton.setEnabled(False)
        self.ui.detectChartShelfPushbutton.setEnabled(False)
        self.ui.finalizeChartPushbutton.setStyleSheet('')

        # Remove all thumbnail widgets in the holder
        holder = self.ui.thumbnailPreviewDisplayFrame_holder
        for child in holder.findChildren(QWidget):
            child.setParent(None)
            child.deleteLater()

        # Clear image list widget
        self.ui.imagesListWidget.clear()

        # Optionally clear the log window (comment out if not desired)
        # self.ui.logOutputTextEdit.clear()

    def browse_images(self):
        self.reset_all_state()
        default = self.ui.rawImagesDirectoryLineEdit.text() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, 'Select Raw Image Directory', default)
        if not folder:
            return

        self.ui.rawImagesDirectoryLineEdit.setText(folder)
        self.settings.setValue('rawFolder', folder)
        self.log(f"[Browse] Raw folder set to {folder}")
        self.ui.imagesListWidget.clear()

        for fname in sorted(os.listdir(folder)):
            if not fname.lower().endswith(RAW_EXTENSIONS):
                continue
            input_path = os.path.join(folder, fname)
            metadata = {
                'input_path': input_path,
                'output_path': None,
                'status': 'raw',
                'calibration': None,
                'data_type': None,
                'chart': False,
                'processing_time': None,
                'debug_images': {},
            }

            item = QListWidgetItem(fname)
            item.setData(Qt.UserRole, metadata)
            self.ui.imagesListWidget.addItem(item)

            # Pre-generate and cache thumbnail
            arr = self.load_thumbnail_array(input_path, max_size=(500, 500))  # Wide but height-limited
            if arr is not None:
                arr_uint8 = (arr * 255).astype(np.uint8)
                h, w, c = arr_uint8.shape
                img = QImage(arr_uint8.data, w, h, w * c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                self.thumbnail_cache[input_path] = {'pixmap': pixmap, 'array': arr}
                self.log("[Browse] Thumbnail image added to cache")
            else:
                self.thumbnail_cache[input_path] = None

        if self.ui.imagesListWidget.count() > 0:
            self.ui.imagesListWidget.setCurrentRow(0)

    def browse_output_directory(self):
        default = self.ui.outputDirectoryLineEdit.text() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Directory', default)
        if folder:
            self.ui.outputDirectoryLineEdit.setText(folder)
            self.settings.setValue('outputFolder', folder)
            self.log(f"[Browse] Output folder set to {folder}")

    def set_selected_as_chart(self):
        # Clear previous chart assignments
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            meta['chart'] = False
            meta.pop('debug_images', None)
            item.setData(Qt.UserRole, meta)
            item.setBackground(QColor('white'))

        selected_item = self.ui.imagesListWidget.currentItem()
        meta = selected_item.data(Qt.UserRole)
        meta['chart'] = True
        chart_path = meta['input_path']
        self.ui.chartPathLineEdit.setText(chart_path)
        self.log(f"[Select] Chart set to {chart_path}")

        self.detect_chart(input_source=chart_path, is_npy=False)

    def preview_selected(self):
        item = self.ui.imagesListWidget.currentItem()
        if not item:
            self.show_debug_frame(False)
            return

        meta = item.data(Qt.UserRole)
        path_to_show = meta.get('output_path') or meta['input_path']
        if path_to_show:
            self.preview_thumbnail(path_to_show)
            self.log(f"[Preview] Showing image: {path_to_show}")

        if meta.get('chart'):
            self.corrected_preview_pixmap = self.pixmap_from_array(meta['debug_images']['corrected_image'])
            self.show_debug_frame(True)
        else:
            self.show_debug_frame(False)

        if self.ui.displayDebugExposureDataCheckBox.isChecked():
            self.show_exposure_debug_overlay()

    def preview_thumbnail(self, path):
        exists = os.path.exists(path)
        self.log(f"[Debug] exists: {exists}")
        real = os.path.realpath(path)
        self.log(f"[Debug] realpath: {real}")
        ext = os.path.splitext(path)[1].lower()
        pixmap = None

        if ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
            img = QImage(path.replace('\\', '/'))
            if not img.isNull():
                pixmap = QPixmap.fromImage(img)
            else:
                self.log("[Preview] QImage failed for non-RAW file, trying PIL fallback")
                try:
                    pil = Image.open(path).convert("RGBA")
                    data = pil.tobytes("raw", "RGBA")
                    w, h = pil.size
                    img2 = QImage(data, w, h, QImage.Format_RGBA8888)
                    pixmap = QPixmap.fromImage(img2)
                    self.log("[Preview] PIL fallback successful")
                except Exception as e:
                    self.log(f"[Preview Error] PIL fallback failed: {e}")

        # For RAW files, use thumbnail loader
        elif ext in RAW_EXTENSIONS:
            try:
                arr = self.load_thumbnail_array(path, max_size=(512, 512))
                if arr is not None:
                    arr_uint8 = (arr * 255).astype(np.uint8)
                    h, w, c = arr_uint8.shape
                    img2 = QImage(arr_uint8.data, w, h, w * c, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(img2)
                    self.log("[Preview] RAW thumbnail loaded successfully")
                else:
                    raise Exception("load_thumbnail_array returned None")
            except Exception as e:
                self.log(f"[Preview Error] RAW load failed: {e}")

        if not pixmap or pixmap.isNull():
            self.log("[Preview Error] could not load thumbnail or image")
            return

        # Display in the preview scene
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(
            self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio
        )
        self.current_image_pixmap = pixmap
        self.log(f"[Preview] Displayed preview for {path}")

    def start_processing(self):
        self.active_workers.clear()
        inf = self.ui.rawImagesDirectoryLineEdit.text().strip()
        outf = self.ui.outputDirectoryLineEdit.text().strip() or os.getcwd()
        thr = self.ui.imageProcessingThreadsSpinbox.value()
        qual = self.ui.jpegQualitySpinbox.value()

        export_masked = self.ui.exportMaskedImagesCheckBox.isChecked()
        use_original_filenames = self.ui.useOriginalFilenamesCheckBox.isChecked()

        # Decide whether to load from .npy or re‐extract
        if self.calibration_file and os.path.exists(self.calibration_file):
            self.log(f"[Process] Loading swatches from {self.calibration_file}")
            swatches = np.load(self.calibration_file, allow_pickle=True)
            if swatches is None:
                self.log("❌ Failed to load swatches from file")
                return
        else:
            chart = self.ui.chartPathLineEdit.text().strip()
            if not (chart and inf and outf):
                self.log("❗ Please fill all paths (chart, raw folder, output).")
                return
            self.log("[Process] Extracting swatches from chart")
            if self.chart_swatches is not None:
                swatches, self.calibration_file = self.extract_chart_swatches(chart)
                if swatches is None:
                    self.log("❌ No chart detected; aborting.")
                    return

        # Gather only RAW images (don’t append chart path here)
        images = [
            os.path.join(inf, f)
            for f in os.listdir(inf)
            if f.lower().endswith(RAW_EXTENSIONS)
        ]
        if not images:
            self.log("❗ No RAW images found in input folder.")
            return

        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            metadata = self.ui.imagesListWidget.item(i).data(Qt.UserRole)
            if metadata:
                metadata['calibration'] = self.calibration_file
                item.setData(Qt.UserRole, metadata)


        self.total_images = len(images)
        self.finished_images = 0
        self.global_start = time.time()
        self.log(f"[Process] Dispatching {self.total_images} images across {thr} Parallel Threads.")

        # Dispatch workers
        size = max(1, len(images) // thr)
        rename_map = {path: idx + 1 for idx, path in enumerate(images)}
        name_base = self.ui.newImageNameLineEdit.text().strip()
        padding = self.ui.imagePaddingSpinBox.value()

        for i in range(0, len(images), size):
            chunk = images[i: i + size]
            sig = WorkerSignals()
            sig.log.connect(self.log)
            sig.preview.connect(self._from_worker_preview)
            sig.status.connect(self.update_image_status)
            image_metadata_map = {
                item.data(Qt.UserRole)['input_path']: item.data(Qt.UserRole)
                for i in range(self.ui.imagesListWidget.count())
                for item in [self.ui.imagesListWidget.item(i)]
            }
            worker = ImageCorrectionWorker(
                chunk, swatches, outf, sig, qual, rename_map,
                name_base=name_base, padding=padding,
                export_masked=export_masked
            )
            shadow_limit = self.ui.shadowLimitSpinBox.value() / 100.0
            highlight_limit = self.ui.highlightLimitSpinBox.value() / 100.0
            worker.shadow_limit = shadow_limit
            worker.highlight_limit = highlight_limit
            worker.use_original_filenames = use_original_filenames
            worker.image_metadata_map = image_metadata_map
            self.active_workers.append(worker)
            self.threadpool.start(worker)

    def cancel_processing(self):
        self.threadpool.clear()
        for worker in self.active_workers:
            if hasattr(worker, "cancel"):
                worker.cancel()
        self.active_workers.clear()
        self.processing_active = False
        self.ui.processImagesPushbutton.setText("Process Images")
        self.ui.processImagesPushbutton.setStyleSheet("")  # Revert to default
        self.log("[Processing] All processing cancelled by user.")

    def update_image_status(self, image_path, status, processing_time, output_path):
        """
        Slot to update the background color of the corresponding QListWidgetItem
        based on the processing status for each image.
        """
        # Iterate over list items to find the matching filename
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            metadata = item.data(Qt.UserRole)
            if metadata['input_path'] == image_path:
                metadata['status'] = status
                metadata['processing_time'] = processing_time
                metadata['output_path'] = output_path
                # Choose color by status
                if status == 'started':
                    bg = QColor('#FFA500')  # light orange
                elif status == 'finished':
                    bg = QColor('#4caf50')  # green
                elif status == 'error':
                    bg = QColor('#f44336')  # red
                else:
                    bg = QColor('white')
                item.setBackground(bg)
                item.setData(Qt.UserRole, metadata)
                break

        if status in ('finished', 'error'):
            self.finished_images += 1
            if self.finished_images >= self.total_images and self.global_start:
                total = time.time() - self.global_start
                self.log(f"[Timing] {self.total_images} images processed in {total:.2f}s")
                self.processing_complete()

    def processing_complete(self):
        self.processing_active = False
        self.ui.processImagesPushbutton.setText("Process Images")
        self.ui.processImagesPushbutton.setStyleSheet("")
        self.log("[Processing] All processing complete.")

    def _from_worker_preview(self, data):
        # Handle array or path
        if isinstance(data, (list, tuple)) and len(data) == 2:
            image_data, image_path = data
        else:
            image_data = data
            image_path = None

        if image_path is not None:
            for i in range(self.ui.imagesListWidget.count()):
                item = self.ui.imagesListWidget.item(i)
                meta = item.data(Qt.UserRole)
                if meta.get('input_path') == image_path or meta.get('output_path') == image_path:
                    self.ui.imagesListWidget.setCurrentRow(i)
                    break

        if isinstance(image_data, np.ndarray):
            # Wrap numpy array (H×W×3 uint8) directly into QImage
            h, w, c = image_data.shape
            bytes_per_line = c * w
            try:
                img = QImage(image_data.data, w, h, bytes_per_line, QImage.Format_RGB888)
                if img.isNull():
                    self.log("[Preview Error] QImage from array is null")
                    return
                pix = QPixmap.fromImage(img)
                if pix.isNull():
                    self.log("[Preview Error] QPixmap.fromImage is null")
                    return
                self._display_preview(pix)
            except Exception as e:
                self.log(f"[Preview Error] QImage creation failed: {e}")
        else:
            self.show_preview(image_path)

    def show_preview(self, image_path):
        path = os.path.normpath(image_path)
        self.log(f"[Preview] Loading processed image: {path}")
        exists = os.path.exists(path)
        self.log(f"[Debug] exists: {exists}")

        try:
            pil_image = Image.open(image_path).convert("RGBA")
            data = pil_image.tobytes("raw", "RGBA")
            width, height = pil_image.size

            img = QImage(data, width, height, QImage.Format_RGBA8888)
        except Exception as e:
            print(f"PIL fallback failed: {e}")
            return

        pixmap = QPixmap.fromImage(img)
        if pixmap.isNull():
            print("QPixmap conversion failed after fallback")
            return

        pw, ph = pixmap.width(), pixmap.height()
        vw = self.ui.imagePreviewGraphicsView.viewport().width()
        vh = self.ui.imagePreviewGraphicsView.viewport().height()
        self.log(f"[Debug] Pixmap size: {pw}x{ph}, Viewport: {vw}x{vh}")
        self._display_preview(pixmap)
        self.log("[Preview] Displayed processed image")

    def _display_preview(self, pixmap):
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(
            self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)


    def update_system_usage(self):
        cpu = psutil.cpu_percent(None)
        self.ui.cpuUsageProgressBar.setValue(cpu)
        # show text on progress bar

        self.ui.cpuUsageProgressBar.setTextVisible(True)
        self.ui.cpuUsageProgressBar.setStyleSheet(self.get_usage_style(cpu))
        mem = psutil.virtual_memory().percent
        self.ui.memoryUsageProgressBar.setValue(mem)
        # show text on progress bar
        self.ui.memoryUsageProgressBar.setTextVisible(True)
        self.ui.memoryUsageProgressBar.setStyleSheet(self.get_usage_style(mem))

        # show or clear high-memory warning
        if mem > 95:
            self.memoryWarningLabel.setText(
                "Memory usage too high, Lowering Image Threads may produce faster results"
            )
        else:
            self.memoryWarningLabel.setText('')

    def get_usage_style(self, pct):
        if pct < 50:
            c = "#4caf50"
        elif pct < 80:
            c = "#ffc107"
        else:
            c = "#f44336"
        return f"QProgressBar{{border:1px solid #bbb;border-radius:5px;text-align:center}}QProgressBar::chunk{{background-color:{c};width:1px}}"

    def extract_chart_swatches(self, chart_path):
        try:
            with rawpy.imread(chart_path) as raw:
                rgb = raw.postprocess(
                    output_bps=16,
                    gamma=(1,1),
                    no_auto_bright=True,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB
                )
                img = np.array(rgb, dtype=np.float32) / 65535.0
        except Exception as e:
            self.log(f"[Chart Load Error] {e}")
            return None, None
        for result in detect_colour_checkers_segmentation(img, additional_data=True):
            swatches = result.swatch_colours
            if isinstance(swatches, np.ndarray) and swatches.shape == (24, 3):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                np.save(tmp_file.name, np.array(swatches))
                return swatches, tmp_file.name
        self.log("[Swatches] No valid colour chart swatches detected.")
        return None, None

    def finalize_manual_chart_selection(self):
        if not self.manual_selection_mode:
            self.log("finalized chart")
        else:
            self.log("exiting manual chart selection")
            self.ui.manuallySelectChartPushbutton.setStyleSheet('')
        self.ui.detectChartToolshelfFrame.setVisible(False)
        self.ui.showOriginalImagePushbutton.setEnabled(False)
        self.ui.detectChartShelfPushbutton.setEnabled(False)
        self.ui.flattenChartImagePushButton.setEnabled(False)
        self.ui.finalizeChartPushbutton.setEnabled(False)
        self.ui.finalizeChartPushbutton.setStyleSheet('')
        self.instruction_label.hide()
        self.ui.detectChartToolshelfFrame.setVisible(False)
        self.show_debug_frame(False)
        self.chart_swatches = self.temp_swatches


    def manually_select_chart(self):
        if self.manual_selection_mode:
            self.finalize_manual_chart_selection()
            return

        path = self.ui.chartPathLineEdit.text().strip()
        if not path:
            self.log('[Manual] No chart selected')
            return
        self.log(f'[Manual] Loading chart for manual selection: {path}')

        if self.instruction_label:
            self.instruction_label.setText(
                "Click and drag box around the colour chart"
            )
            self.instruction_label.show()
        else:
            self.instruction_label = QLabel('Click and drag box around the colour chart')
            self.instruction_label.setAlignment(Qt.AlignCenter)
            self.ui.verticalLayout_4.insertWidget(
                self.ui.verticalLayout_4.indexOf(self.ui.imagePreviewGraphicsView),
                self.instruction_label
            )

        layout = self.ui.detectChartToolshelfFrame
        active = not layout.isVisible()
        layout.setVisible(active)

        # Light green highlight for active mode
        css = "background-color: #A5D6A7"
        self.ui.manuallySelectChartPushbutton.setStyleSheet(css if active else "")

        # Reset flatten button until a new crop is drawn
        self.ui.flattenChartImagePushButton.setEnabled(False)
        self.ui.flattenChartImagePushButton.setStyleSheet("")

        try:
            with rawpy.imread(path) as raw:
                # Create a viewable proxy image
                rgb_full = raw.postprocess(
                    output_bps=16, gamma=(1,1), no_auto_bright=True,
                    use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB
                )
                # Create a full bit depth image to be edited in the background
                self.fp_image_array = np.array(rgb_full, dtype=np.float32) / 65535.0
                thumb = raw.postprocess(
                    output_bps=8, gamma=(1,1), no_auto_bright=True,
                    use_camera_wb=True, output_color=rawpy.ColorSpace.sRGB
                )
                thumb_arr = np.array(thumb, dtype=np.uint8)
        except Exception as e:
            self.log(f'[Manual] Raw load error: {e}')
            return
        h, w, _ = thumb_arr.shape
        bytes_per_line = w * 3
        qimg = QImage(thumb_arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.original_preview_pixmap = pixmap
        self.current_image_pixmap = pixmap
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.manual_selection_mode = True
        self.flatten_mode = False
        self.ui.showOriginalImagePushbutton.setEnabled(False)
        self.ui.detectChartShelfPushbutton.setEnabled(False)

    def flatten_chart_image(self):
        # must have a cropped image first
        if not hasattr(self, 'cropped_preview_pixmap'):
            self.log('[Flatten] No cropped image, select region first')
            return

        self.log('[Flatten] Select the 4 corners of the chart')
        self.instruction_label.setText('Select the 4 corners of the chart')
        css = "background-color: #A5D6A7"
        self.ui.flattenChartImagePushButton.setStyleSheet(css)

        # 1) reset any old state
        self.flatten_mode = True
        self.corner_points = []

        # 2) clear out any previous dots/labels
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(self.cropped_preview_pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(
            self.previewScene.itemsBoundingRect(),
            Qt.KeepAspectRatio
        )

    def eventFilter(self, source, event):
        PADDING = 600
        if source == self.ui.imagePreviewGraphicsView.viewport():
            # Manual selection (initial crop)
            if self.manual_selection_mode and not self.flatten_mode:
                if event.type() == QEvent.MouseButtonPress:
                    self.origin = event.position().toPoint()
                    self.rubberBand.setGeometry(QRect(self.origin, QSize()))
                    self.rubberBand.show()
                    return True
                elif event.type() == QEvent.MouseMove and self.rubberBand.isVisible():
                    self.rubberBand.setGeometry(
                        QRect(self.origin, event.position().toPoint()).normalized()
                    )
                    return True
                elif event.type() == QEvent.MouseButtonRelease and self.rubberBand.isVisible():
                    rect = self.rubberBand.geometry()
                    tl = self.ui.imagePreviewGraphicsView.mapToScene(rect.topLeft())
                    br = self.ui.imagePreviewGraphicsView.mapToScene(rect.bottomRight())
                    x, y = int(tl.x()), int(tl.y())
                    w, h = abs(int(br.x() - tl.x())), abs(int(br.y() - tl.y()))
                    self.manual_selection_mode = False
                    self.rubberBand.hide()

                    # Crop without padding
                    self.cropped_preview_pixmap = self.current_image_pixmap.copy(x, y, w, h)
                    self.cropped_fp = self.fp_image_array[y:y + h, x:x + w, :]

                    self.previewScene.clear()
                    self.previewScene.addItem(QGraphicsPixmapItem(self.cropped_preview_pixmap))
                    self.ui.imagePreviewGraphicsView.resetTransform()
                    self.ui.imagePreviewGraphicsView.fitInView(
                        self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio
                    )
                    self.ui.showOriginalImagePushbutton.setEnabled(True)
                    self.ui.detectChartShelfPushbutton.setEnabled(True)
                    self.ui.flattenChartImagePushButton.setEnabled(True)
                    return True

            # Flatten mode (perspective selection)
            if self.flatten_mode and event.type() == QEvent.MouseButtonPress:
                pt = event.position().toPoint()
                sp = self.ui.imagePreviewGraphicsView.mapToScene(pt)
                idx = len(self.corner_points) + 1
                self.corner_points.append((sp.x(), sp.y()))
                self.log(f"[Flatten] Point {idx}: ({int(sp.x())}, {int(sp.y())})")

                # Draw red dot and label
                dot = FixedSizeEllipse(sp.x(), sp.y(), radius=8, color=QColor('red'))
                self.previewScene.addItem(dot)
                label = FixedSizeText(str(idx), sp.x() + 12, sp.y() - 8, color=QColor('red'))
                self.previewScene.addItem(label)

                if idx == 4:
                    pts_src = np.array(self.corner_points, dtype=np.float32)
                    width = np.linalg.norm(pts_src[1] - pts_src[0])
                    height = int(width * 9.0 / 14.0)
                    dst_pts = np.array([
                        [PADDING, PADDING],
                        [PADDING + width, PADDING],
                        [PADDING + width, PADDING + height],
                        [PADDING, PADDING + height]
                    ], dtype=np.float32)

                    # Extract full cropped image array with correct stride
                    qimg = self.cropped_preview_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
                    h0 = qimg.height()
                    stride = qimg.bytesPerLine()
                    buf = bytes(qimg.constBits())[: h0 * stride]
                    arr2d = np.frombuffer(buf, dtype=np.uint8).reshape((h0, stride))
                    arr = arr2d[:, : qimg.width() * 3].reshape((h0, qimg.width(), 3))

                    # Pad full array
                    arr_padded = cv2.copyMakeBorder(
                        arr, PADDING, PADDING, PADDING, PADDING,
                        borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
                    )
                    pts_src_padded = pts_src + PADDING

                    M = cv2.getPerspectiveTransform(pts_src_padded, dst_pts)
                    out_w = int(width + 2 * PADDING)
                    out_h = int(height + 2 * PADDING)
                    warped = cv2.warpPerspective(arr_padded, M, (out_w, out_h))

                    # Show warped result
                    q2 = QImage(warped.data, out_w, out_h, out_w * 3, QImage.Format_RGB888)
                    pix2 = QPixmap.fromImage(q2)
                    self.previewScene.clear()
                    img_item = QGraphicsPixmapItem(pix2)
                    self.previewScene.addItem(img_item)

                    # Draw swatch grid overlay on top of the warped image
                    swatch_grid_rect = QRect(PADDING, PADDING, int(width), int(height))
                    self.flatten_swatch_rects = self.draw_colorchecker_swatch_grid(
                        self.previewScene, swatch_grid_rect, n_cols=6, n_rows=4
                    )

                    self.ui.imagePreviewGraphicsView.resetTransform()
                    self.ui.imagePreviewGraphicsView.fitInView(
                        self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio
                    )

                    # Flatten FP and store back into cropped_fp
                    fparr = cv2.copyMakeBorder(
                        self.cropped_fp,
                        PADDING, PADDING, PADDING, PADDING,
                        borderType=cv2.BORDER_REFLECT
                    )
                    warped_fp = cv2.warpPerspective(
                        fparr.astype(np.float32),
                        M,
                        (out_w, out_h)
                    )
                    self.cropped_fp = warped_fp

                    self.log("[Flatten] Chart and FP image flattened")
                    self.flatten_mode = False
                    self.instruction_label.setText(
                        "Please Run Detect Chart or Revert image to select new region"
                    )
                    self.ui.finalizeChartPushbutton.setEnabled(True)
                    self.preview_manual_swatch_correction()
                return True

        # Event to update thumbnail strip
        if source == self.ui.thumbnailPreviewFrame and event.type() == QEvent.Resize:
            self.update_thumbnail_strip()
            return True

        # Event to update the image preview size
        if source == self.ui.imagePreviewGraphicsView.viewport() and event.type() == QEvent.Resize:
            # Re-fit the view to the scene contents (the image)
            self.ui.imagePreviewGraphicsView.fitInView(
                self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)
            return True

        return super().eventFilter(source, event)

    def preview_manual_swatch_correction(self):
        """
        Extracts mean colours from swatch rectangles in self.flatten_swatch_rects,
        applies manual color correction, and displays the corrected image preview.
        Assumes self.cropped_fp is the current floating-point chart image,
        and self.flatten_swatch_rects is a list of 24 QRect objects.
        """
        import numpy as np
        import colour

        if not (hasattr(self, 'flatten_swatch_rects') and self.flatten_swatch_rects and len(
                self.flatten_swatch_rects) == 24):
            self.log("[Manual Swatch] Swatch grid not found or incomplete.")
            return
        if not hasattr(self, 'cropped_fp') or self.cropped_fp is None:
            self.log("[Manual Swatch] No image to process.")
            return

        img_fp = self.cropped_fp
        swatch_colours = []
        for rect in self.flatten_swatch_rects:
            x0, y0, w, h = rect.left(), rect.top(), rect.width(), rect.height()
            region = img_fp[int(y0):int(y0 + h), int(x0):int(x0 + w), :]
            mean_color = region.mean(axis=(0, 1))
            swatch_colours.append(mean_color)
        swatch_colours = np.array(swatch_colours)
        self.temp_swatches = swatch_colours
        self.log(f"[Manual Swatch] Extracted {len(swatch_colours)} mean swatch colours.")

        # Reference chart (D65, ColorChecker 24)
        D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
        REFERENCE_COLOUR_CHECKER = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
        ref_swatches = colour.XYZ_to_RGB(
            colour.xyY_to_XYZ(list(REFERENCE_COLOUR_CHECKER.data.values())),
            REFERENCE_COLOUR_CHECKER.illuminant, D65,
            colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
        )
        corrected = colour.colour_correction(img_fp, swatch_colours, ref_swatches)
        corrected = np.clip(corrected, 0, 1)
        corrected_uint8 = np.uint8(255 * colour.cctf_encoding(corrected))

        # Display result
        pixmap = self.pixmap_from_array(corrected_uint8)
        self._display_preview(pixmap)
        self.log("[Manual Swatch] Manual correction preview displayed.")


    def detect_chart(self, input_source=None, is_npy=False):
        """
        Detects the color checker chart using either manual grid (if available)
        or auto-detection as fallback.
        """

        detected = False  # <-- Always define this up front!

        if input_source is not None:
            img_fp = np.load(input_source) if is_npy else input_source
        else:
            if hasattr(self, 'cropped_fp') and self.cropped_fp is not None:
                img_fp = self.cropped_fp
            else:
                self.log("[Detect Chart] No image available for detection.")
                return

        # RAW file check
        if isinstance(img_fp, str) and img_fp.lower().endswith(
                ('.arw', '.nef', '.cr2', '.cr3', '.dng', '.rw2', '.raw')):
            try:
                with rawpy.imread(img_fp) as raw:
                    img_rgb = raw.postprocess(
                        output_bps=16,
                        gamma=(1, 1),
                        no_auto_bright=True,
                        use_camera_wb=True,
                        output_color=rawpy.ColorSpace.sRGB
                    )
                img_fp = np.array(img_rgb, dtype=np.float32) / 65535.0
                self.log("[Detect Chart] Loaded RAW file to RGB for detection.")
            except Exception as e:
                self.log(f"[Detect Chart] RAW decode failed: {e}")
                return

        selected_item = self.ui.imagesListWidget.currentItem()
        if selected_item:
            meta = selected_item.data(Qt.UserRole)
            selected_item.setData(Qt.UserRole, meta)
            selected_item.setBackground(QColor('#ADD8E6'))

        results = detect_colour_checkers_segmentation(img_fp, additional_data=True)

        for colour_checker_data in results:
            swatch_colours, swatch_masks, colour_checker_image = colour_checker_data.values

            # Save calibration data
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            np.save(tmp_file.name, swatch_colours)
            self.calibration_file = tmp_file.name
            self.log(f'[Detect] Calibration saved to {self.calibration_file}')
            self.chart_swatches = swatch_colours

            # Generate Corrected Image
            D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
            reference = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
            ref_swatches = colour.XYZ_to_RGB(
                colour.xyY_to_XYZ(list(reference.data.values())),
                reference.illuminant, D65,
                colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
            )
            corrected = colour.colour_correction(img_fp, swatch_colours, ref_swatches)
            corrected = np.clip(corrected, 0, 1)
            corrected_uint8 = np.uint8(255 * colour.cctf_encoding(corrected))

            # Swatch Overlay
            masks_overlay = np.zeros_like(colour_checker_image)
            for mask in swatch_masks:
                masks_overlay[mask[0]:mask[1], mask[2]:mask[3], :] = 0.25
            swatch_overlay = np.clip(colour_checker_image + masks_overlay, 0, 1)
            swatch_overlay_uint8 = np.uint8(255 * colour.cctf_encoding(swatch_overlay))

            # Detection Debug (Segmented Image)
            detection_debug_uint8 = np.uint8(255 * colour.cctf_encoding(colour_checker_image))

            # Swatches and Clusters (Swatches contours)
            swatches_clusters_img = np.uint8(255 * colour.cctf_encoding(colour_checker_image.copy()))
            for mask in swatch_masks:
                start_row, end_row, start_col, end_col = mask
                cv2.rectangle(
                    swatches_clusters_img,
                    (start_col, start_row),
                    (end_col, end_row),
                    color=(255, 0, 255),
                    thickness=3
                )
            # Store all debug images together
            debug_images = {
                'corrected_image': corrected_uint8,
                'swatch_overlay': swatch_overlay_uint8,
                'detection_debug': detection_debug_uint8,
                'swatches_and_clusters': swatches_clusters_img,
            }

            # Update metadata of currently selected item
            selected_item = self.ui.imagesListWidget.currentItem()
            if selected_item:
                meta = selected_item.data(Qt.UserRole)
                meta['debug_images'] = debug_images
                meta['chart'] = True
                selected_item.setData(Qt.UserRole, meta)
                selected_item.setBackground(QColor('#ADD8E6'))

            # Immediately display corrected preview
            self.corrected_preview_pixmap = self.pixmap_from_array(corrected_uint8)
            self._display_preview(self.corrected_preview_pixmap)

            detected = True

            if is_npy:
                self.ui.finalizeChartPushbutton.setEnabled(True)
                self.instruction_label.setText("Please Finalize your chart")

            break  # Only process first detected checker

        if detected:
            self.log("[Detect] Chart detected successfully. Debug images generated.")
            self.show_debug_frame(True)
        else:
            self.log("[Detect] No valid chart detected.")
            self.show_debug_frame(False)

    def toggle_chart_preview(self):
        if not hasattr(self, 'cropped_preview_pixmap'):
            return
        if self.showing_chart_preview:
            pix = self.cropped_preview_pixmap
            self.ui.showOriginalImagePushbutton.setText('Show Chart Preview')
            self.showing_chart_preview = False
        else:
            pix = self.original_preview_pixmap
            self.ui.showOriginalImagePushbutton.setText('Show Chart Image')
            self.showing_chart_preview = True
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pix))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)

    def revert_image(self):
        if hasattr(self, 'original_preview_pixmap'):
            self.previewScene.clear()
            self.previewScene.addItem(QGraphicsPixmapItem(self.original_preview_pixmap))
            self.ui.imagePreviewGraphicsView.resetTransform()
            self.ui.imagePreviewGraphicsView.fitInView(self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.showing_chart_preview = False
            # Re-enable manual selection for new region
            self.manual_selection_mode = True
            self.ui.showOriginalImagePushbutton.setEnabled(False)
            self.ui.detectChartShelfPushbutton.setEnabled(False)

    def show_debug_frame(self, visible):
        self.ui.colourChartDebugToolsFrame.setVisible(visible)

    def setup_debug_views(self):
        self.ui.correctedImageRadioButton.toggled.connect(lambda checked: checked and self.corrected_image_view())
        self.ui.swatchOverlayRadioButton.toggled.connect(lambda checked: checked and self.swatch_overlay_view())
        self.ui.swatchAndClusterRadioButton.toggled.connect(
            lambda checked: checked and self.swatches_and_clusters_view())
        self.ui.detectionDebugRadioButton.toggled.connect(lambda checked: checked and self.detection_debug_view())

    def swatches_and_clusters_view(self):
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('swatches_and_clusters')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            self.log("[Debug View] Swatches and Clusters shown.")
        else:
            self.log("[Debug View] Swatches and Clusters unavailable.")

    def corrected_image_view(self):
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('corrected_image')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            self.log("[Debug View] Corrected image shown.")
        else:
            self.log("[Debug View] Corrected image unavailable.")

    def swatch_overlay_view(self):
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('swatch_overlay')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            self.log("[Debug View] Swatch overlay shown.")
        else:
            self.log("[Debug View] Swatch overlay unavailable.")

    def detection_debug_view(self):
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log("[Debug View] No item selected.")
            return

        meta = item.data(Qt.UserRole)
        debug_images = meta.get('debug_images', {})

        img = debug_images.get('detection_debug')
        if img is not None:
            pixmap = self.pixmap_from_array(img)
            self._display_preview(pixmap)
            self.log("[Debug View] Detection debug shown.")
        else:
            self.log("[Debug View] Detection debug unavailable.")

    def pixmap_from_array(self, array):
        h, w, c = array.shape
        bytes_per_line = c * w
        img = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    def calculate_average_exposure(self):
        """
        Compute and store exposure normalization multipliers for all images.
        Uses chart image as reference if available, otherwise uses average.
        Ignores top 2% (hot spots) and bottom 5% (black areas) of pixel luminance.
        """
        brightness_list = []
        item_to_brightness = {}
        chart_brightness = None

        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            img_path = meta.get('input_path')
            arr = self.load_thumbnail_array(img_path)
            if arr is None:
                self.log(f"[Exposure Calc] Failed on {img_path}: could not load thumbnail")
                continue
            shadow_threshold = self.ui.shadowLimitSpinBox.value() / 100.0
            highlight_threshold = self.ui.highlightLimitSpinBox.value() / 100.0

            lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]

            mask = (lum > shadow_threshold) & (lum < highlight_threshold)

            valid = lum[mask]
            mean_brightness = valid.mean() if valid.size > 0 else lum.mean()
            brightness_list.append(mean_brightness)
            item_to_brightness[img_path] = mean_brightness
            if meta.get('chart'):
                chart_brightness = mean_brightness  # Only one chart expected, take the first found

        if not brightness_list:
            self.log("[Exposure Calc] No valid images for exposure normalization.")
            return

        # Reference: chart if present, otherwise mean of all
        reference_brightness = chart_brightness if chart_brightness is not None else np.mean(brightness_list)
        if chart_brightness is not None:
            self.log(f"[Exposure Calc] Using chart image as reference ({reference_brightness:.3f})")
        else:
            self.log(f"[Exposure Calc] Using average image brightness as reference ({reference_brightness:.3f})")

        # Set exposure multiplier for each image
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            img_path = meta.get('input_path')
            img_brightness = item_to_brightness.get(img_path)
            if img_brightness is None:
                continue
            multiplier = reference_brightness / img_brightness if img_brightness > 0 else 1.0
            meta['average_exposure'] = multiplier
            item.setData(Qt.UserRole, meta)
            self.log(f"[Exposure Calc] {meta.get('input_path')} multiplier: {multiplier:.3f}")

        self.log("[Exposure Calc] Average exposure multipliers calculated and stored.")

    def load_thumbnail_array(self, path, max_size=(512, 512)):
        """
        Loads an image and creates a small thumbnail as a NumPy RGB array, for fast preview or stats.
        Supports fallback for RAW files using rawpy if PIL fails.
        Returns arr (H, W, 3) float32, values in 0..1, or None on failure.
        """
        cache = self.thumbnail_cache.get(path)
        if cache and 'array' in cache and cache['array'] is not None:
            return cache['array']

        ext = os.path.splitext(path)[1].lower()
        try:
            with Image.open(path) as img:
                img.thumbnail(max_size)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                self.log("[Thumb] Thumbnail loaded from image without further processing.")
                if self.chart_swatches and self.correct_thumbnails:
                    try:
                        D65 = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
                        reference = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
                        ref_swatches = colour.XYZ_to_RGB(
                            colour.xyY_to_XYZ(list(reference.data.values())),
                            reference.illuminant, D65,
                            colour.RGB_COLOURSPACES['sRGB'].matrix_XYZ_to_RGB
                        )
                        arr = np.clip(colour.colour_correction(arr, self.chart_swatches, ref_swatches), 0, 1)
                        self.log("[Thumb] Applied colour correction to thumbnail.")
                    except Exception as e:
                        self.log(f"[Thumb] Colour correction on thumbnail failed: {e}")
                return arr
        except Exception as e:
            # RAW fallback for ARW, NEF, etc
            if ext in ('.nef', '.cr2', '.cr3', '.arw', '.dng', '.raw'):
                try:
                    import rawpy
                    with rawpy.imread(path) as raw:
                        try:
                            thumb = raw.extract_thumb()
                            if thumb.format == rawpy.ThumbFormat.JPEG:
                                import io
                                pil = Image.open(io.BytesIO(thumb.data))
                                pil.thumbnail(max_size)
                                arr = np.asarray(pil, dtype=np.float32) / 255.0
                                self.log("[Thumb] Loaded embedded JPEG thumbnail from RAW.")
                                return arr
                            else:
                                self.log("[Thumb] No embedded JPEG, using raw postprocess.")
                        except Exception:
                            self.log("[Thumb] No embedded thumbnail, using raw postprocess.")

                        rgb = raw.postprocess(output_bps=8, no_auto_bright=True)
                        arr = rgb.astype(np.float32) / 255.0
                        # Downsample with cv2 if needed
                        h, w, _ = arr.shape
                        import cv2
                        scale = min(max_size[0] / h, max_size[1] / w, 1.0)
                        if scale < 1.0:
                            arr = cv2.resize(arr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
                        self.log("[Thumb] Loaded thumbnail from rawpy postprocess.")
                        return arr
                except Exception as e2:
                    self.log(f"[Thumb] RAW fallback failed: {e2}")
            self.log(f"[Thumb] Could not load thumbnail: {e}")
        return None

    def remove_average_exposure_data(self):
        """
        Remove the exposure normalization multipliers from all images.
        """
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            if 'average_exposure' in meta:
                del meta['average_exposure']
                item.setData(Qt.UserRole, meta)
        self.log("[Exposure Calc] All exposure multipliers removed.")

    def show_exposure_debug_overlay(self):
        """
        Show an overlay for hot (highlight) and dark (shadow) areas using absolute luminance thresholds.
        Hot spots are red, dark areas are blue.
        """
        item = self.ui.imagesListWidget.currentItem()
        if item is None:
            self.log("[Exposure Debug] No image selected.")
            return

        meta = item.data(Qt.UserRole)
        img_path = meta.get('input_path')
        arr = self.load_thumbnail_array(img_path, max_size=(800, 800))
        if arr is None:
            self.log(f"[Exposure Debug] Could not load image: {img_path}")
            return

        # Absolute thresholds from spinboxes (expected range: 0–100)
        shadow_threshold = self.ui.shadowLimitSpinBox.value() / 100.0
        highlight_threshold = self.ui.highlightLimitSpinBox.value() / 100.0

        lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        mask_dark = lum <= shadow_threshold
        mask_hot = lum >= highlight_threshold

        overlay = (arr * 255).astype(np.uint8)
        overlay[mask_hot] = [255, 60, 60]  # Red for hot spots
        overlay[mask_dark] = [60, 60, 255]  # Blue for shadows

        alpha = 0.4
        blend = (arr * (1 - alpha) + (overlay / 255.0) * alpha)
        blend = np.clip(blend * 255, 0, 255).astype(np.uint8)

        h, w, c = blend.shape
        img = QImage(blend.data, w, h, w * c, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(img)
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)
        self.current_image_pixmap = pixmap
        self.log(
            f"[Exposure Debug] Overlay shown (Highlight: ≥{highlight_threshold:.2f}, Shadow: ≤{shadow_threshold:.2f})"
        )

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Right:
            self.select_next_image()
            event.accept()
        elif event.key() == Qt.Key_Left:
            self.select_previous_image()
            event.accept()
        else:
            super().keyPressEvent(event)

    def select_next_image(self):
        idx = self.ui.imagesListWidget.currentRow()
        if idx < self.ui.imagesListWidget.count() - 1:
            self.ui.imagesListWidget.setCurrentRow(idx + 1)

    def select_previous_image(self):
        idx = self.ui.imagesListWidget.currentRow()
        if idx > 0:
            self.ui.imagesListWidget.setCurrentRow(idx - 1)

    def select_image_from_thumbnail(self, idx):
        self.ui.imagesListWidget.setCurrentRow(idx)

    def update_thumbnail_strip(self):
        holder = self.ui.thumbnailPreviewDisplayFrame_holder

        # Remove ALL previous child widgets (thumbnails)
        for child in holder.findChildren(QWidget):
            child.setParent(None)
            child.deleteLater()

        count = self.ui.imagesListWidget.count()
        if count == 0:
            return

        sel_idx = self.ui.imagesListWidget.currentRow()
        frame_width = holder.width()

        # Estimate thumbnail widths
        default_aspect = 1.5
        default_width = int(default_aspect * 60)
        widths = []
        for i in range(count):
            meta = self.ui.imagesListWidget.item(i).data(Qt.UserRole)
            cache = self.thumbnail_cache.get(meta['input_path'])
            pixmap = cache['pixmap'] if cache and 'pixmap' in cache else None
            if pixmap is not None and not pixmap.isNull():
                aspect = pixmap.width() / pixmap.height()
                widths.append(int(aspect * 60))
            else:
                widths.append(default_width)

        # Calculate how many thumbnails fit: sum widths outwards from sel_idx until you run out of space
        thumb_indices = [sel_idx]
        total_width = widths[sel_idx] if count > 0 else 0
        left, right = sel_idx - 1, sel_idx + 1
        while (left >= 0 or right < count):
            add_left = (left >= 0)
            add_right = (right < count)
            if add_left and (not add_right or (len(thumb_indices) % 2 == 1)):
                candidate_width = total_width + widths[left]
                if candidate_width > frame_width and len(thumb_indices) >= 3:
                    break
                thumb_indices.insert(0, left)
                total_width += widths[left]
                left -= 1
            elif add_right:
                candidate_width = total_width + widths[right]
                if candidate_width > frame_width and len(thumb_indices) >= 3:
                    break
                thumb_indices.append(right)
                total_width += widths[right]
                right += 1
            else:
                break

        display_items = [self.ui.imagesListWidget.item(i) for i in thumb_indices]
        display_widths = [widths[i] for i in thumb_indices]
        if total_width > 0 and total_width < frame_width:
            # Proportionally scale up to fit frame width
            scale = frame_width / total_width
            display_widths = [int(w * scale) for w in display_widths]

        # Add the thumbnails to the holder layout
        x_offset = 0
        for i, item in enumerate(display_items):
            meta = item.data(Qt.UserRole)
            cache = self.thumbnail_cache.get(meta['input_path'])
            pixmap = cache['pixmap'] if cache and 'pixmap' in cache else None
            real_index = thumb_indices[i]
            label = ClickableLabel(real_index, holder)
            if pixmap is not None and not pixmap.isNull():
                thumb = pixmap.scaled(display_widths[i], 60, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                label.setPixmap(thumb)
                label.setFixedSize(display_widths[i], 60)
            else:
                label.setText("No preview")
                label.setFixedSize(display_widths[i], 60)
            if real_index == sel_idx:
                label.setStyleSheet("border: 2px solid #2196F3;")
            else:
                label.setStyleSheet("border: 1px solid #999;")
            label.clicked.connect(self.select_image_from_thumbnail)
            label.move(x_offset, 0)
            label.show()
            x_offset += display_widths[i]

    def draw_colorchecker_swatch_grid(self, scene, image_rect, n_cols=6, n_rows=4, color=QColor(0, 255, 0, 90)):
        """Draws a 6x4 swatch grid (for ColorChecker) over the image_rect on the provided QGraphicsScene."""
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




if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
