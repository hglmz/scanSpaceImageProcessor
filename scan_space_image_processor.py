#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# To install dependencies, run:
#   pip install colour-checker-detection psutil pillow
import functools
import os
import shutil
import subprocess
import time

import cv2
import numpy as np
import rawpy
import imageio
import colour
import psutil
import tempfile
from OpenImageIO import ImageBuf, ImageBufAlgo, ImageSpec, ImageOutput, TypeFloat, ColorConfig, ROI, ImageInput, \
    TypeDesc
from PIL import Image
from colour.models import RGB_COLOURSPACES
from colour_checker_detection import (
    SETTINGS_SEGMENTATION_COLORCHECKER_CLASSIC,
    detect_colour_checkers_segmentation)

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QListWidgetItem,
    QGraphicsScene, QGraphicsPixmapItem, QProgressBar, QGraphicsTextItem, QLabel, QRubberBand, QGraphicsEllipseItem,
    QPushButton, QRadioButton, QButtonGroup, QHBoxLayout, QSizePolicy, QWidget, QGraphicsRectItem, QMessageBox,
    QGraphicsView
)
from PySide6.QtCore import (
    QRunnable, QThreadPool, Signal, QObject, QTimer, Qt, QSettings, QEvent, QRect, QSize, QEventLoop
)
from PySide6.QtGui import QColor, QPixmap, QImage, QPainter, QIcon
from sympy.codegen.ast import continue_
from tifffile import tifffile

from scanspaceImageProcessor_UI import Ui_MainWindow

# Code to get taskbar icon visible
import ctypes
scanSpaceImageProcessor = u'mycompany.myproduct.subproduct.version' # arbitrary string to trick windows
ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(scanSpaceImageProcessor)

# supported file formats
RAW_EXTENSIONS = ('.nef', '.cr2', '.cr3', '.dng', '.arw', '.raw')
OUTPUT_FORMATS = ('.jpg', '.png', '.tiff', '.exr')

reference = colour.CCS_COLOURCHECKERS['ColorChecker24 - After November 2014']
illuminant_XYZ = colour.xy_to_XYZ(reference.illuminant)

reference_swatches = colour.XYZ_to_RGB(
    colour.xyY_to_XYZ(list(reference.data.values())),
    RGB_COLOURSPACES['sRGB'],
    illuminant=reference.illuminant,           # a 2-tuple (x, y)
    chromatic_adaptation_transform="CAT02",
    apply_cctf_encoding=False,
)

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

class SwatchPreviewSignals(QObject):
    """Worker signals for previewing manual swatch correction."""
    finished = Signal(np.ndarray)       # Emits the corrected uint8 array
    error    = Signal(str)              # Emits an error message on failure

class SwatchPreviewWorker(QRunnable):
    """
    Runs colour correction + CCTF encoding in a background thread.
    Emits the final uint8 image array when done.
    """
    def __init__(self, img_fp, swatch_colours, reference_swatches):
        super().__init__()
        self.img_fp            = img_fp
        self.swatch_colours    = swatch_colours
        self.reference_swatches= reference_swatches
        self.signals           = SwatchPreviewSignals()

    def run(self):
        try:
            # 1) Colour‐correct
            corrected = colour.colour_correction(
                self.img_fp,
                self.swatch_colours,
                self.reference_swatches
            )
            corrected = np.clip(corrected, 0, 1)

            # 2) Apply CCTF & convert to uint8
            corrected_uint8 = np.uint8(255 * colour.cctf_encoding(corrected))

            # 3) Emit the raw array back to the main thread
            self.signals.finished.emit(corrected_uint8)

        except Exception as e:
            self.signals.error.emit(str(e))


class RawLoadSignals(QObject):
    loaded = Signal(np.ndarray)
    error  = Signal(str)


class RawLoadWorker(QRunnable):
    """
    Loads a RAW file in a background thread and emits only the full-precision
    float32 image array.
    """
    def __init__(self, path: str):
        super().__init__()
        self.path    = path
        self.signals = RawLoadSignals()

    def run(self):
        try:
            with rawpy.imread(self.path) as raw:
                common = dict(
                    gamma=(1,1),
                    no_auto_bright=True,
                    use_camera_wb=True,
                    output_color=rawpy.ColorSpace.sRGB
                )
                # Only full-precision pipeline
                rgb_full = raw.postprocess(output_bps=16, **common)
                full_fp  = np.array(rgb_full, dtype=np.float32) / 65535.0

            # Emit just the float32 array
            self.signals.loaded.emit(full_fp)

        except Exception as e:
            self.signals.error.emit(str(e))


def exit_manual_mode(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # self.log(f"[exit_manual_mode] ENTRY: {func.__name__}, manual_selection_mode={self.manual_selection_mode}")
        if self.manual_selection_mode:
            self.finalize_manual_chart_selection(canceled=True)
        result = func(self, *args, **kwargs)
        # self.log(f"[exit_manual_mode]  EXIT: {func.__name__}")
        return result
    return wrapper

class ImageCorrectionWorker(QRunnable):
    def __init__(self, images, swatches, output_folder, signals, jpeg_quality,
                 rename_map=None, name_base='', padding=0, export_masked=False,
                 output_format: str = '.jpg', tiff_bitdepth=8, exr_colorspace: str | None = None):
        super().__init__()
        self.images = images
        self.swatches = swatches
        self.output_folder = output_folder
        self.signals = signals
        self.jpeg_quality = jpeg_quality
        self.output_format = output_format
        self.tiff_bitdepth = tiff_bitdepth
        self.exr_colorspace = exr_colorspace
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
        ext = self.output_format

        for img_path in self.images:
            local_timer_start = time.time()

            # Build output path
            out_fn = None
            if getattr(self, "use_original_filenames", False):
                # Use original filename with .jpg extension
                out_fn = os.path.splitext(os.path.basename(img_path))[0] + self.output_format
            else:
                seq = self.rename_map.get(img_path)
                if seq is not None and self.name_base:
                    out_fn = f"{self.name_base}_{seq:0{self.padding}d}{self.output_format}"
                else:
                    out_fn = os.path.splitext(os.path.basename(img_path))[0] + self.output_format
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
                self.signals.status.emit(
                    img_path,
                    'error',
                    time.time() - local_timer_start,
                    out_path
                )
                continue

            # Check cancel before processing
            if self.cancelled:
                self.signals.log.emit("[Worker] Cancelled by user. Exiting thread.")
                self.signals.status.emit(
                    img_path,
                    'cancelled',
                    time.time() - local_timer_start,
                    out_path
                )
                return

            
            try:
                corrected = colour.colour_correction(img_arr, self.swatches, reference_swatches)

                # Average brightness correction if flag is set
                multiplier = 1.0
                if hasattr(self, "image_metadata_map"):
                    meta = self.image_metadata_map.get(img_path)
                    if meta is not None:
                        value = meta.get('average_exposure')
                        try:
                            multiplier = float(value)
                        except (TypeError, ValueError):
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

                if ext in ('.png', '.jpg', '.jpeg'):
                    # Always 8-bit
                    corrected_uint8 = np.uint8(255 * colour.cctf_encoding(corrected))
                    imageio.imwrite(out_path, corrected_uint8, quality=self.jpeg_quality)

                elif ext == '.tiff':
                    float16 = colour.cctf_encoding(corrected)
                    if self.tiff_bitdepth == 16:
                        image16 = np.uint16(np.clip(float16 * 65535.0, 0, 65535))
                        with tifffile.TiffWriter(out_path) as tiff_writer:
                            tiff_writer.write(image16)
                        corrected_uint8 = np.uint8(np.clip(float16 * 255.0, 0, 255))
                    else:
                        image8 = np.uint8(np.clip(float16 * 255.0, 0, 255))
                        with tifffile.TiffWriter(out_path) as tiff_writer:
                            tiff_writer.write(image8)
                        corrected_uint8 = image8


                elif ext == '.exr':
                    # Write a temp EXR tagged as sRGB
                    h, w, c = corrected.shape
                    fd, temp_exr = tempfile.mkstemp(suffix=".exr")
                    os.close(fd)
                    spec = ImageSpec(w, h, 3, TypeFloat)
                    spec.attribute("oiio:ColorSpace", "sRGB")
                    out = ImageOutput.create(temp_exr)
                    out.open(temp_exr, spec)
                    out.write_image(corrected.flatten())
                    out.close()
                    print(f"[EXR DEBUG] temp EXR written at {temp_exr}", flush=True)

                    # Load it via ImageBuf & colorconvert
                    buf_in = ImageBuf(temp_exr)
                    buf_out = ImageBufAlgo.colorconvert(buf_in, "sRGB", self.exr_colorspace)
                    print(f"[EXR DEBUG] colorconvert done → {self.exr_colorspace}", flush=True)

                    # Write the final EXR in the target space
                    buf_out.write(out_path)
                    print(f"[EXR DEBUG] final EXR written at {out_path}", flush=True)

                    corrected_uint8 = np.uint8(np.clip(corrected * 255.0, 0, 255))
                    # Remove the temp file
                    try:
                        os.remove(temp_exr)
                    except OSError:
                        pass

                self.signals.log.emit(f"[Saved] {out_path}")

                in_img = ImageInput.open(img_path)
                if in_img:
                    src_pvs = in_img.spec().extra_attribs  # ParamValueList of all tags
                    in_img.close()
                else:
                    src_pvs = []

                out_reader = ImageInput.open(out_path)
                if out_reader:
                    out_spec = out_reader.spec()
                    out_reader.close()
                else:
                    if ext == '.exr' or ext == '.tiff':
                        h, w, c = corrected.shape
                        out_spec = ImageSpec(w, h, c, TypeFloat)
                    else:
                        h, w = corrected_uint8.shape[:2]
                        channels = corrected_uint8.shape[2] if corrected_uint8.ndim == 3 else 1
                        out_spec = ImageSpec(w, h, channels, TypeDesc.UINT8)

                for pv in src_pvs:
                    name = pv.name
                    val = pv.value
                    if isinstance(val, (int, float, str)):
                        out_spec.attribute(name, val)
                    else:
                        out_spec.attribute(name, str(val))
                print(f"[Metadata] Injected {len(src_pvs)} tags into spec for {out_path}", flush=True)

                writer = ImageOutput.create(out_path)
                if not writer:
                    print(f"[Metadata ERROR] Could not open writer for {out_path}", flush=True)
                else:
                    writer.open(out_path, out_spec)
                    if ext == '.exr':
                        pixels = buf_out.get_pixels(TypeFloat)
                    else:
                        pixels = corrected_uint8.flatten()
                    writer.write_image(pixels)
                    writer.close()
                    print(f"[Metadata] Wrote {out_path} with embedded metadata", flush=True)

                data = [corrected_uint8, out_path]
                self.signals.preview.emit(data)
                self.signals.status.emit(img_path, 'finished', (time.time() - local_timer_start), out_path)


            except Exception as e:
                self.signals.log.emit(f"[Processing Error] {img_path}: {e}")
                self.signals.status.emit(
                    img_path,
                    'error',
                    time.time() - local_timer_start,
                    out_path
                )

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # ────────────────────────────────────────────────────────────
        # 1) UI Setup
        # ────────────────────────────────────────────────────────────
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Thread pool for background tasks
        self.threadpool = QThreadPool()

        # ────────────────────────────────────────────────────────────
        # 2) Application Icon & Settings
        # ────────────────────────────────────────────────────────────
        self.settings = QSettings('scanSpace', 'ImageProcessor')
        icon_path = "./resources/scanSpaceLogo_256px.ico"
        if os.path.exists(icon_path):
            pixmap = QPixmap(icon_path)
            self.setWindowIcon(QIcon(pixmap))
        else:
            print(f"Icon file not found at: {icon_path}")

        # Restore last-used folders
        rawf = self.settings.value('rawFolder', '')
        outf = self.settings.value('outputFolder', '')
        if rawf:
            self.ui.rawImagesDirectoryLineEdit.setText(rawf)
        if outf:
            self.ui.outputDirectoryLineEdit.setText(outf)

        # ────────────────────────────────────────────────────────────
        # 3) Preview Scene
        # ────────────────────────────────────────────────────────────
        self.previewScene = QGraphicsScene(self)
        view = self.ui.imagePreviewGraphicsView
        view.setScene(self.previewScene)
        view.setRenderHint(QPainter.SmoothPixmapTransform)

        # ────────────────────────────────────────────────────────────
        # 4) Default UI State
        # ────────────────────────────────────────────────────────────
        # Hide chart tools until needed
        self.ui.detectChartToolshelfFrame.setVisible(False)
        self.ui.colourChartDebugToolsFrame.setVisible(False)  # via setup_debug_views()
        self.ui.processingStatusBarFrame.setVisible(False) # hide our processing status bar

        # Disable buttons until prerequisites are met
        for btn in (
            self.ui.manuallySelectChartPushbutton,
            self.ui.detectChartShelfPushbutton,
            self.ui.showOriginalImagePushbutton,
            self.ui.flattenChartImagePushButton,
            self.ui.finalizeChartPushbutton
        ):
            btn.setEnabled(False)

        # Initialize debug‐view toggles
        self.setup_debug_views()

        # Populate the imageFormatComboBox:
        for fmt in OUTPUT_FORMATS:
            self.ui.imageFormatComboBox.addItem(fmt)

        # Hide bit-depth and JPEG-quality frames until the user picks those formats:
        self.ui.bitDepthFrame.setVisible(False)
        self.ui.jpgQualityFrame.setVisible(False)

        # Whenever the format changes, update which controls are visible:
        self.ui.imageFormatComboBox.currentTextChanged.connect(
            self._update_format_controls
        )

        # Populate the combo with every OIIO colourspace
        config = ColorConfig()
        names = config.getColorSpaceNames()  # returns List[str]
        for cs in names:
            self.ui.exrColourSpaceComboBox.addItem(cs)

        # Whenever the format changes, update which frames are shown:
        self.ui.imageFormatComboBox.currentTextChanged.connect(self._update_format_controls)
        # Initial call
        self._update_format_controls(self.ui.imageFormatComboBox.currentText())

        # ────────────────────────────────────────────────────────────
        # 5) Internal State Variables
        # ────────────────────────────────────────────────────────────
        # Calibration & chart data
        self.calibration_file       = None
        self.chart_image            = None
        self.chart_swatches         = None
        self.temp_swatches          = []
        self.flatten_swatch_rects   = None
        self.average_enabled = False
        self.selected_average_source = None

        # Mode flags
        self.manual_selection_mode  = False
        self.flatten_mode           = False
        self.showing_chart_preview  = False

        # Image buffers
        self.cropped_fp             = None
        self.original_preview_pixmap= None
        self.fp_image_array         = None

        # Thumbnails & metadata
        self.thumbnail_cache        = {}
        self.image_metadata_map     = {}

        # UI helpers
        self.instruction_label      = None
        self.corner_points          = []

        # Profiling counters
        self.total_images           = 0
        self.finished_images        = 0
        self.global_start           = None
        self.processing_active      = False
        self.active_workers         = []

        # ────────────────────────────────────────────────────────────
        # 6) System Usage Meters
        # ────────────────────────────────────────────────────────────
        for bar in (self.ui.cpuUsageProgressBar, self.ui.memoryUsageProgressBar):
            bar.setRange(0, 100)
            bar.setFormat(f"{bar.objectName().replace('ProgressBar','')} Usage: %p%")

        self.ui.cpuUsageProgressBar.setRange(0, 100)
        self.ui.memoryUsageProgressBar.setRange(0, 100)
        self.ui.cpuUsageProgressBar.setFormat("CPU usage: %p%")
        self.ui.memoryUsageProgressBar.setFormat("Memory usage: %p%")
        self.cpuTimer = QTimer(self)
        self.cpuTimer.timeout.connect(self.update_system_usage)
        self.cpuTimer.start(1000)

        # Status‐bar warning label
        self.memoryWarningLabel = QLabel('', self)
        self.statusBar().addPermanentWidget(self.memoryWarningLabel)

        # ────────────────────────────────────────────────────────────
        # 7) Manual‐Selection RubberBand
        # ────────────────────────────────────────────────────────────
        self.rubberBand = QRubberBand(QRubberBand.Rectangle,
                                      self.ui.imagePreviewGraphicsView.viewport())

        # ────────────────────────────────────────────────────────────
        # 8) Signal Connections
        # ────────────────────────────────────────────────────────────
        ui = self.ui  # alias for brevity

        ui.browseForChartPushbutton.clicked.connect(self.browse_chart)
        ui.browseForImagesPushbutton.clicked.connect(self.browse_images)
        ui.browseoutputDirectoryPushbutton.clicked.connect(self.browse_output_directory)

        ui.setSelectedAsChartPushbutton.clicked.connect(self.set_selected_as_chart)
        ui.processImagesPushbutton.clicked.connect(self.process_images_button_clicked)

        ui.imagesListWidget.itemSelectionChanged.connect(self.preview_selected)
        ui.imagesListWidget.currentRowChanged.connect(self.update_thumbnail_strip)

        ui.manuallySelectChartPushbutton.clicked.connect(self.manually_select_chart)
        ui.detectChartShelfPushbutton.clicked.connect(
            lambda: self.detect_chart(input_source=ui.chartPathLineEdit.text(), is_npy=False)
        )
        ui.revertImagePushbutton.clicked.connect(self.revert_image)
        ui.showOriginalImagePushbutton.clicked.connect(self.toggle_chart_preview)
        ui.flattenChartImagePushButton.clicked.connect(self.flatten_chart_image)
        ui.finalizeChartPushbutton.clicked.connect(self.finalize_manual_chart_selection)

        ui.calculateAverageExposurePushbutton.clicked.connect(self.calculate_average_exposure)
        ui.removeAverageDataPushbutton.clicked.connect(self.remove_average_exposure_data)
        ui.displayDebugExposureDataCheckBox.toggled.connect(
            lambda checked: checked and self.show_exposure_debug_overlay()
        )
        ui.highlightLimitSpinBox.valueChanged.connect(
            lambda: self.show_exposure_debug_overlay() if ui.displayDebugExposureDataCheckBox.isChecked() else None
        )
        ui.shadowLimitSpinBox.valueChanged.connect(
            lambda: self.show_exposure_debug_overlay() if ui.displayDebugExposureDataCheckBox.isChecked() else None
        )

        ui.nextImagePushbutton.clicked.connect(self.select_next_image)
        ui.previousImagePushbutton.clicked.connect(self.select_previous_image)
        ui.setSelectedImageAsAveragePushbutton.clicked.connect(self.set_selected_image_as_average_source)

        # ────────────────────────────────────────────────────────────
        # 9) Event Filters & Focus
        # ────────────────────────────────────────────────────────────
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        ui.thumbnailPreviewFrame.installEventFilter(self)
        ui.imagePreviewGraphicsView.viewport().installEventFilter(self)
        ui.thumbnailPreviewFrame.installEventFilter(self)

        # Keep track of the currently displayed pixmap
        self.current_image_pixmap = None

        # print to log that the application is ready
        self.log("Scan Space Image Processor initialized")


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
        """
        Open a file dialog to select a RAW chart file.
        On selection, update the chart‐path line edit and log the choice.
        """
        default = self.ui.chartPathLineEdit.text() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select Chart Image', default,
            filter='RAW Files (*.nef *.cr2 *.cr3 *.dng *.arw *.raw)'
        )
        if path:
            self.ui.chartPathLineEdit.setText(path)
            # self.log(f"[Browse] Chart set to {path}")

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
        self.ui.manuallySelectChartPushbutton.setEnabled(False)

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

    @exit_manual_mode
    def browse_images(self):
        """
        Open a directory dialog to select a folder of RAW images.
        Clears any previous state, populates the image list widget with all supported RAW files,
        and stores the chosen folder in settings.
        """
        self.reset_all_state()
        default = self.ui.rawImagesDirectoryLineEdit.text() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, 'Select Raw Image Directory', default)
        if not folder:
            return

        self.ui.rawImagesDirectoryLineEdit.setText(folder)
        self.settings.setValue('rawFolder', folder)
        # self.log(f"[Browse] Raw folder set to {folder}")
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
                # self.log("[Browse] Thumbnail image added to cache")
            else:
                self.thumbnail_cache[input_path] = None

        if self.ui.imagesListWidget.count() > 0:
            self.ui.imagesListWidget.setCurrentRow(0)

    def browse_output_directory(self):
        """
        Open a directory dialog to select where processed images will be saved.
        Updates the output‐directory line edit and persists the choice in settings.
        """
        default = self.ui.outputDirectoryLineEdit.text() or os.getcwd()
        folder = QFileDialog.getExistingDirectory(self, 'Select Output Directory', default)
        if folder:
            self.ui.outputDirectoryLineEdit.setText(folder)
            self.settings.setValue('outputFolder', folder)
            # self.log(f"[Browse] Output folder set to {folder}")

    def _update_format_controls(self, ext: str):
        """
        Show the bit-depth frame only when TIFF is selected;
        show the JPEG-quality frame only when JPEG is selected.
        """
        fmt = ext.lower()
        # Show bit-depth options for TIFF
        self.ui.bitDepthFrame.setVisible(fmt == '.tiff')
        # Show JPEG-quality options for JPEG
        self.ui.jpgQualityFrame.setVisible(fmt in ('.jpg', '.jpeg'))
        # show exr options
        self.ui.exrOptionsFrame.setVisible(fmt == '.exr')

    def set_selected_as_chart(self):
        """
        Mark the currently selected image in the list as the reference colour chart.
        Clears any prior chart flags, sets the 'chart' metadata on the new selection,
        updates the chart‐path line edit, enables manual selection, and triggers detection.
        """
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
        # self.log(f"[Select] Chart set to {chart_path}")
        self.ui.manuallySelectChartPushbutton.setEnabled(True)

        self.detect_chart(input_source=chart_path, is_npy=False)

    @exit_manual_mode
    def preview_selected(self):
        """
        Show a preview of the currently selected image (either input or processed).
        Toggles debug‐frame visibility if this image is the reference chart,
        and updates exposure debug overlay if enabled.
        """
        item = self.ui.imagesListWidget.currentItem()
        if not item:
            self.show_debug_frame(False)
            return

        meta = item.data(Qt.UserRole)
        path_to_show = meta.get('output_path') or meta['input_path']
        if path_to_show:
            self.preview_thumbnail(path_to_show)
            # self.log(f"[Preview] Showing image: {path_to_show}")

        if meta.get('chart'):
            self.corrected_preview_pixmap = self.pixmap_from_array(meta['debug_images']['corrected_image'])
            self.show_debug_frame(True)
            self.ui.manuallySelectChartPushbutton.setEnabled(True)
        else:
            self.show_debug_frame(False)
            self.ui.manuallySelectChartPushbutton.setEnabled(False)

        if self.ui.displayDebugExposureDataCheckBox.isChecked():
            self.show_exposure_debug_overlay()

    def preview_thumbnail(self, path):
        """
        Load and display a thumbnail for the given file path.
        Uses cached pixmap if available, applies exposure adjustment,
        and supports fallback for common formats and RAW thumbnails.
        """
        # Try cache first
        cache = self.thumbnail_cache.get(path, {})
        pixmap = cache.get('pixmap') if cache else None

        # If not in cache or invalid, load fresh
        if not pixmap or pixmap.isNull():
            ext = os.path.splitext(path)[1].lower()
            if ext in ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'):
                img = QImage(path.replace('\\', '/'))
                if not img.isNull():
                    pixmap = QPixmap.fromImage(img)
            elif ext in RAW_EXTENSIONS:
                try:
                    arr = self.load_thumbnail_array(path, max_size=(512, 512))
                    if arr is not None:
                        arr_uint8 = (arr * 255).astype(np.uint8)
                        h, w, c = arr_uint8.shape
                        img2 = QImage(arr_uint8.data, w, h, w * c, QImage.Format_RGB888)
                        pixmap = QPixmap.fromImage(img2)
                except Exception as e:
                    self.log(f"[Preview Error] RAW load failed: {e}")

            if pixmap and not pixmap.isNull():
                self.thumbnail_cache[path] = {'pixmap': pixmap}
            else:
                self.log("[Preview Error] could not load thumbnail or image")
                return

        # Apply exposure brightness if set
        # Find corresponding list item to get metadata
        for i in range(self.ui.imagesListWidget.count()):
            item = self.ui.imagesListWidget.item(i)
            meta = item.data(Qt.UserRole)
            if meta.get('input_path') == path or meta.get('output_path') == path:
                factor = meta.get('average_exposure', 1.0)
                if factor != 1.0:
                    pixmap = self._adjust_pixmap_brightness(pixmap, factor)
                break

        # Show in preview scene
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(
            self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio
        )
        self.current_image_pixmap = pixmap

    @exit_manual_mode
    def start_processing(self):
        """
        Launch background workers to process every image in the list;
        switches the UI into 'processing' mode and shows the progress frame.
        """
        self.active_workers.clear()
        inf = self.ui.rawImagesDirectoryLineEdit.text().strip()
        outf = self.ui.outputDirectoryLineEdit.text().strip() or os.getcwd()
        thr = self.ui.imageProcessingThreadsSpinbox.value()
        qual = self.ui.jpegQualitySpinbox_2.value()

        export_masked = self.ui.exportMaskedImagesCheckBox.isChecked()
        use_original_filenames = self.ui.useOriginalFilenamesCheckBox.isChecked()
        output_ext = self.ui.imageFormatComboBox.currentText().lower()
        if output_ext == '.tiff':
            tiff_bits = 16 if self.ui.sixteenBitRadioButton.isChecked() else 8
        else:
            tiff_bits = None

        # Decide whether to load from .npy or re‐extract
        if self.calibration_file and os.path.exists(self.calibration_file):
            # self.log(f"[Process] Loading swatches from {self.calibration_file}")
            swatches = np.load(self.calibration_file, allow_pickle=True)
            if swatches is None:
                self.log("❌ Failed to load swatches from file")
                return
        else:
            chart = self.ui.chartPathLineEdit.text().strip()
            if not (chart and inf and outf):
                self.log("❗ Please fill all paths (chart, raw folder, output).")
                return
            # self.log("[Process] Extracting swatches from chart")
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

        # Show and initialize the status bar
        self.ui.processingStatusBarFrame.setVisible(True)
        self.ui.processingStatusProgressBar.setRange(0, self.total_images)
        self.ui.processingStatusProgressBar.setValue(0)
        self.ui.processingStatusProgressBar.setFormat(
            f"0 out of {self.total_images} images processed"
        )

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

            exr_cs = None
            if output_ext == '.exr':
                exr_cs = self.ui.exrColourSpaceComboBox.currentText()

            worker = ImageCorrectionWorker(
                chunk, swatches, outf, sig, qual, rename_map,
                name_base=name_base, padding=padding,
                export_masked=export_masked, output_format=output_ext, tiff_bitdepth=tiff_bits, exr_colorspace=exr_cs,
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
        """
        Signal all active workers to cancel, reset the processing UI,
        and revert the 'Process' button to its original state.
        """
        self.threadpool.clear()
        for worker in self.active_workers:
            if hasattr(worker, "cancel"):
                worker.cancel()
        self.active_workers.clear()
        self.processing_active = False
        self.ui.processImagesPushbutton.setText("Process Images")
        self.ui.processImagesPushbutton.setStyleSheet("")
        self.ui.processingStatusBarFrame.setVisible(False)
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

            # Update status bar
            self.ui.processingStatusProgressBar.setValue(self.finished_images)
            self.ui.processingStatusProgressBar.setFormat(
                f"{self.finished_images} out of {self.total_images} images processed"
            )

            if self.finished_images >= self.total_images and self.global_start:
                total = time.time() - self.global_start
                self.log(f"[Timing] {self.total_images} images processed in {total:.2f}s")
                self.processing_complete()

    def processing_complete(self):
        """
        Called when all images are finished (or cancelled).
        Hides the progress frame, resets the 'Process' button, and logs completion.
        """
        self.processing_active = False
        self.ui.processingStatusBarFrame.setVisible(False)
        self.ui.processImagesPushbutton.setText("Process Images")
        self.ui.processImagesPushbutton.setStyleSheet("")
        self.log("[Processing] All processing complete.")

    def _from_worker_preview(self, data):
        """
        Displays a provided npy image array from the image processing worker
        Called whenever an image finishes processing
        """
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
        """
        Called when when an image is selected and a thumbnail or image object in memory does not exist for the image
        Loads image from disk using its path
        """
        path = os.path.normpath(image_path)
        # self.log(f"[Preview] Loading processed image: {path}")
        exists = os.path.exists(path)
        # self.log(f"[Debug] exists: {exists}")

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

        self._display_preview(pixmap)

    def _display_preview(self, pixmap):
        """
        clears the preview scene and displays the provided pixmap
        """
        self.current_image_pixmap = pixmap
        self.previewScene.clear()
        self.previewScene.addItem(QGraphicsPixmapItem(pixmap))
        self.ui.imagePreviewGraphicsView.resetTransform()
        self.ui.imagePreviewGraphicsView.fitInView(
            self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)


    def update_system_usage(self):
        """
        Updates the sytem usage progress bars with current values based on the QTimer created in __init__
        """
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
        """
        sets the stylesheet for the usage bar
        Uses colours depending on the percentage
        """
        if pct < 50:
            c = "#4caf50"
        elif pct < 80:
            c = "#ffc107"
        else:
            c = "#f44336"
        return f"QProgressBar{{border:1px solid #bbb;border-radius:5px;text-align:center}}QProgressBar::chunk{{background-color:{c};width:1px}}"

    def load_raw_image(self, path: str) -> np.ndarray | None:
        """
        Load a RAW file off the main thread but return synchronously:
          - Schedules a RawLoadWorker(path) on self.threadpool.
          - Spins a QEventLoop until the worker emits loaded(fp_array) or error(msg).
          - Returns the float32 array in [0,1], or None on failure.
        """
        loop = QEventLoop()
        result = {'fp': None}

        # 1) Create the worker
        worker = RawLoadWorker(path)

        # 2) Hook up signals to capture the result and quit the local loop
        def _on_loaded(fp):
            result['fp'] = fp
            loop.quit()

        def _on_error(msg):
            self.log(f"[RAW Load Error] {msg}")
            loop.quit()

        worker.signals.loaded.connect(_on_loaded)
        worker.signals.error.connect(_on_error)

        # 3) Start the worker & wait
        self.threadpool.start(worker)
        loop.exec()  # <- yields to Qt event loop until quit()

        # 4) Return whatever we got
        return result['fp']

    def extract_chart_swatches(self, chart_path):
        """
        Load a RAW or numpy chart file, detect the 24 swatch colours,
        save them to a temporary .npy file, and return (swatches, filename).
        """
        img = self.load_raw_image(chart_path)

        if img is None:
            return None, None

        for result in detect_colour_checkers_segmentation(img, additional_data=True, swatch_area_minimum_area_factor=20):
            swatches = result.swatch_colours
            if isinstance(swatches, np.ndarray) and swatches.shape == (24, 3):
                tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
                np.save(tmp_file.name, np.array(swatches))
                return swatches, tmp_file.name
        self.log("[Swatches] No valid colour chart swatches detected.")
        return None, None

    def finalize_manual_chart_selection(self, *, canceled: bool = False):
        """
        Exit manual chart selection (either commit or cancel) and restore UI.

        Args:
            canceled: if True, we’re aborting manual mode and do NOT commit temp_swatches.
                      if False, this is a real “Finalize Chart” and we copy temp_swatches → chart_swatches.
        """
        # 1) Log & commit (only if finalize, not cancel)
        if canceled:
            self.log("[Manual] Manual chart selection canceled.")
        else:
            self.log("[Manual] Manual chart selection finalized.")
            # commit the temporary swatches/calibration
            if hasattr(self, 'temp_swatches'):
                self.chart_swatches = self.temp_swatches
            if hasattr(self, 'temp_calibration_file'):
                self.calibration_file = self.temp_calibration_file

        # 2) Reset internal modes & data
        self.manual_selection_mode = False
        self.flatten_mode = False
        self.corner_points.clear()
        self.flatten_swatch_rects = None

        # 3) Hide any visible rubber‐band
        if hasattr(self, 'rubberBand') and self.rubberBand.isVisible():
            self.rubberBand.hide()

        # 4) Restore the *original* preview if we have it
        if hasattr(self, 'original_preview_pixmap') and self.original_preview_pixmap:
            self._display_preview(self.original_preview_pixmap)
            self.showing_chart_preview = False
            self.ui.showOriginalImagePushbutton.setText("Show Chart Preview")

        # 5) Disable & reset style of all chart‐tools buttons
        self._set_chart_tools_enabled()

        # 6) Hide the tool‐shelf & instruction overlay
        self.ui.detectChartToolshelfFrame.setVisible(False)
        if hasattr(self, 'instruction_label') and isinstance(self.instruction_label, QLabel):
            self.instruction_label.hide()

        # 7) Hide debug frame if present
        self.show_debug_frame(False)

    def _set_chart_tools_enabled(self, *, manual=False, detect=False, show=False, flatten=False, finalize=False):
        """
        Enable or disable all chart-tools buttons with optional highlight on Manual & Flatten.
        """
        mapping = {
            'manual':   self.ui.manuallySelectChartPushbutton,
            'detect':   self.ui.detectChartShelfPushbutton,
            'show':     self.ui.showOriginalImagePushbutton,
            'flatten':  self.ui.flattenChartImagePushButton,
            'finalize': self.ui.finalizeChartPushbutton,
        }
        for key, btn in mapping.items():
            enabled = locals()[key]
            btn.setEnabled(enabled)
            # highlight Manual & Flatten when active
            if (key == 'manual' and enabled) or (key == 'flatten' and enabled):
                btn.setStyleSheet("background-color: #A5D6A7")
            else:
                btn.setStyleSheet("")

    @exit_manual_mode
    def manually_select_chart(self):
        """
        Enter manual chart selection mode:
          - Load RAW + thumbnail, reset UI, show instructions, enable only Manual-Select.
        """
        path = self.ui.chartPathLineEdit.text().strip()
        if not path:
            self.log('[Manual] No chart selected')
            return
        full_fp = self.load_raw_image(path)
        if full_fp is None:
            self.log(f'[Manual] RAW load failed: {path}')
            return
        self.fp_image_array = full_fp

        # build 8-bit thumbnail
        thumb_arr = np.uint8(255 * np.clip(full_fp, 0.0, 1.0))
        h, w, _ = thumb_arr.shape
        bytes_per_line = w * 3
        qimg = QImage(thumb_arr.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)
        self.original_preview_pixmap = pixmap
        self.current_image_pixmap  = pixmap
        self._display_preview(pixmap)

        # Reset modes and hide overlays/debug
        self.manual_selection_mode = True
        self.flatten_mode          = False
        self.show_debug_frame(False)
        if hasattr(self, 'rubberBand') and self.rubberBand.isVisible():
            self.rubberBand.hide()

        # Reset and enable only the Manual-Select button
        self._set_chart_tools_enabled(manual=True)

        # Show instruction label
        instr = getattr(self, 'instruction_label', None)
        if isinstance(instr, QLabel):
            instr.setText("Click and drag box around the colour chart")
            instr.show()
        else:
            self.instruction_label = QLabel("Click and drag box around the colour chart")
            self.instruction_label.setAlignment(Qt.AlignCenter)
            self.ui.verticalLayout_4.insertWidget(
                self.ui.verticalLayout_4.indexOf(self.ui.imagePreviewGraphicsView),
                self.instruction_label
            )
        # Reveal toolshelf
        self.ui.detectChartToolshelfFrame.setVisible(True)

    def on_manual_crop_complete(self, rect: QRect):
        """
        Called when the QRect is drawn in manual crop mode
        Enables the buttons for Show Original Image, Flatten Chart Mode and Detect Chart
        """

        self.ui.detectChartShelfPushbutton.setEnabled(True)
        self.ui.showOriginalImagePushbutton.setEnabled(True)

        css = "background-color: #A5D6A7"
        self.ui.flattenChartImagePushButton.setEnabled(True)
        self.ui.flattenChartImagePushButton.setStyleSheet(css)

    def flatten_chart_image(self):
        """
        Enter corner‐picking mode to flatten the manually selected chart region;
        on completion, stores the cropped-preview pixmap and enables flattening.
        """
        if not hasattr(self, 'cropped_preview_pixmap'):
            self.log('[Flatten] No cropped image, select region first')
            return

        self.log('[Flatten] Select the 4 corners of the chart')
        self.instruction_label.setText('Select the 4 corners of the chart')
        css = "background-color: #A5D6A7"
        self.ui.flattenChartImagePushButton.setStyleSheet(css)

        self.flatten_mode = True
        self.corner_points = []

        self._display_preview(self.cropped_preview_pixmap)
        # enable only Flatten & Finalize
        self._set_chart_tools_enabled(flatten=True, finalize=True)

    # def eventFilter(self, source, event):
    #     ## TODO: Split event functions into their own functions, use this to handle these calls.
    #     """
    #     Master event handler for various modes.
    #     :param source:
    #     :param event:
    #     """
    #     PADDING = 600
    #     if source == self.ui.imagePreviewGraphicsView.viewport():
    #         # Manual selection (initial crop)
    #         if self.manual_selection_mode and not self.flatten_mode:
    #             if event.type() == QEvent.MouseButtonPress:
    #                 self.origin = event.position().toPoint()
    #                 self.rubberBand.setGeometry(QRect(self.origin, QSize()))
    #                 self.rubberBand.show()
    #                 return True
    #             elif event.type() == QEvent.MouseMove and self.rubberBand.isVisible():
    #                 self.rubberBand.setGeometry(
    #                     QRect(self.origin, event.position().toPoint()).normalized()
    #                 )
    #                 return True
    #             elif event.type() == QEvent.MouseButtonRelease and self.rubberBand.isVisible():
    #                 rect = self.rubberBand.geometry()
    #                 tl = self.ui.imagePreviewGraphicsView.mapToScene(rect.topLeft())
    #                 br = self.ui.imagePreviewGraphicsView.mapToScene(rect.bottomRight())
    #                 x, y = int(tl.x()), int(tl.y())
    #                 w, h = abs(int(br.x() - tl.x())), abs(int(br.y() - tl.y()))
    #                 self.manual_selection_mode = False
    #                 self.rubberBand.hide()
    #
    #                 # Crop without padding
    #                 self.cropped_preview_pixmap = self.current_image_pixmap.copy(x, y, w, h)
    #                 self.log(
    #                     f"[State] fp_image_array is {'set' if self.fp_image_array is not None else 'None'}; cropped_fp shape={getattr(self, 'cropped_fp', None).shape if hasattr(self, 'cropped_fp') else 'N/A'}")
    #
    #                 self.cropped_fp = self.fp_image_array[y:y + h, x:x + w, :]
    #                 self.log(
    #                     f"[State] fp_image_array is {'set' if self.fp_image_array is not None else 'None'}; cropped_fp shape={getattr(self, 'cropped_fp', None).shape if hasattr(self, 'cropped_fp') else 'N/A'}")
    #
    #                 self._display_preview(self.cropped_preview_pixmap)
    #                 # Switch out of manual-selection
    #                 self.manual_selection_mode = False
    #                 # Enable Detect, Show, Flatten
    #                 self._set_chart_tools_enabled(detect=True, show=True, flatten=True)
    #                 return True
    #
    #         # Flatten mode (perspective selection)
    #         if self.flatten_mode and event.type() == QEvent.MouseButtonPress:
    #             pt = event.position().toPoint()
    #             sp = self.ui.imagePreviewGraphicsView.mapToScene(pt)
    #             idx = len(self.corner_points) + 1
    #             self.corner_points.append((sp.x(), sp.y()))
    #             self.log(f"[Flatten] Point {idx}: ({int(sp.x())}, {int(sp.y())})")
    #
    #             # Draw red dot and label
    #             dot = FixedSizeEllipse(sp.x(), sp.y(), radius=8, color=QColor('red'))
    #             self.previewScene.addItem(dot)
    #             label = FixedSizeText(str(idx), sp.x() + 12, sp.y() - 8, color=QColor('red'))
    #             self.previewScene.addItem(label)
    #
    #             if idx == 4:
    #                 pts_src = np.array(self.corner_points, dtype=np.float32)
    #                 width = np.linalg.norm(pts_src[1] - pts_src[0])
    #                 height = int(width * 9.0 / 14.0)
    #                 dst_pts = np.array([
    #                     [PADDING, PADDING],
    #                     [PADDING + width, PADDING],
    #                     [PADDING + width, PADDING + height],
    #                     [PADDING, PADDING + height]
    #                 ], dtype=np.float32)
    #
    #                 # Extract full cropped image array with correct stride
    #                 qimg = self.cropped_preview_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
    #                 h0 = qimg.height()
    #                 stride = qimg.bytesPerLine()
    #                 buf = bytes(qimg.constBits())[: h0 * stride]
    #                 arr2d = np.frombuffer(buf, dtype=np.uint8).reshape((h0, stride))
    #                 arr = arr2d[:, : qimg.width() * 3].reshape((h0, qimg.width(), 3))
    #
    #                 # Pad full array
    #                 arr_padded = cv2.copyMakeBorder(
    #                     arr, PADDING, PADDING, PADDING, PADDING,
    #                     borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0)
    #                 )
    #                 pts_src_padded = pts_src + PADDING
    #
    #                 M = cv2.getPerspectiveTransform(pts_src_padded, dst_pts)
    #                 out_w = int(width + 2 * PADDING)
    #                 out_h = int(height + 2 * PADDING)
    #                 warped = cv2.warpPerspective(arr_padded, M, (out_w, out_h))
    #
    #                 # Show warped result
    #                 q2 = QImage(warped.data, out_w, out_h, out_w * 3, QImage.Format_RGB888)
    #                 pix2 = QPixmap.fromImage(q2)
    #                 self.previewScene.clear()
    #                 img_item = QGraphicsPixmapItem(pix2)
    #                 self.previewScene.addItem(img_item)
    #
    #                 # Draw swatch grid overlay on top of the warped image
    #                 swatch_grid_rect = QRect(PADDING, PADDING, int(width), int(height))
    #                 self.flatten_swatch_rects = self.draw_colorchecker_swatch_grid(
    #                     self.previewScene, swatch_grid_rect, n_cols=6, n_rows=4
    #                 )
    #
    #                 self.ui.imagePreviewGraphicsView.resetTransform()
    #                 self.ui.imagePreviewGraphicsView.fitInView(
    #                     self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio
    #                 )
    #
    #                 # Flatten FP and store back into cropped_fp
    #                 fparr = cv2.copyMakeBorder(
    #                     self.cropped_fp,
    #                     PADDING, PADDING, PADDING, PADDING,
    #                     borderType=cv2.BORDER_REFLECT
    #                 )
    #                 warped_fp = cv2.warpPerspective(
    #                     fparr.astype(np.float32),
    #                     M,
    #                     (out_w, out_h)
    #                 )
    #                 self.cropped_fp = warped_fp
    #                 self.log(
    #                     f"[State] fp_image_array is {'set' if self.fp_image_array is not None else 'None'}; cropped_fp shape={getattr(self, 'cropped_fp', None).shape if hasattr(self, 'cropped_fp') else 'N/A'}")
    #
    #                 # self.log("[Flatten] Chart and FP image flattened")
    #                 self.flatten_mode = False
    #                 self.instruction_label.setText(
    #                     "Please Run Detect Chart or Revert image to select new region"
    #                 )
    #                 self.ui.finalizeChartPushbutton.setEnabled(True)
    #                 self.preview_manual_swatch_correction()
    #             return True
    #

    def eventFilter(self, source, event):
        """
        Master event handler for manual-selection, flatten modes,
        idle zoom/pan, and thumbnail scroll/resize.
        """
        viewport = self.ui.imagePreviewGraphicsView.viewport()
        thumbnail_frame = self.ui.thumbnailPreviewFrame
        ev = event.type()

        # 1) Thumbnail scroll & resize
        if source == thumbnail_frame:
            if ev == QEvent.Wheel:
                delta = event.angleDelta().y()
                if delta > 0:
                    self.select_previous_image()
                elif delta < 0:
                    self.select_next_image()
                return True
            if ev == QEvent.Resize:
                self.update_thumbnail_strip()
                return True
            return super().eventFilter(source, event)

        # 2) Preview viewport events
        if source == viewport:
            # a) Idle zoom
            if ev == QEvent.Wheel and not (self.manual_selection_mode or self.flatten_mode):
                return self._handle_preview_zoom(event)
            # b) Idle pan
            if ev == QEvent.MouseButtonPress and event.button() == Qt.LeftButton \
                    and not (self.manual_selection_mode or self.flatten_mode):
                return self._handle_preview_pan_press(event)
            if ev == QEvent.MouseMove and hasattr(self, '_pan_start_pos'):
                return self._handle_preview_pan_move(event)
            if ev == QEvent.MouseButtonRelease and hasattr(self, '_pan_start_pos'):
                return self._handle_preview_pan_release(event)
            # c) Idle resize (respect auto-fit)
            if ev == QEvent.Resize and not (self.manual_selection_mode or self.flatten_mode):
                if getattr(self, '_auto_fit_enabled', True):
                    self.ui.imagePreviewGraphicsView.fitInView(
                        self.previewScene.itemsBoundingRect(), Qt.KeepAspectRatio)
                return True
            # d) Block stray pointer events when idle
            if not (self.manual_selection_mode or self.flatten_mode) and ev in (
                    QEvent.MouseButtonPress, QEvent.MouseMove, QEvent.MouseButtonRelease):
                return False
            # e) Manual-selection logic
            if self.manual_selection_mode and not self.flatten_mode:
                if ev == QEvent.MouseButtonPress:
                    pos = event.position().toPoint()
                    scene_pt = self.ui.imagePreviewGraphicsView.mapToScene(pos)
                    if not self.previewScene.itemsBoundingRect().contains(scene_pt):
                        return False
                    return self._handle_manual_press(event)
                elif ev == QEvent.MouseMove:
                    return self._handle_manual_move(event)
                elif ev == QEvent.MouseButtonRelease:
                    return self._handle_manual_release(event)
            # f) Flatten mode logic
            if self.flatten_mode and ev == QEvent.MouseButtonPress:
                return self._handle_flatten_press(event)

        # 3) Default fallback
        return super().eventFilter(source, event)

    def _handle_manual_press(self, event):
        self.origin = event.position().toPoint()
        if not hasattr(self, 'rubberBand'):
            self.rubberBand = QRubberBand(QRubberBand.Rectangle, self.ui.imagePreviewGraphicsView)
        self.rubberBand.setGeometry(QRect(self.origin, QSize()))
        self.rubberBand.show()
        return True

    def _handle_manual_move(self, event):
        if self.rubberBand.isVisible():
            rect = QRect(self.origin, event.position().toPoint()).normalized()
            self.rubberBand.setGeometry(rect)
            return True
        return False

    def _handle_manual_release(self, event):
        if not self.rubberBand.isVisible():
            return False
        # Map rubber-band rect to scene coords
        rect = self.rubberBand.geometry()
        tl = self.ui.imagePreviewGraphicsView.mapToScene(rect.topLeft())
        br = self.ui.imagePreviewGraphicsView.mapToScene(rect.bottomRight())
        x, y = int(tl.x()), int(tl.y())
        w, h = abs(int(br.x() - tl.x())), abs(int(br.y() - tl.y()))

        self.rubberBand.hide()
        self.manual_selection_mode = False

        # Crop data
        self.cropped_preview_pixmap = self.current_image_pixmap.copy(x, y, w, h)
        self.cropped_fp = self.fp_image_array[y:y+h, x:x+w, :]

        self._display_preview(self.cropped_preview_pixmap)
        self._set_chart_tools_enabled(detect=True, show=True, flatten=True)
        return True

    def _handle_flatten_press(self, event):
        pt = event.position().toPoint()
        sp = self.ui.imagePreviewGraphicsView.mapToScene(pt)
        idx = len(self.corner_points) + 1
        self.corner_points.append((sp.x(), sp.y()))
        self.log(f"[Flatten] Point {idx}: ({int(sp.x())}, {int(sp.y())})")

        # Draw UI markers
        dot = FixedSizeEllipse(sp.x(), sp.y(), radius=8, color=QColor('red'))
        label = FixedSizeText(str(idx), sp.x()+12, sp.y()-8, color=QColor('red'))
        self.previewScene.addItem(dot)
        self.previewScene.addItem(label)

        # Once four points selected, perform flatten
        if idx == 4:
            self._perform_flatten_transform()
        return True

    def _perform_flatten_transform(self):
        """
        Executes perspective warp and grid overlay after four corner points are set.
        """
        PADDING = 600
        pts_src = np.array(self.corner_points, dtype=np.float32)
        width = np.linalg.norm(pts_src[1] - pts_src[0])
        height = int(width * 9.0 / 14.0)
        dst_pts = np.array([
            [PADDING, PADDING], [PADDING+width, PADDING],
            [PADDING+width, PADDING+height], [PADDING, PADDING+height]
        ], dtype=np.float32)

        # Extract byte buffer from cropped_preview_pixmap
        qimg = self.cropped_preview_pixmap.toImage().convertToFormat(QImage.Format_RGB888)
        h0 = qimg.height(); stride = qimg.bytesPerLine()
        buf = bytes(qimg.constBits())[:h0*stride]
        arr = np.frombuffer(buf, dtype=np.uint8).reshape((h0, stride))
        arr = arr[:, :qimg.width()*3].reshape((h0, qimg.width(), 3))

        # Pad and compute warp
        arr_p = cv2.copyMakeBorder(arr, PADDING, PADDING, PADDING, PADDING,
                                  borderType=cv2.BORDER_CONSTANT, value=(0,0,0))
        pts_src_p = pts_src + PADDING
        M = cv2.getPerspectiveTransform(pts_src_p, dst_pts)
        warped = cv2.warpPerspective(arr_p, M, (int(width+2*PADDING), int(height+2*PADDING)))

        # Display warped image
        q2 = QImage(warped.data, warped.shape[1], warped.shape[0], warped.shape[1]*3,
                    QImage.Format_RGB888)
        self._display_preview(QPixmap.fromImage(q2))

        # Draw swatch grid and update fp array
        swatch_rect = QRect(PADDING, PADDING, int(width), int(height))
        self.flatten_swatch_rects = self.draw_colorchecker_swatch_grid(
            self.previewScene, swatch_rect, n_cols=6, n_rows=4)

        # Transform float-precision data
        fparr = cv2.copyMakeBorder(self.cropped_fp, PADDING, PADDING, PADDING, PADDING,
                                   borderType=cv2.BORDER_REFLECT)
        self.cropped_fp = cv2.warpPerspective(fparr.astype(np.float32), M,
                                              (warped.shape[1], warped.shape[0]))
        self.flatten_mode = False
        self.instruction_label.setText(
            "Please Run Detect Chart or Revert image to select new region")
        self.ui.finalizeChartPushbutton.setEnabled(True)
        self.preview_manual_swatch_correction()

    def _handle_preview_zoom(self, event):
        """
        Zoom in/out on the preview viewport with mouse wheel when idle.
        Uses smooth zoom based on angleDelta and transforms around cursor.
        Disables auto-fit once user zooms.
        """
        # Ensure auto-fit flag exists
        if not hasattr(self, '_auto_fit_enabled'):
            self._auto_fit_enabled = True
        # Normalize wheel delta (120 units = one notch)
        delta = event.angleDelta().y() / 120.0
        factor = 1.15 ** delta
        view = self.ui.imagePreviewGraphicsView
        view.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        view.scale(factor, factor)
        # After manual zoom, prevent automatic fit-on-resize
        self._auto_fit_enabled = False
        return True

    def _handle_preview_pan_press(self, event):
        """
        Start panning on left mouse button press when idle.
        """
        if event.button() == Qt.LeftButton:
            self._pan_start_pos = event.position().toPoint()
            view = self.ui.imagePreviewGraphicsView
            self._pan_h_scroll = view.horizontalScrollBar().value()
            self._pan_v_scroll = view.verticalScrollBar().value()
            return True
        return False

    def _handle_preview_pan_move(self, event):
        """
        Pan the view as the mouse moves when dragging.
        """
        if hasattr(self, '_pan_start_pos'):
            delta = event.position().toPoint() - self._pan_start_pos
            view = self.ui.imagePreviewGraphicsView
            view.horizontalScrollBar().setValue(self._pan_h_scroll - delta.x())
            view.verticalScrollBar().setValue(self._pan_v_scroll - delta.y())
            return True
        return False

    def _handle_preview_pan_release(self, event):
        """
        End panning on mouse release.
        """
        if hasattr(self, '_pan_start_pos'):
            del self._pan_start_pos, self._pan_h_scroll, self._pan_v_scroll
            return True
        return False

    def preview_manual_swatch_correction(self):
        """
        Extracts mean colours from swatch rectangles in self.flatten_swatch_rects,
        applies manual color correction, and displays the corrected image preview.
        Assumes self.cropped_fp is the current floating-point chart image,
        and self.flatten_swatch_rects is a list of 24 QRect objects.
        """

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
        self.chart_swatches = swatch_colours
        tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
        np.save(tmp_file.name, swatch_colours)
        self.calibration_file = tmp_file.name

        worker = SwatchPreviewWorker(
            self.cropped_fp,
            swatch_colours,
            reference_swatches
        )
        worker.signals.finished.connect(self._on_manual_swatch_preview_done)
        worker.signals.error.connect(self._on_manual_swatch_preview_error)

        self.threadpool.start(worker)

    def _on_manual_swatch_preview_done(self, img_uint8):
        """
        Receives the uint8 array, converts to QPixmap, updates preview, re-enables UI.
        """
        pixmap = self.pixmap_from_array(img_uint8)
        self._display_preview(pixmap)
        # self.log("[Manual Swatch] Manual correction preview displayed.")

    def _on_manual_swatch_preview_error(self, message):
        """
        Called if the worker threw an exception.
        """
        self.log(f"[Manual Swatch] Preview failed: {message}")


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
            img_fp = self.load_raw_image(input_source)

        selected_item = self.ui.imagesListWidget.currentItem()
        if selected_item:
            meta = selected_item.data(Qt.UserRole)
            selected_item.setData(Qt.UserRole, meta)
            selected_item.setBackground(QColor('#ADD8E6'))

        results = detect_colour_checkers_segmentation(img_fp, additional_data=True)

        for colour_checker_data in results:
            if hasattr(colour_checker_data, 'swatch_colours'):
                swatch_colours = colour_checker_data.swatch_colours
                swatch_masks = colour_checker_data.swatch_masks
                colour_checker_image = colour_checker_data.colour_checker
            else:
                swatch_colours, swatch_masks, colour_checker_image = colour_checker_data[:3]

            # Save calibration data
            tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.npy')
            np.save(tmp_file.name, swatch_colours)
            self.calibration_file = tmp_file.name
            self.log(f'[Detect] Calibration saved to {self.calibration_file}')
            self.chart_swatches = swatch_colours

            # Generate Corrected Image
            corrected = colour.colour_correction(img_fp, swatch_colours, reference_swatches)
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
            # self.log("[Detect] Chart detected successfully. Debug images generated.")
            self.show_debug_frame(True)
            self.ui.finalizeChartPushbutton.setEnabled(True)

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
        """
        Restore the original preview and fully reset the manual‐selection UI.

        - Displays the saved original_preview_pixmap.
        - Hides any rubber‐band or instruction label.
        - Closes the debug frame.
        - Disables all chart‐tool buttons (detect, show, flatten, finalize).
        - Enables only the manual‐select button for a new crop.
        """
        # 1) Show original preview
        if hasattr(self, 'original_preview_pixmap'):
            self.current_image_pixmap = self.original_preview_pixmap
            self._display_preview(self.original_preview_pixmap)
            self.showing_chart_preview = False

        # 2) Hide any debug/chart overlay
        self.show_debug_frame(False)
        if hasattr(self, 'instruction_label'):
            self.instruction_label.hide()

        # 3) Remove any rubber‐band selection
        if hasattr(self, 'rubberBand') and self.rubberBand.isVisible():
            self.rubberBand.hide()
        # Prepare for a fresh manual selection
        self.manual_selection_mode = True
        self.flatten_mode = False
        self.corner_points.clear()

        self._set_chart_tools_enabled(manual=True)

        # 6) Clear any temp data from prior selection:
        if hasattr(self, 'temp_swatches'):
            del self.temp_swatches
        if hasattr(self, 'temp_calibration_file'):
            del self.temp_calibration_file


    def show_debug_frame(self, visible):
        """
        Shows the debug window for colour chart debugging
        :param visible:
        :return:
        """
        self.ui.colourChartDebugToolsFrame.setVisible(visible)

    def setup_debug_views(self):
        """
        Setup the UI for debugging
        :return:
        """
        self.ui.correctedImageRadioButton.toggled.connect(lambda checked: checked and self.corrected_image_view())
        self.ui.swatchOverlayRadioButton.toggled.connect(lambda checked: checked and self.swatch_overlay_view())
        self.ui.swatchAndClusterRadioButton.toggled.connect(
            lambda checked: checked and self.swatches_and_clusters_view())
        self.ui.detectionDebugRadioButton.toggled.connect(lambda checked: checked and self.detection_debug_view())

    def swatches_and_clusters_view(self):
        """
        Display the debug image showing swatch bounding‐boxes and cluster outlines in the preview area.
        """
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
            # self.log("[Debug View] Swatches and Clusters shown.")
        else:
            self.log("[Debug View] Swatches and Clusters unavailable.")

    def corrected_image_view(self):
        """
        Display the colour‐corrected debug image for the selected item in the preview area.
        """
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
            # self.log("[Debug View] Corrected image shown.")
        else:
            self.log("[Debug View] Corrected image unavailable.")

    def swatch_overlay_view(self):
        """
        Display the swatch‐overlay debug image (chart with semi‐transparent swatches) in the preview area.
        """
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
            # self.log("[Debug View] Swatch overlay shown.")
        else:
            self.log("[Debug View] Swatch overlay unavailable.")

    def detection_debug_view(self):
        """
        Display the segmentation (detected patch shapes) debug image for the selected item.
        """
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
            # self.log("[Debug View] Detection debug shown.")
        else:
            self.log("[Debug View] Detection debug unavailable.")

    def pixmap_from_array(self, array):
        """
        Convert a H×W×3 uint8 NumPy array into a QPixmap for display.
        Assumes RGB888 layout.
        """
        h, w, c = array.shape
        bytes_per_line = c * w
        img = QImage(array.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return QPixmap.fromImage(img)

    @exit_manual_mode
    def calculate_average_exposure(self):
        """
        Compute and store exposure normalization multipliers for all images.
        Uses chart image as reference if available, otherwise uses average.
        Ignores top 2% (hot spots) and bottom 5% (black areas) of pixel luminance.
        """
        brightness_list = []
        item_to_brightness = {}
        chart_brightness = None
        self.average_enabled = True

        if self.selected_average_source is None:
            self.log("[Exposure Calc] No selected average source.")

        else:

            for i in range(self.ui.imagesListWidget.count()):
                max_size = (512, 512)
                item = self.ui.imagesListWidget.item(i)
                meta = item.data(Qt.UserRole)
                img_path = meta.get('input_path')
                arr = None
                cache = self.thumbnail_cache.get(img_path)
                if cache and 'array' in cache and cache['array'] is not None:
                    arr = cache['array']

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
                if meta.get('average_source'):
                    chart_brightness = mean_brightness  # Only one chart expected, take the first found

            if not brightness_list:
                self.log("[Exposure Calc] No valid images for exposure normalization.")
                return

            # Reference: chart if present, otherwise mean of all
            reference_brightness = chart_brightness if chart_brightness is not None else np.mean(brightness_list)
            if chart_brightness is None:
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
                # self.log("[Thumb] Thumbnail loaded from image without further processing.")
                if self.chart_swatches and self.correct_thumbnails:
                    try:
                        corrected = colour.colour_correction(arr, self.chart_swatches, reference_swatches)
                        corrected = np.clip(corrected, 0, 1)
                        arr = np.uint8(255 * colour.cctf_encoding(corrected))
                        # self.log("[Thumb] Applied colour correction to thumbnail.")
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
                                # self.log("[Thumb] Loaded embedded JPEG thumbnail from RAW.")
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
                        # self.log("[Thumb] Loaded thumbnail from rawpy postprocess.")
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

        self.average_enabled = False
        self.update_thumbnail_strip()
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
        # self.log(f"[Exposure Debug] Overlay shown (Highlight: ≥{highlight_threshold:.2f}, Shadow: ≤{shadow_threshold:.2f})")

    @exit_manual_mode
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

    def set_selected_image_as_average_source(self):
        self.selected_average_source = self.ui.imagesListWidget.currentItem()
        data = self.selected_average_source.data(Qt.UserRole)
        img_path = data.get("input_path")
        data["average_source"] = True
        self.selected_average_source.setData(Qt.UserRole, data)
        self.log(f"set {img_path} as average source")

    def _adjust_pixmap_brightness(self, pixmap: QPixmap, factor: float) -> QPixmap:
        """
        Return a new QPixmap with brightness adjusted by `factor`.
        Factor >1 brightens, <1 darkens.
        """
        img = pixmap.toImage().convertToFormat(QImage.Format_RGBA8888)
        w, h = img.width(), img.height()

        ptr = img.bits()
        arr = np.ndarray((h, w, 4), dtype=np.uint8, buffer=ptr)

        rgb = arr[..., :3].astype(np.float32) * factor
        arr[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)

        return QPixmap.fromImage(img)

    def update_thumbnail_strip(self):
        """
        Refresh and layout all thumbnail widgets; apply any average-exposure brightness tweak.
        """
        holder = self.ui.thumbnailPreviewDisplayFrame_holder
        # Clear previous
        for child in holder.findChildren(QWidget):
            child.setParent(None)
            child.deleteLater()

        count = self.ui.imagesListWidget.count()
        if count == 0:
            return

        sel_idx = self.ui.imagesListWidget.currentRow()
        frame_width = holder.width()

        # Compute widths
        default_aspect = 1.5
        default_width = int(default_aspect * 60)
        widths = []
        for i in range(count):
            meta = self.ui.imagesListWidget.item(i).data(Qt.UserRole)
            cache = self.thumbnail_cache.get(meta['input_path'], {})
            pixmap = cache.get('pixmap')
            if pixmap and not pixmap.isNull():
                aspect = pixmap.width() / pixmap.height()
                widths.append(int(aspect * 60))
            else:
                widths.append(default_width)

        thumb_indices = [sel_idx]
        total_width = widths[sel_idx]
        left, right = sel_idx-1, sel_idx+1
        while left>=0 or right<count:
            if left>=0 and (right>=count or len(thumb_indices)%2==1):
                if total_width+widths[left] > frame_width and len(thumb_indices)>=3: break
                thumb_indices.insert(0,left); total_width+=widths[left]; left-=1
            elif right<count:
                if total_width+widths[right] > frame_width and len(thumb_indices)>=3: break
                thumb_indices.append(right); total_width+=widths[right]; right+=1
            else:
                break

        display_items = [self.ui.imagesListWidget.item(i) for i in thumb_indices]
        display_widths = [widths[i] for i in thumb_indices]
        if 0<total_width<frame_width:
            scale = frame_width/total_width
            display_widths = [int(w*scale) for w in display_widths]

        # Build thumbnails
        x_offset = 0
        for idx, item in zip(thumb_indices, display_items):
            meta = item.data(Qt.UserRole)
            cache = self.thumbnail_cache.get(meta['input_path'], {})
            pixmap = cache.get('pixmap')
            label = ClickableLabel(idx, holder)
            if pixmap and not pixmap.isNull():
                thumb = pixmap.scaled(display_widths[thumb_indices.index(idx)], 60,
                                      Qt.KeepAspectRatio, Qt.SmoothTransformation)
                # Apply brightness adjust if enabled
                factor = meta.get('average_exposure', 1.0)
                if factor != 1.0:
                    thumb = self._adjust_pixmap_brightness(thumb, factor)
                label.setPixmap(thumb)
                label.setFixedSize(thumb.width(), thumb.height())
            else:
                label.setText("No preview")
                label.setFixedSize(display_widths[thumb_indices.index(idx)], 60)

            label.setStyleSheet(
                "border: 2px solid #2196F3;" if idx==sel_idx else "border: 1px solid #999;"
            )
            label.clicked.connect(self.select_image_from_thumbnail)
            label.move(x_offset, 0)
            label.show()
            x_offset += label.width()

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
