# -*- coding: utf-8 -*-

################################################################################
## Form generated from reading UI file 'scanSpaceImageProcessor.ui'
##
## Created by: Qt User Interface Compiler version 6.5.1
##
## WARNING! All changes made in this file will be lost when recompiling UI file!
################################################################################

from PySide6.QtCore import (QCoreApplication, QDate, QDateTime, QLocale,
    QMetaObject, QObject, QPoint, QRect,
    QSize, QTime, QUrl, Qt)
from PySide6.QtGui import (QBrush, QColor, QConicalGradient, QCursor,
    QFont, QFontDatabase, QGradient, QIcon,
    QImage, QKeySequence, QLinearGradient, QPainter,
    QPalette, QPixmap, QRadialGradient, QTransform)
from PySide6.QtWidgets import (QApplication, QCheckBox, QFrame, QGraphicsView,
    QGridLayout, QHBoxLayout, QLabel, QLayout,
    QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QProgressBar, QPushButton, QRadioButton, QSizePolicy,
    QSpacerItem, QSpinBox, QStatusBar, QTextEdit,
    QVBoxLayout, QWidget)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        if not MainWindow.objectName():
            MainWindow.setObjectName(u"MainWindow")
        MainWindow.resize(1301, 766)
        MainWindow.setMinimumSize(QSize(1074, 650))
        self.centralwidget = QWidget(MainWindow)
        self.centralwidget.setObjectName(u"centralwidget")
        self.gridLayout_2 = QGridLayout(self.centralwidget)
        self.gridLayout_2.setObjectName(u"gridLayout_2")
        self.toolsLayoutFrame = QFrame(self.centralwidget)
        self.toolsLayoutFrame.setObjectName(u"toolsLayoutFrame")
        self.toolsLayoutFrame.setMinimumSize(QSize(180, 0))
        self.toolsLayoutFrame.setMaximumSize(QSize(180, 16777215))
        self.toolsLayoutFrame.setFrameShape(QFrame.StyledPanel)
        self.toolsLayoutFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout = QVBoxLayout(self.toolsLayoutFrame)
        self.verticalLayout.setObjectName(u"verticalLayout")
        self.toolsLayoutGrid = QGridLayout()
        self.toolsLayoutGrid.setObjectName(u"toolsLayoutGrid")
        self.useOriginalFilenamesCheckBox = QCheckBox(self.toolsLayoutFrame)
        self.useOriginalFilenamesCheckBox.setObjectName(u"useOriginalFilenamesCheckBox")

        self.toolsLayoutGrid.addWidget(self.useOriginalFilenamesCheckBox, 15, 0, 1, 2)

        self.label_13 = QLabel(self.toolsLayoutFrame)
        self.label_13.setObjectName(u"label_13")

        self.toolsLayoutGrid.addWidget(self.label_13, 12, 0, 1, 1)

        self.label_9 = QLabel(self.toolsLayoutFrame)
        self.label_9.setObjectName(u"label_9")
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        self.label_9.setFont(font)
        self.label_9.setAlignment(Qt.AlignCenter)

        self.toolsLayoutGrid.addWidget(self.label_9, 6, 0, 1, 2)

        self.newImageNameLineEdit = QLineEdit(self.toolsLayoutFrame)
        self.newImageNameLineEdit.setObjectName(u"newImageNameLineEdit")

        self.toolsLayoutGrid.addWidget(self.newImageNameLineEdit, 16, 0, 1, 2)

        self.calculateAverageExposurePushbutton = QPushButton(self.toolsLayoutFrame)
        self.calculateAverageExposurePushbutton.setObjectName(u"calculateAverageExposurePushbutton")

        self.toolsLayoutGrid.addWidget(self.calculateAverageExposurePushbutton, 7, 0, 1, 2)

        self.setSelectedAsChartPushbutton = QPushButton(self.toolsLayoutFrame)
        self.setSelectedAsChartPushbutton.setObjectName(u"setSelectedAsChartPushbutton")

        self.toolsLayoutGrid.addWidget(self.setSelectedAsChartPushbutton, 1, 0, 1, 2)

        self.imagePaddingSpinBox = QSpinBox(self.toolsLayoutFrame)
        self.imagePaddingSpinBox.setObjectName(u"imagePaddingSpinBox")
        self.imagePaddingSpinBox.setMaximum(6)
        self.imagePaddingSpinBox.setValue(4)

        self.toolsLayoutGrid.addWidget(self.imagePaddingSpinBox, 17, 1, 1, 1)

        self.removeAverageDataPushbutton = QPushButton(self.toolsLayoutFrame)
        self.removeAverageDataPushbutton.setObjectName(u"removeAverageDataPushbutton")

        self.toolsLayoutGrid.addWidget(self.removeAverageDataPushbutton, 8, 0, 1, 2)

        self.label_7 = QLabel(self.toolsLayoutFrame)
        self.label_7.setObjectName(u"label_7")
        self.label_7.setIndent(5)

        self.toolsLayoutGrid.addWidget(self.label_7, 17, 0, 1, 1)

        self.shadowLimitSpinBox = QSpinBox(self.toolsLayoutFrame)
        self.shadowLimitSpinBox.setObjectName(u"shadowLimitSpinBox")
        self.shadowLimitSpinBox.setValue(6)

        self.toolsLayoutGrid.addWidget(self.shadowLimitSpinBox, 12, 1, 1, 1)

        self.label_5 = QLabel(self.toolsLayoutFrame)
        self.label_5.setObjectName(u"label_5")
        self.label_5.setMinimumSize(QSize(80, 0))
        self.label_5.setAlignment(Qt.AlignLeading|Qt.AlignLeft|Qt.AlignVCenter)
        self.label_5.setMargin(5)
        self.label_5.setIndent(5)

        self.toolsLayoutGrid.addWidget(self.label_5, 22, 0, 1, 1)

        self.processImagesPushbutton = QPushButton(self.toolsLayoutFrame)
        self.processImagesPushbutton.setObjectName(u"processImagesPushbutton")
        self.processImagesPushbutton.setMinimumSize(QSize(100, 40))

        self.toolsLayoutGrid.addWidget(self.processImagesPushbutton, 25, 0, 1, 2)

        self.previewChartPushbutton = QPushButton(self.toolsLayoutFrame)
        self.previewChartPushbutton.setObjectName(u"previewChartPushbutton")

        self.toolsLayoutGrid.addWidget(self.previewChartPushbutton, 2, 0, 1, 2)

        self.imageProcessingThreadsSpinbox = QSpinBox(self.toolsLayoutFrame)
        self.imageProcessingThreadsSpinbox.setObjectName(u"imageProcessingThreadsSpinbox")
        self.imageProcessingThreadsSpinbox.setMinimum(1)
        self.imageProcessingThreadsSpinbox.setValue(4)

        self.toolsLayoutGrid.addWidget(self.imageProcessingThreadsSpinbox, 24, 1, 1, 1)

        self.verticalSpacer_2 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.toolsLayoutGrid.addItem(self.verticalSpacer_2, 4, 0, 1, 2)

        self.label_4 = QLabel(self.toolsLayoutFrame)
        self.label_4.setObjectName(u"label_4")
        self.label_4.setMargin(5)
        self.label_4.setIndent(5)

        self.toolsLayoutGrid.addWidget(self.label_4, 24, 0, 1, 1)

        self.label_3 = QLabel(self.toolsLayoutFrame)
        self.label_3.setObjectName(u"label_3")
        self.label_3.setFont(font)
        self.label_3.setAlignment(Qt.AlignCenter)

        self.toolsLayoutGrid.addWidget(self.label_3, 14, 0, 1, 2)

        self.label_10 = QLabel(self.toolsLayoutFrame)
        self.label_10.setObjectName(u"label_10")
        self.label_10.setFont(font)
        self.label_10.setAlignment(Qt.AlignCenter)

        self.toolsLayoutGrid.addWidget(self.label_10, 0, 0, 1, 2)

        self.verticalSpacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)

        self.toolsLayoutGrid.addItem(self.verticalSpacer, 26, 0, 1, 2)

        self.highlightLimitSpinBox = QSpinBox(self.toolsLayoutFrame)
        self.highlightLimitSpinBox.setObjectName(u"highlightLimitSpinBox")
        self.highlightLimitSpinBox.setValue(96)

        self.toolsLayoutGrid.addWidget(self.highlightLimitSpinBox, 11, 1, 1, 1)

        self.verticalSpacer_3 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.toolsLayoutGrid.addItem(self.verticalSpacer_3, 13, 0, 1, 2)

        self.exportMaskedImagesCheckBox = QCheckBox(self.toolsLayoutFrame)
        self.exportMaskedImagesCheckBox.setObjectName(u"exportMaskedImagesCheckBox")

        self.toolsLayoutGrid.addWidget(self.exportMaskedImagesCheckBox, 10, 0, 1, 2)

        self.manuallySelectChartPushbutton = QPushButton(self.toolsLayoutFrame)
        self.manuallySelectChartPushbutton.setObjectName(u"manuallySelectChartPushbutton")

        self.toolsLayoutGrid.addWidget(self.manuallySelectChartPushbutton, 3, 0, 1, 2)

        self.displayDebugExposureDataCheckBox = QCheckBox(self.toolsLayoutFrame)
        self.displayDebugExposureDataCheckBox.setObjectName(u"displayDebugExposureDataCheckBox")

        self.toolsLayoutGrid.addWidget(self.displayDebugExposureDataCheckBox, 9, 0, 1, 2)

        self.label_11 = QLabel(self.toolsLayoutFrame)
        self.label_11.setObjectName(u"label_11")
        self.label_11.setFont(font)
        self.label_11.setAlignment(Qt.AlignCenter)

        self.toolsLayoutGrid.addWidget(self.label_11, 19, 0, 1, 2)

        self.verticalSpacer_4 = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Maximum)

        self.toolsLayoutGrid.addItem(self.verticalSpacer_4, 18, 0, 1, 2)

        self.label_12 = QLabel(self.toolsLayoutFrame)
        self.label_12.setObjectName(u"label_12")

        self.toolsLayoutGrid.addWidget(self.label_12, 11, 0, 1, 1)

        self.jpegQualitySpinbox = QSpinBox(self.toolsLayoutFrame)
        self.jpegQualitySpinbox.setObjectName(u"jpegQualitySpinbox")
        self.jpegQualitySpinbox.setMinimum(10)
        self.jpegQualitySpinbox.setMaximum(100)
        self.jpegQualitySpinbox.setSingleStep(10)
        self.jpegQualitySpinbox.setValue(100)

        self.toolsLayoutGrid.addWidget(self.jpegQualitySpinbox, 22, 1, 1, 1)


        self.verticalLayout.addLayout(self.toolsLayoutGrid)


        self.gridLayout_2.addWidget(self.toolsLayoutFrame, 0, 1, 1, 1)

        self.monotoringHorizontalLayout = QHBoxLayout()
        self.monotoringHorizontalLayout.setObjectName(u"monotoringHorizontalLayout")
        self.horizontalSpacer_5 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.monotoringHorizontalLayout.addItem(self.horizontalSpacer_5)

        self.memoryUsageProgressBar = QProgressBar(self.centralwidget)
        self.memoryUsageProgressBar.setObjectName(u"memoryUsageProgressBar")
        self.memoryUsageProgressBar.setValue(24)

        self.monotoringHorizontalLayout.addWidget(self.memoryUsageProgressBar)

        self.cpuUsageProgressBar = QProgressBar(self.centralwidget)
        self.cpuUsageProgressBar.setObjectName(u"cpuUsageProgressBar")
        self.cpuUsageProgressBar.setValue(24)

        self.monotoringHorizontalLayout.addWidget(self.cpuUsageProgressBar)

        self.horizontalSpacer_6 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.monotoringHorizontalLayout.addItem(self.horizontalSpacer_6)


        self.gridLayout_2.addLayout(self.monotoringHorizontalLayout, 1, 0, 1, 5)

        self.frame = QFrame(self.centralwidget)
        self.frame.setObjectName(u"frame")
        self.frame.setMinimumSize(QSize(350, 0))
        self.frame.setMaximumSize(QSize(350, 16777215))
        self.frame.setFrameShape(QFrame.StyledPanel)
        self.frame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_3 = QVBoxLayout(self.frame)
        self.verticalLayout_3.setObjectName(u"verticalLayout_3")
        self.verticalLayout_2 = QVBoxLayout()
        self.verticalLayout_2.setObjectName(u"verticalLayout_2")
        self.chartImageHorizontalLayout = QHBoxLayout()
        self.chartImageHorizontalLayout.setObjectName(u"chartImageHorizontalLayout")
        self.label_8 = QLabel(self.frame)
        self.label_8.setObjectName(u"label_8")
        self.label_8.setMinimumSize(QSize(0, 0))
        self.label_8.setMaximumSize(QSize(120, 16777215))
        self.label_8.setMargin(1)
        self.label_8.setIndent(5)

        self.chartImageHorizontalLayout.addWidget(self.label_8)

        self.chartPathLineEdit = QLineEdit(self.frame)
        self.chartPathLineEdit.setObjectName(u"chartPathLineEdit")
        self.chartPathLineEdit.setMaximumSize(QSize(250, 16777215))

        self.chartImageHorizontalLayout.addWidget(self.chartPathLineEdit)

        self.browseForChartPushbutton = QPushButton(self.frame)
        self.browseForChartPushbutton.setObjectName(u"browseForChartPushbutton")
        self.browseForChartPushbutton.setMinimumSize(QSize(60, 0))
        self.browseForChartPushbutton.setMaximumSize(QSize(100, 16777215))

        self.chartImageHorizontalLayout.addWidget(self.browseForChartPushbutton)


        self.verticalLayout_2.addLayout(self.chartImageHorizontalLayout)

        self.rawImagesHorizontalLayout = QHBoxLayout()
        self.rawImagesHorizontalLayout.setObjectName(u"rawImagesHorizontalLayout")
        self.label_2 = QLabel(self.frame)
        self.label_2.setObjectName(u"label_2")
        self.label_2.setMinimumSize(QSize(0, 0))
        self.label_2.setMaximumSize(QSize(120, 16777215))
        self.label_2.setMargin(1)
        self.label_2.setIndent(5)

        self.rawImagesHorizontalLayout.addWidget(self.label_2)

        self.rawImagesDirectoryLineEdit = QLineEdit(self.frame)
        self.rawImagesDirectoryLineEdit.setObjectName(u"rawImagesDirectoryLineEdit")
        self.rawImagesDirectoryLineEdit.setMaximumSize(QSize(250, 16777215))

        self.rawImagesHorizontalLayout.addWidget(self.rawImagesDirectoryLineEdit)

        self.browseForImagesPushbutton = QPushButton(self.frame)
        self.browseForImagesPushbutton.setObjectName(u"browseForImagesPushbutton")
        self.browseForImagesPushbutton.setMinimumSize(QSize(60, 0))
        self.browseForImagesPushbutton.setMaximumSize(QSize(100, 16777215))

        self.rawImagesHorizontalLayout.addWidget(self.browseForImagesPushbutton)


        self.verticalLayout_2.addLayout(self.rawImagesHorizontalLayout)

        self.outputDirectoryHorizontalLayout = QHBoxLayout()
        self.outputDirectoryHorizontalLayout.setObjectName(u"outputDirectoryHorizontalLayout")
        self.label_6 = QLabel(self.frame)
        self.label_6.setObjectName(u"label_6")
        self.label_6.setMinimumSize(QSize(0, 0))
        self.label_6.setMaximumSize(QSize(120, 16777215))
        self.label_6.setMargin(1)
        self.label_6.setIndent(5)

        self.outputDirectoryHorizontalLayout.addWidget(self.label_6)

        self.outputDirectoryLineEdit = QLineEdit(self.frame)
        self.outputDirectoryLineEdit.setObjectName(u"outputDirectoryLineEdit")
        self.outputDirectoryLineEdit.setMaximumSize(QSize(250, 16777215))

        self.outputDirectoryHorizontalLayout.addWidget(self.outputDirectoryLineEdit)

        self.browseoutputDirectoryPushbutton = QPushButton(self.frame)
        self.browseoutputDirectoryPushbutton.setObjectName(u"browseoutputDirectoryPushbutton")
        self.browseoutputDirectoryPushbutton.setMinimumSize(QSize(60, 0))
        self.browseoutputDirectoryPushbutton.setMaximumSize(QSize(100, 16777215))

        self.outputDirectoryHorizontalLayout.addWidget(self.browseoutputDirectoryPushbutton)


        self.verticalLayout_2.addLayout(self.outputDirectoryHorizontalLayout)

        self.label = QLabel(self.frame)
        self.label.setObjectName(u"label")
        font1 = QFont()
        font1.setBold(True)
        self.label.setFont(font1)
        self.label.setAlignment(Qt.AlignCenter)

        self.verticalLayout_2.addWidget(self.label)

        self.imagesListWidget = QListWidget(self.frame)
        self.imagesListWidget.setObjectName(u"imagesListWidget")
        self.imagesListWidget.setMaximumSize(QSize(340, 16777215))

        self.verticalLayout_2.addWidget(self.imagesListWidget)


        self.verticalLayout_3.addLayout(self.verticalLayout_2)


        self.gridLayout_2.addWidget(self.frame, 0, 0, 1, 1)

        self.verticalLayout_4 = QVBoxLayout()
        self.verticalLayout_4.setSpacing(8)
        self.verticalLayout_4.setObjectName(u"verticalLayout_4")
        self.verticalLayout_4.setContentsMargins(-1, 10, -1, 0)
        self.detectChartToolshelfFrame = QFrame(self.centralwidget)
        self.detectChartToolshelfFrame.setObjectName(u"detectChartToolshelfFrame")
        self.detectChartToolshelfFrame.setMinimumSize(QSize(0, 20))
        self.detectChartToolshelfFrame.setFrameShape(QFrame.StyledPanel)
        self.detectChartToolshelfFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_5 = QVBoxLayout(self.detectChartToolshelfFrame)
        self.verticalLayout_5.setObjectName(u"verticalLayout_5")
        self.verticalLayout_5.setContentsMargins(-1, 0, -1, 0)
        self.detectChartToolshelfLayout = QHBoxLayout()
        self.detectChartToolshelfLayout.setObjectName(u"detectChartToolshelfLayout")
        self.detectChartToolshelfLayout.setSizeConstraint(QLayout.SetDefaultConstraint)
        self.horizontalSpacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.detectChartToolshelfLayout.addItem(self.horizontalSpacer)

        self.showOriginalImagePushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.showOriginalImagePushbutton.setObjectName(u"showOriginalImagePushbutton")

        self.detectChartToolshelfLayout.addWidget(self.showOriginalImagePushbutton)

        self.flattenChartImagePushButton = QPushButton(self.detectChartToolshelfFrame)
        self.flattenChartImagePushButton.setObjectName(u"flattenChartImagePushButton")

        self.detectChartToolshelfLayout.addWidget(self.flattenChartImagePushButton)

        self.revertImagePushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.revertImagePushbutton.setObjectName(u"revertImagePushbutton")

        self.detectChartToolshelfLayout.addWidget(self.revertImagePushbutton)

        self.detectChartShelfPushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.detectChartShelfPushbutton.setObjectName(u"detectChartShelfPushbutton")

        self.detectChartToolshelfLayout.addWidget(self.detectChartShelfPushbutton)

        self.finalizeChartPushbutton = QPushButton(self.detectChartToolshelfFrame)
        self.finalizeChartPushbutton.setObjectName(u"finalizeChartPushbutton")

        self.detectChartToolshelfLayout.addWidget(self.finalizeChartPushbutton)

        self.horizontalSpacer_2 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.detectChartToolshelfLayout.addItem(self.horizontalSpacer_2)


        self.verticalLayout_5.addLayout(self.detectChartToolshelfLayout)


        self.verticalLayout_4.addWidget(self.detectChartToolshelfFrame)

        self.colourChartDebugToolsFrame = QFrame(self.centralwidget)
        self.colourChartDebugToolsFrame.setObjectName(u"colourChartDebugToolsFrame")
        self.colourChartDebugToolsFrame.setMinimumSize(QSize(0, 20))
        self.colourChartDebugToolsFrame.setFrameShape(QFrame.StyledPanel)
        self.colourChartDebugToolsFrame.setFrameShadow(QFrame.Raised)
        self.verticalLayout_6 = QVBoxLayout(self.colourChartDebugToolsFrame)
        self.verticalLayout_6.setObjectName(u"verticalLayout_6")
        self.verticalLayout_6.setContentsMargins(-1, 0, -1, 0)
        self.horizontalLayout = QHBoxLayout()
        self.horizontalLayout.setObjectName(u"horizontalLayout")
        self.horizontalSpacer_3 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_3)

        self.correctedImageRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.correctedImageRadioButton.setObjectName(u"correctedImageRadioButton")

        self.horizontalLayout.addWidget(self.correctedImageRadioButton)

        self.swatchOverlayRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.swatchOverlayRadioButton.setObjectName(u"swatchOverlayRadioButton")

        self.horizontalLayout.addWidget(self.swatchOverlayRadioButton)

        self.detectionDebugRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.detectionDebugRadioButton.setObjectName(u"detectionDebugRadioButton")

        self.horizontalLayout.addWidget(self.detectionDebugRadioButton)

        self.swatchAndClusterRadioButton = QRadioButton(self.colourChartDebugToolsFrame)
        self.swatchAndClusterRadioButton.setObjectName(u"swatchAndClusterRadioButton")

        self.horizontalLayout.addWidget(self.swatchAndClusterRadioButton)

        self.horizontalSpacer_4 = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)

        self.horizontalLayout.addItem(self.horizontalSpacer_4)


        self.verticalLayout_6.addLayout(self.horizontalLayout)


        self.verticalLayout_4.addWidget(self.colourChartDebugToolsFrame)

        self.imagePreviewGraphicsView = QGraphicsView(self.centralwidget)
        self.imagePreviewGraphicsView.setObjectName(u"imagePreviewGraphicsView")
        self.imagePreviewGraphicsView.setMinimumSize(QSize(500, 0))
        self.imagePreviewGraphicsView.setAutoFillBackground(False)

        self.verticalLayout_4.addWidget(self.imagePreviewGraphicsView)

        self.thumbnailPreviewFrame = QFrame(self.centralwidget)
        self.thumbnailPreviewFrame.setObjectName(u"thumbnailPreviewFrame")
        self.thumbnailPreviewFrame.setMinimumSize(QSize(0, 64))
        self.thumbnailPreviewFrame.setFrameShape(QFrame.StyledPanel)
        self.thumbnailPreviewFrame.setFrameShadow(QFrame.Raised)
        self.horizontalLayout_3 = QHBoxLayout(self.thumbnailPreviewFrame)
        self.horizontalLayout_3.setObjectName(u"horizontalLayout_3")
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.previousImagePushbutton = QPushButton(self.thumbnailPreviewFrame)
        self.previousImagePushbutton.setObjectName(u"previousImagePushbutton")
        self.previousImagePushbutton.setMinimumSize(QSize(0, 60))
        self.previousImagePushbutton.setMaximumSize(QSize(20, 16777215))

        self.horizontalLayout_3.addWidget(self.previousImagePushbutton)

        self.thumbnailPreviewDisplayFrame_holder = QFrame(self.thumbnailPreviewFrame)
        self.thumbnailPreviewDisplayFrame_holder.setObjectName(u"thumbnailPreviewDisplayFrame_holder")
        self.thumbnailPreviewDisplayFrame_holder.setFrameShape(QFrame.Box)
        self.thumbnailPreviewDisplayFrame_holder.setFrameShadow(QFrame.Raised)
        self.thumbnailPreviewDisplayFrame_holder.setMidLineWidth(1)
        self.horizontalLayout_5 = QHBoxLayout(self.thumbnailPreviewDisplayFrame_holder)
        self.horizontalLayout_5.setSpacing(0)
        self.horizontalLayout_5.setObjectName(u"horizontalLayout_5")
        self.horizontalLayout_5.setContentsMargins(0, 0, 0, 0)

        self.horizontalLayout_3.addWidget(self.thumbnailPreviewDisplayFrame_holder)

        self.nextImagePushbutton = QPushButton(self.thumbnailPreviewFrame)
        self.nextImagePushbutton.setObjectName(u"nextImagePushbutton")
        self.nextImagePushbutton.setMinimumSize(QSize(0, 60))
        self.nextImagePushbutton.setMaximumSize(QSize(20, 16777215))

        self.horizontalLayout_3.addWidget(self.nextImagePushbutton)


        self.verticalLayout_4.addWidget(self.thumbnailPreviewFrame)

        self.logOutputTextEdit = QTextEdit(self.centralwidget)
        self.logOutputTextEdit.setObjectName(u"logOutputTextEdit")
        self.logOutputTextEdit.setMaximumSize(QSize(16777215, 150))
        self.logOutputTextEdit.setUndoRedoEnabled(False)
        self.logOutputTextEdit.setReadOnly(True)

        self.verticalLayout_4.addWidget(self.logOutputTextEdit)


        self.gridLayout_2.addLayout(self.verticalLayout_4, 0, 3, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QStatusBar(MainWindow)
        self.statusbar.setObjectName(u"statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)

        QMetaObject.connectSlotsByName(MainWindow)
    # setupUi

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(QCoreApplication.translate("MainWindow", u"Scan Space Image Processor", None))
        self.useOriginalFilenamesCheckBox.setText(QCoreApplication.translate("MainWindow", u"Use Original Filenames", None))
        self.label_13.setText(QCoreApplication.translate("MainWindow", u"Shadow Limit", None))
        self.label_9.setText(QCoreApplication.translate("MainWindow", u"Average Exposure", None))
#if QT_CONFIG(tooltip)
        self.newImageNameLineEdit.setToolTip(QCoreApplication.translate("MainWindow", u"New Image Name, will be followed by _0001, numbers are based on padding length", None))
#endif // QT_CONFIG(tooltip)
        self.newImageNameLineEdit.setText(QCoreApplication.translate("MainWindow", u"Image", None))
        self.calculateAverageExposurePushbutton.setText(QCoreApplication.translate("MainWindow", u"Calculate Average Exposure", None))
        self.setSelectedAsChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Set Selected As Chart", None))
#if QT_CONFIG(tooltip)
        self.imagePaddingSpinBox.setToolTip(QCoreApplication.translate("MainWindow", u"Indicates how  many digits the image number will be", None))
#endif // QT_CONFIG(tooltip)
#if QT_CONFIG(whatsthis)
        self.imagePaddingSpinBox.setWhatsThis("")
#endif // QT_CONFIG(whatsthis)
        self.removeAverageDataPushbutton.setText(QCoreApplication.translate("MainWindow", u"Remove Average Data", None))
        self.label_7.setText(QCoreApplication.translate("MainWindow", u"Padding", None))
        self.label_5.setText(QCoreApplication.translate("MainWindow", u"Jpeg Quality", None))
        self.processImagesPushbutton.setText(QCoreApplication.translate("MainWindow", u"Process Images", None))
        self.previewChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Preview Chart Image", None))
        self.label_4.setText(QCoreApplication.translate("MainWindow", u"Threads", None))
        self.label_3.setText(QCoreApplication.translate("MainWindow", u"New Image Name", None))
        self.label_10.setText(QCoreApplication.translate("MainWindow", u"Colour Chart Tools", None))
        self.exportMaskedImagesCheckBox.setText(QCoreApplication.translate("MainWindow", u"Export Masked Images", None))
        self.manuallySelectChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Manually Select Chart", None))
#if QT_CONFIG(tooltip)
        self.displayDebugExposureDataCheckBox.setToolTip(QCoreApplication.translate("MainWindow", u"Display overlays over each image showing the hot and cold spots of each image", None))
#endif // QT_CONFIG(tooltip)
        self.displayDebugExposureDataCheckBox.setText(QCoreApplication.translate("MainWindow", u"Display debug data", None))
        self.label_11.setText(QCoreApplication.translate("MainWindow", u"Processing and Export", None))
        self.label_12.setText(QCoreApplication.translate("MainWindow", u"Highlight Limit", None))
        self.label_8.setText(QCoreApplication.translate("MainWindow", u"Chart Image", None))
        self.browseForChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_2.setText(QCoreApplication.translate("MainWindow", u"Raw Image Directory", None))
        self.browseForImagesPushbutton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label_6.setText(QCoreApplication.translate("MainWindow", u"Output Directory", None))
        self.browseoutputDirectoryPushbutton.setText(QCoreApplication.translate("MainWindow", u"Browse", None))
        self.label.setText(QCoreApplication.translate("MainWindow", u"Images", None))
        self.showOriginalImagePushbutton.setText(QCoreApplication.translate("MainWindow", u"Show Original Image", None))
#if QT_CONFIG(tooltip)
        self.flattenChartImagePushButton.setToolTip(QCoreApplication.translate("MainWindow", u"Select the four corners of the chart to straighten it.", None))
#endif // QT_CONFIG(tooltip)
        self.flattenChartImagePushButton.setText(QCoreApplication.translate("MainWindow", u"Flatten Chart Image", None))
        self.revertImagePushbutton.setText(QCoreApplication.translate("MainWindow", u"Revert Image", None))
        self.detectChartShelfPushbutton.setText(QCoreApplication.translate("MainWindow", u"Detect Chart", None))
        self.finalizeChartPushbutton.setText(QCoreApplication.translate("MainWindow", u"Finalize Chart", None))
        self.correctedImageRadioButton.setText(QCoreApplication.translate("MainWindow", u"Corrected Image", None))
        self.swatchOverlayRadioButton.setText(QCoreApplication.translate("MainWindow", u"Swatch Overlay", None))
        self.detectionDebugRadioButton.setText(QCoreApplication.translate("MainWindow", u"Detection Debug", None))
        self.swatchAndClusterRadioButton.setText(QCoreApplication.translate("MainWindow", u"Swatches And Clusters", None))
        self.previousImagePushbutton.setText(QCoreApplication.translate("MainWindow", u"<", None))
        self.nextImagePushbutton.setText(QCoreApplication.translate("MainWindow", u">", None))
        self.logOutputTextEdit.setDocumentTitle(QCoreApplication.translate("MainWindow", u"Log", None))
    # retranslateUi

