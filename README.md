# Scan Space Image Processor

A Qt/PySide6 desktop application for automated Xrite/Calibrite colour checker calibration and batch colour correction of RAW camera images from photogrammetry datasets.

---

## Features

- **Automatic Chart Detection**  
  Detects X-Rite ColorChecker swatches in an image and extracts mean patch colours.
- Uses the brilliant Colour Checker detection from KelSolaar
- https://github.com/colour-science/colour-checker-detection

- **Manual Chart Selection & Flatten**  
  • Draw a crop box around the chart  
  • Click its four corners to warp/flatten perspective  
  • Extract swatches precisely, even on angled shots

- **Batch Colour Correction**  
  Apply a reference chart profile to an entire folder of RAW images, in parallel.

- **Formats & Bit-Depth**  
  Imports the following formats:
  - **.nef** Nikon .NEF raw Format
  - **.cr2** Canon CR2 Raw Format
  - **.cr3** Canon CR3 Raw Format
  - **.dng** Digital Negative files (Often from Drones)
  - **.arw** Sony ARW format

  Export corrected images to:
  - **JPEG** (`.jpg`) with adjustable quality  
  - **PNG** (`.png`)  
  - **TIFF** (`.tiff`), 8-bit or 16-bit  
  - **OpenEXR** (`.exr`) in any OIIO-supported colourspace, float

- **Metadata Preservation**  
  Carries over all original EXIF metadata into your output files.

- **Exposure Normalization**  
  Compute and apply average-exposure multipliers across your dataset, with hot-spot/shadow clipping.
  - Useful for datasets shot outdoors or using a flash from varying distances

- **Debug Overlays**  
  Preview intermediate results:  
  - Corrected patch image  
  - Swatch overlay  
  - Swatches & clusters  
  - Segmentation debug

---

## 📦 Installation

- Requires Python 3.10

1. **Clone this repository**  
   ```bash
   git clone https://github.com/your-org/scan-space-image-processor.git
   cd scan-space-image-processor

2. **Install package requirements**
   ```
    pip install \
      PySide6 \
      rawpy \
      imageio \
      colour-science \
      colour-checker-detection \
      oiio-python \
      psutil \
      pillow \
      tifffile

3. **Launch**
    ```
   python scan_space_image_processor.py