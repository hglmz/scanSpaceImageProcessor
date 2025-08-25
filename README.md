## Experimental: Negative Film Mode (Repurposing)

This repository has been repurposed/extended to support DSLR-scanned colour negatives while preserving the original photogrammetry workflow. The new Negative Film Mode is currently in progress and experimental.

- What it does: Automatically inverts DSLR-scanned film negatives using robust per-channel percentile normalization to approximate orange mask removal and dynamic range mapping. The positive result is then compatible with the existing ColorChecker-based colour correction workflow.
- Status: In progress (experimental). Behaviour, defaults, and UI may evolve.
- How to use:
  1. Open the app (`python image_space.py`).
  2. Open Settings and enable “Enable Negative Film Mode (auto invert)”.
  3. Load RAWs captured from negatives that include a ColorChecker in frame.
  4. If automatic chart detection struggles, use “Manually Select Chart” and mark 4 corners, then “Flatten” and “Finalize Chart”.
  5. Run batch processing as usual; colour correction will operate on the inverted (positive) image.
- Notes:
  - The auto-inversion is designed to be fast and robust for previews and batch runs. More advanced film workflows (e.g., base sampling UI, stock-specific profiles) may be added later.
  - Original features and workflows remain unchanged when this mode is disabled.

Türkçe Not:
- Bu proje, mevcut işlevler korunarak DSLR ile taranmış negatif filmler için “Negatif Film Modu” ile yeniden amaçlandırılmıştır. Özellik şu an deneyseldir ve geliştirme aşamasındadır.

---

# Image Space

A powerful desktop application for automated X-Rite/Calibrite colour checker calibration and batch colour correction of RAW camera images from photogrammetry datasets. Features distributed client/server processing, real-time image adjustments, and professional-grade colour correction workflows.

---

## Features

### **Color Correction & Chart Detection**

**Automatic Chart Detection**  
  Detects X-Rite ColorChecker swatches in an image and extracts mean patch colours.
  Uses the brilliant Colour Checker detection from KelSolaar
  https://github.com/colour-science/colour-checker-detection

**Manual Chart Selection & Flatten**  
  - Draw a crop box around the chart  
  - Click its four corners to warp/flatten perspective  
  - Extract swatches precisely, even on angled shots

**Batch Colour Correction**  
  Apply a reference chart profile to an entire folder of RAW images, with configurable threading and memory management.

### **Distributed Processing**

**Client/Server Architecture**  
  Scale your processing across multiple machines with robust distributed computing:
  - **Server GUI**: Built-in processing server with real-time client monitoring
  - **Headless Clients**: Deploy processing clients on multiple machines
  - **Automatic Load Balancing**: Jobs distributed based on client capacity and memory
  - **Fault Tolerance**: Automatic reconnection and job reassignment
  - **Memory Aware**: Prevents system overload with intelligent memory estimation

**Network Features**  
  - TCP/IP communication with JSON protocol
  - Heartbeat monitoring and dead client cleanup
  - UNC path support for seamless network file access
  - Real-time progress reporting and statistics

### **Real-Time Image Adjustments**

**Professional Image Enhancement**  
  - **Shadow/Highlight Recovery**: RawTherapee-style LAB color space processing
  - **Denoise**: AI-powered noise reduction with configurable strength
  - **Sharpen**: Unsharp mask with radius, amount, and threshold controls
  - **White Balance**: Automatic or manual color temperature adjustment
  - **Real-Time Preview**: Instant feedback with background RAW loading

**Interactive Preview System**  
  - Animated loading spinner for RAW processing feedback
  - Live adjustment sliders with immediate visual feedback
  - Exposure normalization preview with hot-spot detection
  - Debug overlays for exposure analysis

### **Formats & Export Options**

**Supported RAW Formats**  
  - **.nef** Nikon NEF Raw Format
  - **.cr2** Canon CR2 Raw Format
  - **.cr3** Canon CR3 Raw Format
  - **.dng** Digital Negative files (Often from Drones)
  - **.arw** Sony ARW format

**Export Formats**  
  - **JPEG** (`.jpg`) with adjustable quality and color space
  - **PNG** (`.png`) with transparency support
  - **TIFF** (`.tiff`), 8-bit or 16-bit with compression options
  - **OpenEXR** (`.exr`) in any OIIO-supported colourspace, float precision

**Advanced Export Features**  
  - **Custom File Naming**: Flexible naming schemas with variables
  - **Metadata Preservation**: Carries over all original EXIF metadata
  - **Batch Processing**: Multi-threaded processing with progress tracking
  - **Memory Optimization**: Intelligent memory management for large datasets

### **Workflow Features**

**Exposure Normalization**  
  Compute and apply average-exposure multipliers across your dataset, with hot-spot/shadow clipping.
  Useful for datasets shot outdoors or using a flash from varying distances

**Debug & Analysis Tools**  
  Preview intermediate results:  
  - Corrected patch image  
  - Swatch overlay and color analysis
  - Swatches & clusters visualization
  - Segmentation debug information
  - Exposure histograms and statistics

---

## Installation

**Requirements**
- Python 3.10 or higher
- Windows, macOS, or Linux
- For distributed processing: Network access between machines

### **Quick Start**

1. **Download Image Space**
   ```bash
   # Option 1: Git clone
   git clone https://github.com/ErikScansNz/scanSpaceImageProcessor.git
   cd scanSpaceImageProcessor
   
   # Option 2: Download ZIP
   # Download from: https://github.com/ErikScansNz/scanSpaceImageProcessor/archive/refs/heads/main.zip
   ```

2. **Install Dependencies**

```
Windows:
bash
python setup.py
``` 

```
Mac/Linux:
bash
pip install -r requirements.txt
```

3. **Launch Application**
   ```bash
   # Main GUI application (includes built-in server)
   python image_space.py
   ```

### **Distributed Processing Setup**

For scaling across multiple machines:

**Server Machine (Main Application)**
```bash
# Launch with built-in server GUI
python standalone_server.py
# Server starts automatically on port 8888
```

**Client Machines (Processing Workers)**
```bash
# Basic headless client
python run_headless_processing_client.py --host 192.168.1.100 --port 8888 --threads 4

# Client with custom restart settings
python hrun_headless_processing_client.py --host SERVER_IP --port 8888 --threads 2 --maxRestarts 5

# Auto-restart capability for production deployments
python run_headless_processing_client.py --host SERVER_IP --port 8888 --threads 4
```

**Client Parameters:**
- `--host`: Server IP address
- `--port`: Server port (default: 8888)
- `--threads`: Number of processing threads per client
- `--maxRestarts`: Maximum automatic restarts on failure

---

## Usage Guide

### **Basic Workflow**

1. **Import Images**
   - Click "Select Directory" to load RAW images
   - Images are automatically grouped by subfolder or filename prefix
   - Supported formats: NEF, CR2, CR3, DNG, ARW

2. **Color Chart Setup**
   - **Automatic Detection**: Click "Detect Chart" for automatic ColorChecker detection
   - **Manual Selection**: Use "Manually Select Chart" for precise 4-corner selection
   - Charts can be assigned per group or globally

3. **Real-Time Adjustments** 
   - **Shadow/Highlight**: Adjust exposure recovery with LAB color processing
   - **Denoise**: Apply AI-powered noise reduction (0-100% strength)
   - **Sharpen**: Fine-tune with radius, amount, and threshold controls
   - **White Balance**: Enable/disable automatic color temperature correction
   - All adjustments show real-time preview

4. **Processing Options**
   - **Local Processing**: Multi-threaded processing on current machine
   - **Network Processing**: Distribute jobs across connected clients
   - Monitor progress with real-time status updates

5. **Export Configuration**
   - Choose output format (JPEG, PNG, TIFF, OpenEXR)
   - Set custom file naming schemas
   - Configure quality and color space settings

### **Advanced Features**

**Distributed Processing Workflow**
1. Launch main application on server machine
2. Deploy headless clients on worker machines
3. Clients automatically register and report capacity
4. Jobs are intelligently distributed based on memory and CPU availability
5. Real-time monitoring shows client status and processing statistics

**Custom File Naming**
Use flexible naming schemas with variables:
- `[r]`: Root folder name
- `[s]`: Subfolder name
- `[o]`: Original filename
- `[n]`: Image number with padding
- `[e]`: File extension
- Example: `[r]_[s]_[n4][e]` → `dataset_cross_0001.jpg`

**Exposure Normalization**
- Automatically compute exposure differences across dataset
- Apply average exposure multipliers to normalize lighting
- Useful for outdoor datasets or varying flash distances

**Debug & Analysis**
- View intermediate processing steps
- Analyze color correction matrices
- Monitor exposure histograms and statistics
- Visualize chart detection and segmentation results

---

## Troubleshooting

**Network Processing Issues**
- Ensure all machines can access shared storage via UNC paths
- Check firewall settings allow TCP communication on port 8888
- Verify client machines have sufficient memory for processing

**Performance Optimization**
- Adjust thread count based on CPU cores and available memory
- Use SSD storage for improved RAW loading performance
- Monitor memory usage during large batch processing

**Color Correction Problems**
 - Ensure ColorChecker chart is well-lit and visible
 - Try manual chart selection for difficult lighting conditions or charts on odd angles
 - Check that all images in a group use the same lighting setup
