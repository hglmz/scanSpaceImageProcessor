#!/usr/bin/env python3
"""
Simple Headless Processing Client Launcher

This script launches the headless image processing client by running the
networkProcessor.py module directly. This approach ensures OIIO operations
work correctly without Qt application context interference.

Usage:
    python run_headless_processing_client.py --host 192.168.1.100 --port 8888 --threads 4
    python run_headless_processing_client.py --validateConnection --threads 2
    python run_headless_processing_client.py  # Interactive setup
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Launch the headless processing client."""
    
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Path to the networkProcessor.py module
    network_processor_path = script_dir / "ImageProcessor" / "networkProcessor.py"
    
    # Verify the networkProcessor.py exists
    if not network_processor_path.exists():
        print("ERROR: networkProcessor.py not found!")
        print(f"Expected location: {network_processor_path}")
        print("Please ensure you're running from the correct directory.")
        return 1
    
    # Verify we're in the right directory structure
    if not (script_dir / "ImageProcessor" / "imageProcessorWorker.py").exists():
        print("ERROR: imageProcessorWorker.py not found!")
        print("This suggests you're not running from the correct project root.")
        print(f"Current directory: {script_dir}")
        return 1
    
    print("Scan Space Image Processor - Headless Client Launcher")
    print("=" * 55)
    print(f"Project root: {script_dir}")
    print(f"Launching: {network_processor_path}")
    print()
    
    # Change to the ImageProcessor directory for correct imports
    original_cwd = os.getcwd()
    image_processor_dir = script_dir / "ImageProcessor"
    
    try:
        os.chdir(image_processor_dir)
        print(f"Changed working directory to: {image_processor_dir}")
        
        # Prepare the command to run networkProcessor.py
        cmd = [sys.executable, "networkProcessor.py"] + sys.argv[1:]
        
        print(f"Executing: {' '.join(cmd)}")
        print("-" * 55)
        print()
        
        # Run the networkProcessor.py with all arguments passed through
        result = subprocess.run(cmd, cwd=image_processor_dir)
        return result.returncode
        
    except KeyboardInterrupt:
        print("\nLauncher interrupted by user")
        return 130  # Standard exit code for Ctrl+C
        
    except Exception as e:
        print(f"ERROR: Failed to launch client: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Restore original working directory
        try:
            os.chdir(original_cwd)
        except Exception:
            pass


if __name__ == "__main__":
    # Print help information if no arguments provided
    if len(sys.argv) == 1:
        print("Scan Space Image Processor - Headless Client Launcher")
        print("=" * 55)
        print()
        print("Usage:")
        print("  python run_headless_processing_client.py --host SERVER_IP --port PORT --threads N")
        print()
        print("Examples:")
        print("  python run_headless_processing_client.py --host 192.168.1.100 --port 8888 --threads 4")
        print("  python run_headless_processing_client.py --host 192.168.1.100 --threads 2 --delayLoad 0.5")
        print("  python run_headless_processing_client.py --validateConnection")
        print("  python run_headless_processing_client.py  # Interactive setup")
        print()
        print("Available options:")
        print("  --host HOST              Server host address")
        print("  --port PORT              Server port (default: 8888)")
        print("  --threads N              Number of processing threads")
        print("  --delayLoad SECONDS      Delay loading of images by X seconds")
        print("  --maxRetries N           Maximum connection retry attempts (default: 5)")
        print("  --retryDelay SECONDS     Delay between retry attempts (default: 5.0)")
        print("  --validateConnection     Validate server connection before starting")
        print()
        print("For more options, run:")
        print("  python run_headless_processing_client.py --help")
        print()
        
        # Ask if user wants to proceed with interactive setup
        try:
            proceed = input("Would you like to start with interactive setup? [y/N]: ").strip().lower()
            if proceed in ['y', 'yes']:
                # Don't add any arguments - networkProcessor.py will prompt for missing settings
                pass
            else:
                print("Exiting...")
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            print("\nExiting...")
            sys.exit(0)
    
    sys.exit(main())