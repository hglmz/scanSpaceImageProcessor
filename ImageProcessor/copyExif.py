"""
EXIF Copy Module with Subprocess Safety

This module provides thread-safe EXIF metadata copying using a subprocess
to completely isolate OpenImageIO from the main process, avoiding all
threading and permission issues.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Any

from PySide6.QtCore import QThread, QObject, Signal


class ExifCopySignals(QObject):
    """Signals for EXIF copy operations."""
    finished = Signal(str, bool)  # output_path, success
    error = Signal(str, str)      # output_path, error_message
    log = Signal(str)             # log_message


class ExifCopyWorker(QThread):
    """
    QThread worker for copying EXIF metadata using subprocess isolation.
    
    This worker runs OpenImageIO in a completely separate subprocess to avoid
    any threading or permission issues that cause crashes in the main process.
    """
    
    def __init__(self, source_path: str, dest_path: str, image_data: Any = None):
        super().__init__()
        self.source_path = source_path
        self.dest_path = dest_path
        self.image_data = image_data  # Not used in subprocess approach
        self.signals = ExifCopySignals()
        self.success = False
        self.error_message = ""
        
    def run(self):
        """Main thread execution - copy EXIF metadata using subprocess."""
        try:
            self.signals.log.emit(f"[EXIF] Starting metadata copy: {os.path.basename(self.source_path)} -> {os.path.basename(self.dest_path)}")
            
            # Get path to subprocess script
            script_dir = Path(__file__).parent.parent
            subprocess_script = script_dir / "exif_copy_subprocess.py"
            
            if not subprocess_script.exists():
                raise FileNotFoundError(f"EXIF subprocess script not found: {subprocess_script}")
                
            # Run EXIF copy in isolated subprocess
            cmd = [sys.executable, str(subprocess_script), self.source_path, self.dest_path]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,  # 30 second timeout
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            if result.returncode == 0:
                self.success = True
                self.signals.log.emit(f"[EXIF] Metadata copied successfully: {os.path.basename(self.dest_path)}")
            else:
                self.success = False
                error_msg = result.stderr.strip() if result.stderr else "Unknown subprocess error"
                self.error_message = f"EXIF copy failed: {error_msg}"
                self.signals.log.emit(f"[EXIF] Copy failed: {error_msg}")
                self.signals.error.emit(self.dest_path, self.error_message)
                return
                
            # Signal completion
            self.signals.finished.emit(self.dest_path, self.success)
                    
        except subprocess.TimeoutExpired:
            self.error_message = "EXIF copy timeout (30s)"
            self.signals.error.emit(self.dest_path, self.error_message)
        except Exception as e:
            self.error_message = f"EXIF copy error: {str(e)}"
            self.signals.error.emit(self.dest_path, self.error_message)


class ExifCopyManager(QObject):
    """
    Manager class for handling EXIF copy operations.
    
    Provides a simple interface for copying EXIF metadata between images
    using the subprocess-based ExifCopyWorker in a dedicated Qt thread.
    """
    
    # Signals for operation completion
    copy_finished = Signal(str, bool)  # output_path, success
    copy_error = Signal(str, str)      # output_path, error_message
    log_message = Signal(str)          # log_message
    
    def __init__(self):
        super().__init__()
        self.active_workers = []
    
    def copy_exif_async(self, source_path: str, dest_path: str, image_data: Any = None):
        """
        Asynchronously copy EXIF metadata from source to destination.
        
        Args:
            source_path: Path to source image with metadata
            dest_path: Path to destination image to receive metadata
            image_data: Optional image data (not used in subprocess approach)
        """
        # Create worker thread
        worker = ExifCopyWorker(source_path, dest_path, image_data)
        
        # Connect signals
        worker.signals.finished.connect(self._on_worker_finished)
        worker.signals.error.connect(self._on_worker_error)
        worker.signals.log.connect(self._on_worker_log)
        
        # Store reference and start
        self.active_workers.append(worker)
        worker.start()
        
        return worker
    
    def _on_worker_finished(self, output_path: str, success: bool):
        """Handle worker completion."""
        self.copy_finished.emit(output_path, success)
        self._cleanup_worker()
    
    def _on_worker_error(self, output_path: str, error_message: str):
        """Handle worker error."""
        self.copy_error.emit(output_path, error_message)
        self._cleanup_worker()
    
    def _on_worker_log(self, message: str):
        """Handle worker log messages."""
        self.log_message.emit(message)
    
    def _cleanup_worker(self):
        """Clean up finished workers."""
        # Remove finished workers
        self.active_workers = [w for w in self.active_workers if w.isRunning()]
    
    def wait_for_completion(self, timeout_ms: int = 30000):
        """Wait for all active workers to complete."""
        for worker in self.active_workers[:]:  # Copy list to avoid modification during iteration
            if worker.isRunning():
                worker.wait(timeout_ms)
        self.active_workers.clear()


def copy_exif_sync(source_path: str, dest_path: str, timeout_ms: int = 30000) -> bool:
    """
    Synchronously copy EXIF metadata from source to destination.
    
    This is a convenience function for simple, blocking EXIF copy operations.
    
    Args:
        source_path: Path to source image with metadata
        dest_path: Path to destination image to receive metadata
        timeout_ms: Maximum time to wait for completion
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Create worker and run synchronously
    worker = ExifCopyWorker(source_path, dest_path)
    worker.start()
    
    # Wait for completion
    success = worker.wait(timeout_ms)
    if success:
        return worker.success
    else:
        print(f"[EXIF] Timeout waiting for metadata copy: {source_path} -> {dest_path}")
        return False