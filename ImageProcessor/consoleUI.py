#!/usr/bin/env python3
"""
Console UI for Scan Space Image Processor Network Components

This module provides a beautiful console interface with status bar and colored logging
for the headless client and other network processing components.
"""

import os
import sys
import time
import shutil
import threading
from datetime import datetime
from queue import Queue, Empty
from typing import Optional, Callable


class ConsoleUI:
    """Beautiful console interface with status bar and colored logging."""
    
    def __init__(self, enable_colors=True):
        self.enable_colors = enable_colors and self._supports_color()
        self.terminal_width = shutil.get_terminal_size().columns
        self.terminal_height = shutil.get_terminal_size().lines
        self.status_bar_enabled = False
        self.status_lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.current_jobs = {}
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'processing_time': 0.0,
            'connection_status': 'Disconnected',
            'server_host': '',
            'client_id': '',
            'thread_count': 0
        }
        
        # Interactive command system
        self.command_thread = None
        self.command_running = False
        self.client_instance = None  # Will be set by create_headless_client
        self.command_queue = Queue()  # For sending commands to main thread

        # Status line placement
        self.log_line1 = self.terminal_height - 5
        self.input_line1 = self.terminal_height - 4
        self.input_line2 = self.terminal_height - 3
        self.status_line1 = self.terminal_height - 2
        self.status_line2 = self.terminal_height - 1
        self.status_line3 = self.terminal_height
        
        # ANSI color codes
        self.colors = {
            'reset': '\033[0m',
            'bold': '\033[1m',
            'dim': '\033[2m',
            'red': '\033[31m',
            'green': '\033[32m',
            'yellow': '\033[33m',
            'blue': '\033[34m',
            'magenta': '\033[35m',
            'cyan': '\033[36m',
            'white': '\033[37m',
            'bg_blue': '\033[44m',
            'bg_green': '\033[42m',
            'bg_red': '\033[41m'
        }
        
        if not self.enable_colors:
            self.colors = {k: '' for k in self.colors.keys()}
    
    def _supports_color(self):
        """Check if terminal supports color output."""
        try:
            import colorama
            colorama.init()
            return True
        except ImportError:
            pass
        
        # Check environment variables
        return (
            hasattr(sys.stdout, 'isatty') and sys.stdout.isatty() and
            os.environ.get('TERM', '').lower() not in ['', 'dumb']
        )
    
    def enable_status_bar(self):
        """Enable the status bar display with log area limited to top 12 lines."""
        self.status_bar_enabled = True
        # Clear screen and position cursor
        if self.enable_colors:
            print('\033[2J\033[H', end='', flush=True)
            # Set scroll region to top 12 lines only
            print(f'\033[1;{self.log_line1}r', end='', flush=True)  # Set scroll region to the defined height
            # Position cursor at top of scroll region
            print('\033[1;1H', end='', flush=True)

    def disable_status_bar(self):
        """Disable the status bar display."""
        self.status_bar_enabled = False
        if self.enable_colors:
            # Reset scroll region to full screen
            print('\033[r', end='', flush=True)  # Reset scroll region
            print('\033[?30h', end='', flush=True)  # Show cursor
            # Clear the status bar and input area
            terminal_height = shutil.get_terminal_size().lines
            for i in range(5):  # Clear input area (lines 14-15) and status bar (bottom 3)
                print(f'\033[{terminal_height-4+i};1H\033[K', end='')
            # Also clear lines 14-15 for input area
            print('\033[24;1H\033[K', end='')
            print('\033[25;1H\033[K', end='')
            print('', flush=True)
    
    def update_stats(self, **kwargs):
        """Update statistics for the status bar."""
        with self.status_lock:
            self.stats.update(kwargs)
    
    def add_job(self, job_id: str, filename: str):
        """Add a job to the current processing list."""
        with self.status_lock:
            self.current_jobs[job_id] = {
                'filename': filename,
                'start_time': time.time(),
                'status': 'Processing'
            }
    
    def complete_job(self, job_id: str, success: bool = True, processing_time: float = 0):
        """Mark a job as completed."""
        with self.status_lock:
            if job_id in self.current_jobs:
                del self.current_jobs[job_id]
            
            if success:
                self.stats['total_processed'] += 1
            else:
                self.stats['total_failed'] += 1
            
            if processing_time > 0:
                self.stats['processing_time'] += processing_time
    
    def log(self, message: str, level: str = 'INFO'):
        """Log a message with beautiful formatting."""
        with self.log_lock:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Color mapping for log levels
            level_colors = {
                'DEBUG': self.colors['dim'],
                'INFO': self.colors['cyan'],
                'WARNING': self.colors['yellow'],
                'ERROR': self.colors['red'],
                'SUCCESS': self.colors['green']
            }
            
            color = level_colors.get(level, self.colors['white'])
            reset = self.colors['reset']
            
            # Format the log message
            formatted_msg = f"{self.colors['dim']}[{timestamp}]{reset} {color}{level:>7}{reset} {message}"

            # Save cursor position
            print('\033[s', end='')

            # Move to bottom of log area
            print(f'\033[{self.log_line1};1H', end='')

            # Print the message (this will scroll within lines 1-12)
            print(formatted_msg)

            # Restore cursor position
            print('\033[u', end='', flush=True)

            # Redraw status bar and input area
        self._draw_status_bar()

    
    def _draw_status_bar(self):
        """Draw the status bar at the bottom of the terminal."""
        if not self.status_bar_enabled or not self.enable_colors:
            return
        
        with self.status_lock:
            # Get actual terminal dimensions
            terminal_height = shutil.get_terminal_size().lines
            terminal_width = shutil.get_terminal_size().columns
            
            # Calculate positions: input area at line 14-15, status bar at bottom 3 lines
            self.input_line1 = terminal_height - 4
            self.input_line2 = terminal_width - 3
            self.status_line1 = terminal_height - 2
            self.status_line2 = terminal_height - 1
            self.status_line3 = terminal_height
            
            # Save cursor position
            print('\033[s', end='')

            # Move to bottom of screen
            print('\033[9999;1H', end='')

            # Create status bar content
            connection_color = self.colors['bg_green'] if self.stats['connection_status'] == 'Connected' else self.colors['bg_red']
            
            # Line 1: Connection status and server info
            line1_parts = []
            if self.stats['connection_status'] == 'Connected':
                line1_parts.append(f"{connection_color}{self.colors['white']} CONNECTED {self.colors['reset']}")
                if self.stats['server_host']:
                    line1_parts.append(f"{self.colors['cyan']}{self.stats['server_host']}{self.colors['reset']}")
                if self.stats['client_id']:
                    line1_parts.append(f"{self.colors['dim']}ID: {self.stats['client_id'][-8:]}{self.colors['reset']}")
            else:
                line1_parts.append(f"{connection_color}{self.colors['white']} DISCONNECTED {self.colors['reset']}")
            
            line1 = " | ".join(line1_parts)
            
            # Line 2: Processing stats
            active_jobs = len(self.current_jobs)
            avg_time = self.stats['processing_time'] / max(1, self.stats['total_processed']) if self.stats['total_processed'] > 0 else 0
            
            line2_parts = [
                f"{self.colors['green']}OK {self.stats['total_processed']}{self.colors['reset']}",
                f"{self.colors['red']}ERR {self.stats['total_failed']}{self.colors['reset']}",
                f"{self.colors['yellow']}JOBS {active_jobs}/{self.stats['thread_count']}{self.colors['reset']}",
            ]
            
            if avg_time > 0:
                line2_parts.append(f"{self.colors['blue']}AVG {avg_time:.1f}s/img{self.colors['reset']}")
            
            line2 = " | ".join(line2_parts)
            
            # Line 3: Current jobs
            line3 = ""
            if self.current_jobs:
                job_displays = []
                for job_id, job_info in list(self.current_jobs.items())[:3]:  # Show max 3 jobs
                    filename = job_info['filename']
                    if len(filename) > 20:
                        filename = filename[:17] + "..."
                    elapsed = time.time() - job_info['start_time']
                    job_displays.append(f"{self.colors['magenta']}{filename}{self.colors['reset']} {self.colors['dim']}({elapsed:.0f}s){self.colors['reset']}")
                
                if len(self.current_jobs) > 3:
                    job_displays.append(f"{self.colors['dim']}+{len(self.current_jobs)-3} more{self.colors['reset']}")
                
                line3 = " | ".join(job_displays)
            else:
                line3 = f"{self.colors['dim']}        Waiting for jobs...{self.colors['reset']}"
            
            # Clear and draw the status bar (3 lines)
            for i in range(6):
                print(f'\033[{9999-2+i};1H\033[K', end='')

            # Line 1
            print(f'\033[{self.status_line1};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.status_line1};2H{line1}', end='')
            
            # Line 2
            print(f'\033[{self.status_line2};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.status_line2};2H{line2}', end='')
            
            # Line 3
            print(f'\033[{self.status_line3};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.status_line3};2H{line3}', end='')
            
            # Restore cursor position
            print('\033[u', end='', flush=True)

            # Get terminal dimensions
            terminal_width = shutil.get_terminal_size().columns
            # Save cursor position
            print('\033[s', end='')
            
            # Clear and draw input area background
            # Line 1: Command prompt label
            prompt_text = f"{self.colors['bold']}{self.colors['cyan']}Commands:{self.colors['reset']} help, status, threads N, stats, clear, reconnect, quit"
            print(f'\033[{self.input_line1};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.input_line1};2H{prompt_text}', end='')
            
            # Line 2: Input prompt
            input_prompt = f"{self.colors['bold']}{self.colors['white']}> {self.colors['reset']}"
            print(f'\033[{self.input_line2};1H\033{" " * terminal_width}{self.colors["reset"]}', end='')
            print(f'\033[{self.input_line2};2H{input_prompt}', end='')
            
            # Restore cursor position
            print('\033[u', end='', flush=True)
    
    def start_command_interface(self, client_instance):
        """Start the interactive command interface in a separate thread."""
        self.client_instance = client_instance
        self.command_running = True
        self.command_thread = threading.Thread(target=self._command_loop, daemon=True)
        self.command_thread.start()
        self.log("Interactive commands enabled. Use input area below for commands.", "INFO")
    
    def stop_command_interface(self):
        """Stop the interactive command interface."""
        self.command_running = False
        if self.command_thread:
            self.command_thread.join(timeout=1)
    
    def _command_loop(self):
        """Main loop for processing interactive commands."""
        while self.command_running:
            try:
                # Position cursor in input area (line 15, after the prompt)
                if self.status_bar_enabled and self.enable_colors:
                    print(f'\033[{self.input_line2};4H', end='', flush=True)  # Position after "> "
                
                # Get command input (this will block)
                command = input().strip().lower()
                
                # Clear the input line after command is entered
                if self.status_bar_enabled and self.enable_colors:
                    print(f'\033[{self.input_line2};4H\033[K', end='', flush=True)
                
                if not command:
                    continue
                
                # Queue the command for processing in the main thread
                self.command_queue.put(command)
                
            except (EOFError, KeyboardInterrupt):
                # User pressed Ctrl+C or EOF, stop command interface
                self.command_running = False
                break
            except Exception as e:
                self.log(f"Command input error: {e}", "ERROR")
    
    def process_command(self, command: str):
        """Process a user command."""
        parts = command.split()
        if not parts:
            return
        
        cmd = parts[0]
        args = parts[1:] if len(parts) > 1 else []
        
        try:
            if cmd == 'help' or cmd == 'h':
                self._show_help()
            elif cmd == 'status' or cmd == 's':
                self._show_detailed_status()
            elif cmd == 'threads' or cmd == 't':
                self._handle_threads_command(args)
            elif cmd == 'stats' or cmd == 'st':
                self._show_stats()
            elif cmd == 'clear' or cmd == 'c':
                self._clear_screen()
            elif cmd == 'reset':
                self._reset_stats()
            elif cmd == 'reconnect' or cmd == 'r':
                self._request_reconnect()
            elif cmd == 'quit' or cmd == 'q' or cmd == 'exit':
                self._request_quit()
            else:
                self.log(f"Unknown command: '{cmd}'. Type 'help' for available commands.", "WARNING")
        except Exception as e:
            self.log(f"Command error: {e}", "ERROR")
    
    def _show_help(self):
        """Show available commands."""
        help_text = [
            "Available Commands:",
            "  help, h          - Show this help message",
            "  status, s        - Show detailed client status", 
            "  threads N, t N   - Set thread count to N (1-32)",
            "  stats, st        - Show processing statistics",
            "  clear, c         - Clear the screen",
            "  reset            - Reset processing statistics",
            "  reconnect, r     - Request reconnection to server",
            "  quit, q, exit    - Gracefully shutdown client",
            "",
            "Examples:",
            "  threads 4        - Set to 4 processing threads",
            "  t 2              - Set to 2 processing threads"
        ]
        
        for line in help_text:
            self.log(line, "INFO")
    
    def _show_detailed_status(self):
        """Show detailed client status."""
        with self.status_lock:
            self.log("=== Client Status ===", "INFO")
            self.log(f"Connection: {self.stats['connection_status']}", "INFO")
            self.log(f"Server: {self.stats['server_host']}", "INFO")
            self.log(f"Client ID: {self.stats['client_id']}", "INFO")
            self.log(f"Thread Count: {self.stats['thread_count']}", "INFO")
            self.log(f"Active Jobs: {len(self.current_jobs)}/{self.stats['thread_count']}", "INFO")
            
            if self.current_jobs:
                self.log("Current Jobs:", "INFO")
                for job_id, job_info in list(self.current_jobs.items()):
                    elapsed = time.time() - job_info['start_time']
                    self.log(f"  {job_info['filename']} ({elapsed:.0f}s)", "INFO")
    
    def _handle_threads_command(self, args):
        """Handle thread count adjustment command."""
        if not args:
            self.log(f"Current thread count: {self.stats['thread_count']}", "INFO")
            self.log("Usage: threads <number> or t <number>", "INFO")
            return
        
        try:
            new_count = int(args[0])
            if new_count < 1 or new_count > 32:
                self.log("Thread count must be between 1 and 32", "ERROR")
                return
            
            if self.client_instance:
                old_count = self.client_instance.thread_count
                self.client_instance._handle_thread_count_change(new_count)
                self.update_stats(thread_count=new_count)
                self.log(f"Thread count changed from {old_count} to {new_count}", "SUCCESS")
            else:
                self.log("Client instance not available", "ERROR")
                
        except ValueError:
            self.log("Invalid thread count. Please enter a number between 1 and 32.", "ERROR")
    
    def _show_stats(self):
        """Show processing statistics."""
        with self.status_lock:
            self.log("=== Processing Statistics ===", "INFO")
            self.log(f"Total Processed: {self.stats['total_processed']}", "SUCCESS")
            self.log(f"Total Failed: {self.stats['total_failed']}", "ERROR")
            
            if self.stats['total_processed'] > 0:
                avg_time = self.stats['processing_time'] / self.stats['total_processed']
                self.log(f"Average Processing Time: {avg_time:.2f}s per image", "INFO")
                success_rate = (self.stats['total_processed'] / (self.stats['total_processed'] + self.stats['total_failed'])) * 100
                self.log(f"Success Rate: {success_rate:.1f}%", "INFO")
    
    def _clear_screen(self):
        """Clear the screen and redraw status bar."""
        if self.enable_colors:
            print('\033[2J\033[H', end='', flush=True)
            if self.status_bar_enabled:
                # Reset scroll region
                terminal_height = shutil.get_terminal_size().lines
                scroll_bottom = terminal_height - 3
                print(f'\033[1;{scroll_bottom}r', end='', flush=True)
                self._draw_status_bar()
        self.log("Screen cleared", "INFO")
    
    def _reset_stats(self):
        """Reset processing statistics."""
        with self.status_lock:
            self.stats['total_processed'] = 0
            self.stats['total_failed'] = 0
            self.stats['processing_time'] = 0.0
        self.log("Statistics reset", "SUCCESS")
    
    def _request_reconnect(self):
        """Request reconnection to server."""
        self.command_queue.put('__reconnect__')
        self.log("Reconnection requested...", "INFO")
    
    def _request_quit(self):
        """Request graceful shutdown."""
        self.command_queue.put('__quit__')
        self.log("Graceful shutdown requested...", "WARNING")


def console_log_callback(message: str):
    """Log callback for console output with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}")