"""
HTTP Client for Standalone Processing Server

This module provides an HTTP client interface for communicating with the
standalone processing server. It handles job submission, monitoring, and
status queries through REST API calls.
"""

import json
import logging
import requests
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass


@dataclass
class ServerConfig:
    """Configuration for server connection."""
    host: str = "localhost"
    api_port: int = 8889
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    @property
    def base_url(self) -> str:
        """Get the base URL for API calls."""
        return f"http://{self.host}:{self.api_port}"


class ServerConnectionError(Exception):
    """Raised when server connection fails."""
    pass


class ServerAPIError(Exception):
    """Raised when server API returns an error."""
    pass


class ProcessingServerClient:
    """HTTP client for the standalone processing server."""
    
    def __init__(self, config: ServerConfig, log_callback: Optional[Callable[[str], None]] = None):
        self.config = config
        self.log_callback = log_callback or self._default_log
        self.session = requests.Session()
        
        # Setup session with reasonable defaults
        self.session.timeout = config.timeout
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'ScanSpace-ImageProcessor/1.0'
        })
        
        self._last_status = None
        self._connection_verified = False
    
    def _default_log(self, message: str):
        """Default logging implementation."""
        print(f"[ServerClient] {message}")
    
    def _log_info(self, message: str):
        """Log info message."""
        self.log_callback(f"[INFO] {message}")
    
    def _log_error(self, message: str):
        """Log error message."""
        self.log_callback(f"[ERROR] {message}")
    
    def _log_warning(self, message: str):
        """Log warning message."""
        self.log_callback(f"[WARNING] {message}")
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                      retry: bool = True) -> Dict[str, Any]:
        """
        Make HTTP request to server with error handling and retry logic.
        
        Args:
            method: HTTP method ('GET', 'POST', etc.)
            endpoint: API endpoint (e.g., '/api/status')
            data: Request data for POST requests
            retry: Whether to retry on failure
            
        Returns:
            Response data as dictionary
            
        Raises:
            ServerConnectionError: When connection fails
            ServerAPIError: When API returns error
        """
        url = f"{self.config.base_url}{endpoint}"
        attempts = self.config.retry_attempts if retry else 1
        
        for attempt in range(attempts):
            try:
                if method.upper() == 'GET':
                    response = self.session.get(url)
                elif method.upper() == 'POST':
                    response = self.session.post(url, json=data)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check for HTTP errors
                if response.status_code == 404:
                    raise ServerAPIError(f"Endpoint not found: {endpoint}")
                elif response.status_code >= 400:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('error', f'HTTP {response.status_code}')
                    except:
                        error_msg = f"HTTP {response.status_code}: {response.text}"
                    raise ServerAPIError(error_msg)
                
                # Parse JSON response
                try:
                    return response.json()
                except json.JSONDecodeError:
                    raise ServerAPIError(f"Invalid JSON response from server")
                
            except requests.exceptions.ConnectionError as e:
                if attempt == attempts - 1:  # Last attempt
                    raise ServerConnectionError(f"Cannot connect to server at {url}: {e}")
                else:
                    self._log_warning(f"Connection attempt {attempt + 1}/{attempts} failed, retrying...")
                    time.sleep(self.config.retry_delay)
            
            except requests.exceptions.Timeout as e:
                if attempt == attempts - 1:  # Last attempt
                    raise ServerConnectionError(f"Timeout connecting to server at {url}: {e}")
                else:
                    self._log_warning(f"Timeout attempt {attempt + 1}/{attempts}, retrying...")
                    time.sleep(self.config.retry_delay)
            
            except (ServerAPIError, ValueError) as e:
                # Don't retry for API errors or invalid methods
                raise e
            
            except Exception as e:
                if attempt == attempts - 1:  # Last attempt
                    raise ServerConnectionError(f"Unexpected error: {e}")
                else:
                    self._log_warning(f"Unexpected error attempt {attempt + 1}/{attempts}: {e}")
                    time.sleep(self.config.retry_delay)
    
    def verify_connection(self) -> bool:
        """
        Verify connection to the server.
        
        Returns:
            True if server is reachable and responding
        """
        try:
            status = self.get_server_status()
            self._connection_verified = True
            self._log_info(f"Connected to server at {self.config.base_url}")
            return True
        except (ServerConnectionError, ServerAPIError) as e:
            self._connection_verified = False
            self._log_error(f"Failed to connect to server: {e}")
            return False
    
    def get_server_status(self) -> Dict[str, Any]:
        """
        Get current server status.
        
        Returns:
            Server status information
        """
        status = self._make_request('GET', '/api/status')
        self._last_status = status
        return status
    
    def get_clients_info(self) -> Dict[str, Any]:
        """
        Get information about connected clients.
        
        Returns:
            Clients information including status and capacity
        """
        return self._make_request('GET', '/api/clients')
    
    def get_jobs_overview(self) -> Dict[str, Any]:
        """
        Get overview of job queues and status.
        
        Returns:
            Jobs overview with queue sizes and active jobs
        """
        return self._make_request('GET', '/api/jobs')
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get status of a specific job.
        
        Args:
            job_id: ID of the job to query
            
        Returns:
            Job status information
        """
        return self._make_request('GET', f'/api/job/{job_id}')
    
    def submit_project_jobs(self, project_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Submit processing jobs from project data.
        
        Args:
            project_data: Complete project data structure (from export_current_project)
            
        Returns:
            Job submission result with created job IDs
        """
        try:
            result = self._make_request('POST', '/api/jobs/submit', data=project_data)
            self._log_info(f"Successfully submitted {result.get('jobs_created', 0)} jobs to server")
            return result
        except (ServerConnectionError, ServerAPIError) as e:
            self._log_error(f"Failed to submit jobs: {e}")
            raise
    
    def clear_job_queues(self) -> Dict[str, Any]:
        """
        Clear all job queues on the server.
        
        Returns:
            Clear operation result
        """
        try:
            result = self._make_request('POST', '/api/jobs/clear')
            self._log_info("Successfully cleared job queues on server")
            return result
        except (ServerConnectionError, ServerAPIError) as e:
            self._log_error(f"Failed to clear job queues: {e}")
            raise
    
    def wait_for_jobs_completion(self, job_ids: List[str], 
                                 poll_interval: float = 5.0,
                                 timeout: Optional[float] = None,
                                 progress_callback: Optional[Callable[[Dict], None]] = None) -> Dict[str, Any]:
        """
        Wait for specific jobs to complete.
        
        Args:
            job_ids: List of job IDs to monitor
            poll_interval: How often to poll for status (seconds)
            timeout: Maximum time to wait (seconds), None for no timeout
            progress_callback: Optional callback for progress updates
            
        Returns:
            Dictionary with completed and failed jobs
        """
        start_time = time.time()
        completed_jobs = {}
        failed_jobs = {}
        remaining_jobs = set(job_ids)
        
        while remaining_jobs:
            # Check timeout
            if timeout and (time.time() - start_time) > timeout:
                self._log_warning(f"Timeout waiting for jobs. {len(remaining_jobs)} jobs still pending")
                break
            
            # Check each remaining job
            for job_id in list(remaining_jobs):
                try:
                    job_status = self.get_job_status(job_id)
                    status = job_status.get('status')
                    
                    if status == 'completed':
                        completed_jobs[job_id] = job_status
                        remaining_jobs.remove(job_id)
                        self._log_info(f"Job {job_id} completed")
                    elif status == 'failed':
                        failed_jobs[job_id] = job_status
                        remaining_jobs.remove(job_id)
                        self._log_error(f"Job {job_id} failed: {job_status.get('error', 'Unknown error')}")
                
                except (ServerConnectionError, ServerAPIError) as e:
                    # If we can't get job status, assume it's still pending
                    self._log_warning(f"Could not check status of job {job_id}: {e}")
            
            # Call progress callback if provided
            if progress_callback:
                progress_info = {
                    'total_jobs': len(job_ids),
                    'completed_jobs': len(completed_jobs),
                    'failed_jobs': len(failed_jobs),
                    'remaining_jobs': len(remaining_jobs),
                    'elapsed_time': time.time() - start_time
                }
                progress_callback(progress_info)
            
            # Sleep before next poll
            if remaining_jobs:  # Only sleep if there are still jobs to check
                time.sleep(poll_interval)
        
        return {
            'completed': completed_jobs,
            'failed': failed_jobs,
            'total_completed': len(completed_jobs),
            'total_failed': len(failed_jobs),
            'elapsed_time': time.time() - start_time
        }
    
    def is_server_available(self) -> bool:
        """
        Quick check if server is available.
        
        Returns:
            True if server responds to status request
        """
        try:
            self._make_request('GET', '/api/status', retry=False)
            return True
        except (ServerConnectionError, ServerAPIError):
            return False
    
    def get_server_capacity(self) -> Dict[str, int]:
        """
        Get current server processing capacity.
        
        Returns:
            Dictionary with capacity information
        """
        try:
            status = self.get_server_status()
            server_status = status.get('server_status', {})
            
            return {
                'total_capacity': server_status.get('total_capacity', 0),
                'used_capacity': server_status.get('used_capacity', 0),
                'available_capacity': server_status.get('available_capacity', 0),
                'active_clients': server_status.get('clients', 0),
                'queue_size': server_status.get('queue_size', 0)
            }
        except (ServerConnectionError, ServerAPIError):
            return {
                'total_capacity': 0,
                'used_capacity': 0,
                'available_capacity': 0,
                'active_clients': 0,
                'queue_size': 0
            }
    
    def close(self):
        """Close the HTTP session."""
        if self.session:
            self.session.close()


class ServerMonitor:
    """Monitor server status and jobs with periodic updates."""
    
    def __init__(self, client: ProcessingServerClient, update_interval: float = 5.0):
        self.client = client
        self.update_interval = update_interval
        self.running = False
        self.callbacks = {
            'status_update': [],
            'job_update': [],
            'error': []
        }
    
    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for monitoring events."""
        if event_type in self.callbacks:
            self.callbacks[event_type].append(callback)
    
    def remove_callback(self, event_type: str, callback: Callable):
        """Remove callback for monitoring events."""
        if event_type in self.callbacks and callback in self.callbacks[event_type]:
            self.callbacks[event_type].remove(callback)
    
    def _notify_callbacks(self, event_type: str, data: Any):
        """Notify all callbacks for an event type."""
        for callback in self.callbacks.get(event_type, []):
            try:
                callback(data)
            except Exception as e:
                print(f"Error in callback: {e}")
    
    def start_monitoring(self):
        """Start monitoring server status."""
        import threading
        
        if self.running:
            return
        
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop monitoring server status."""
        self.running = False
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Get server status
                status = self.client.get_server_status()
                self._notify_callbacks('status_update', status)
                
                # Get jobs overview
                jobs = self.client.get_jobs_overview()
                self._notify_callbacks('job_update', jobs)
                
            except (ServerConnectionError, ServerAPIError) as e:
                self._notify_callbacks('error', str(e))
            
            # Sleep for update interval
            time.sleep(self.update_interval)