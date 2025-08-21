#!/usr/bin/env python3
"""
Standalone Processing Server for Scan Space Image Processor

This is a standalone server application that handles distributed image processing
jobs for the Scan Space Image Processor. It provides a REST API for job submission
and monitoring, while maintaining the existing TCP-based client communication.

Usage:
    python standalone_server.py [--host HOST] [--port PORT] [--api-port API_PORT] [--log-level LEVEL]

Examples:
    # Start server with default settings
    python standalone_server.py
    
    # Start server on specific IPs and ports
    python standalone_server.py --host 0.0.0.0 --port 8888 --api-port 8889
    
    # Start with debug logging
    python standalone_server.py --log-level DEBUG
"""

import argparse
import json
import logging
import sys
import threading
import time
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Dict, Any, Optional
from urllib.parse import urlparse, parse_qs
import signal

# Import our existing network processor
from ImageProcessor.networkProcessor import ProcessingServer, ProcessingJob, LogLevel


class ProcessingServerAPI(BaseHTTPRequestHandler):
    """HTTP API handler for the processing server."""
    
    def __init__(self, server_instance, *args, **kwargs):
        self.server_instance = server_instance
        super().__init__(*args, **kwargs)
    
    def do_GET(self):
        """Handle GET requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        query_params = parse_qs(parsed_path.query)
        
        try:
            # Web interface routes
            if path == '/' or path == '/monitor':
                self._serve_web_interface()
            elif path == '/resources/scanSpaceLogo_256px.ico':
                self._serve_favicon()
            # API routes
            elif path == '/api/status':
                self._handle_status()
            elif path == '/api/jobs':
                self._handle_jobs_list()
            elif path == '/api/jobs/current':
                self._handle_current_jobs()
            elif path == '/api/jobs/queued':
                self._handle_queued_jobs()
            elif path == '/api/jobs/groups':
                self._handle_job_groups()
            elif path == '/api/jobs/completed':
                self._handle_completed_jobs()
            elif path == '/api/groups/completed':
                self._handle_completed_groups()
            elif path.startswith('/api/jobs/group/'):
                group_id = path.split('/')[-1]
                self._handle_job_group_detail(group_id)
            elif path == '/api/clients':
                self._handle_clients_list()
            elif path.startswith('/api/job/'):
                job_id = path.split('/')[-1]
                self._handle_job_status(job_id)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def do_POST(self):
        """Handle POST requests."""
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        try:
            if path == '/api/jobs/submit':
                self._handle_job_submission()
            elif path == '/api/jobs/clear':
                self._handle_clear_jobs()
            elif path.startswith('/api/client/') and path.endswith('/restart'):
                client_id = path.split('/')[-2]
                self._handle_client_restart(client_id)
            elif path.startswith('/api/client/') and path.endswith('/threads'):
                client_id = path.split('/')[-2]
                self._handle_client_thread_count(client_id)
            elif path.startswith('/api/jobs/group/') and path.endswith('/stop'):
                group_id = path.split('/')[-2]
                self._handle_stop_job_group(group_id)
            else:
                self._send_error(404, "Endpoint not found")
        except Exception as e:
            self._send_error(500, f"Internal server error: {str(e)}")
    
    def _read_json_body(self) -> Dict[str, Any]:
        """Read and parse JSON from request body."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}
        
        body = self.rfile.read(content_length)
        return json.loads(body.decode('utf-8'))
    
    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response."""
        self.send_response(status_code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        
        response = json.dumps(data, indent=2, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))
    
    def _send_error(self, status_code: int, message: str):
        """Send error response."""
        self._send_json_response({
            'error': message,
            'timestamp': datetime.now().isoformat()
        }, status_code)
    
    def _handle_status(self):
        """Handle server status request."""
        status = self.server_instance.get_status()
        
        # Add API-specific status information
        api_status = {
            'server_status': status,
            'api_version': '1.0',
            'uptime_seconds': time.time() - self.server_instance.start_time if hasattr(self.server_instance, 'start_time') else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        self._send_json_response(api_status)
    
    def _handle_jobs_list(self):
        """Handle jobs list request."""
        jobs_info = {
            'queue_size': self.server_instance.job_queue.qsize(),
            'active_jobs': len(self.server_instance.active_jobs),
            'completed_jobs': len(self.server_instance.completed_jobs),
            'failed_jobs': len(self.server_instance.failed_jobs),
            'active_job_ids': list(self.server_instance.active_jobs.keys()),
            'completed_job_ids': list(self.server_instance.completed_jobs.keys()),
            'failed_job_ids': list(self.server_instance.failed_jobs.keys()),
            'timestamp': datetime.now().isoformat()
        }
        
        self._send_json_response(jobs_info)
    
    def _handle_clients_list(self):
        """Handle clients list request."""
        clients_info = []
        
        for client_id, client_info in self.server_instance.clients.items():
            client_data = {
                'client_id': client_id,
                'address': client_info.address,
                'port': client_info.port,
                'status': client_info.status,
                'thread_count': client_info.thread_count,
                'current_jobs': client_info.current_jobs.copy(),
                'total_memory_gb': client_info.total_memory_gb,
                'available_memory_gb': client_info.available_memory_gb,
                'last_heartbeat': client_info.last_heartbeat,
                'is_alive': client_info.is_alive()
            }
            clients_info.append(client_data)
        
        response = {
            'clients': clients_info,
            'total_clients': len(clients_info),
            'active_clients': len([c for c in clients_info if c['status'] != 'disconnected']),
            'timestamp': datetime.now().isoformat()
        }
        
        self._send_json_response(response)
    
    def _handle_job_status(self, job_id: str):
        """Handle individual job status request."""
        job_status = None
        
        if job_id in self.server_instance.active_jobs:
            job = self.server_instance.active_jobs[job_id]
            job_status = {
                'job_id': job_id,
                'status': 'active',
                'input_file': job.input_file,
                'output_file': job.output_file,
                'assigned_client': job.assigned_client,
                'progress': getattr(job, 'progress', 0)
            }
        elif job_id in self.server_instance.completed_jobs:
            result = self.server_instance.completed_jobs[job_id]
            job_status = {
                'job_id': job_id,
                'status': 'completed',
                'result': result,
                'completed_at': result.get('completed_at', 'unknown')
            }
        elif job_id in self.server_instance.failed_jobs:
            result = self.server_instance.failed_jobs[job_id]
            job_status = {
                'job_id': job_id,
                'status': 'failed',
                'error': result.get('error', 'Unknown error'),
                'failed_at': result.get('failed_at', 'unknown')
            }
        else:
            self._send_error(404, f"Job {job_id} not found")
            return
        
        job_status['timestamp'] = datetime.now().isoformat()
        self._send_json_response(job_status)
    
    def _handle_job_submission(self):
        """Handle job submission from project data."""
        try:
            project_data = self._read_json_body()
            
            if not project_data:
                self._send_error(400, "No project data provided")
                return
            
            # Validate project data structure
            required_fields = ['images', 'processing_settings']
            missing_fields = [field for field in required_fields if field not in project_data]
            if missing_fields:
                self._send_error(400, f"Missing required fields: {missing_fields}")
                return
            
            # Extract job parameters from project data
            images = project_data['images']
            processing_settings = project_data['processing_settings']
            image_groups = project_data.get('image_groups', {})
            
            # Debug: Log image groups and their chart status
            print(f"Debug: Received {len(image_groups)} image groups:")
            for group_name, group_data in image_groups.items():
                swatches_count = len(group_data.get('chart_swatches', []))
                print(f"  Group '{group_name}': {swatches_count} swatches")
            
            # Filter to only selected images or all if none selected
            selected_images = [img for img in images if img.get('selected', False)]
            if not selected_images:
                selected_images = images
            
            if not selected_images:
                self._send_error(400, "No images to process")
                return
            
            # Update project data with only selected images for processing
            filtered_project_data = project_data.copy()
            filtered_project_data['images'] = selected_images
            
            # Use the new ProcessingServer method to handle project data
            result = self.server_instance.add_project_jobs(filtered_project_data)
            
            if not result['success']:
                self._send_error(500, f"Error processing project: {result.get('error', 'Unknown error')}")
                return
            
            # Extract job information for response
            jobs_created = [{
                'job_id': job['job_id'],
                'input_file': job['image_path'], 
                'output_file': job['output_path']
            } for job in result['created_jobs']]
            
            # Build response with detailed information
            total_created = result['jobs_created']
            total_failed = result['jobs_failed']
            
            response = {
                'message': f'Successfully submitted {total_created} jobs ({total_failed} failed)',
                'jobs_created': total_created,
                'jobs_failed': total_failed,
                'job_ids': [job['job_id'] for job in jobs_created],
                'jobs': jobs_created,
                'failed_jobs': result.get('failed_jobs', []),
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response, 201)
            
        except json.JSONDecodeError:
            self._send_error(400, "Invalid JSON in request body")
        except Exception as e:
            self._send_error(500, f"Error processing job submission: {str(e)}")
    
    def _generate_output_path(self, image_data: Dict, processing_settings: Dict) -> str:
        """Generate output file path based on processing settings."""
        input_path = Path(image_data['full_path'])
        output_dir = processing_settings.get('output_directory', str(input_path.parent))
        export_format = processing_settings.get('export_format', '.jpg')
        
        # Ensure export format starts with dot
        if not export_format.startswith('.'):
            export_format = '.' + export_format
        
        # Generate output filename
        output_filename = input_path.stem + export_format
        output_path = Path(output_dir) / output_filename
        
        return str(output_path)
    
    def _handle_clear_jobs(self):
        """Handle request to clear job queues."""
        try:
            self.server_instance.clear_job_queue()
            
            response = {
                'message': 'Job queues cleared successfully',
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error clearing jobs: {str(e)}")
    
    def _serve_web_interface(self):
        """Serve the web monitoring interface."""
        try:
            # Get the path to the HTML file
            html_path = Path(__file__).parent / 'resources' / 'server_web_interface.html'
            
            if not html_path.exists():
                self._send_error(404, "Web interface file not found")
                return
            
            # Read the HTML file
            with open(html_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Send HTML response
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Expires', '0')
            self.end_headers()
            
            self.wfile.write(content.encode('utf-8'))
            
        except Exception as e:
            self._send_error(500, f"Error serving web interface: {str(e)}")
    
    def _serve_favicon(self):
        """Serve the favicon."""
        try:
            # Get the path to the favicon file
            favicon_path = Path(__file__).parent / 'resources' / 'scanSpaceLogo_256px.ico'
            
            if not favicon_path.exists():
                self._send_error(404, "Favicon not found")
                return
            
            # Read the favicon file
            with open(favicon_path, 'rb') as f:
                content = f.read()
            
            # Send favicon response
            self.send_response(200)
            self.send_header('Content-Type', 'image/x-icon')
            self.send_header('Cache-Control', 'public, max-age=86400')  # Cache for 1 day
            self.end_headers()
            
            self.wfile.write(content)
            
        except Exception as e:
            self._send_error(500, f"Error serving favicon: {str(e)}")
    
    def _handle_current_jobs(self):
        """Handle current/active jobs request for web interface."""
        try:
            jobs_list = []
            
            for job_id, job in self.server_instance.active_jobs.items():
                job_data = {
                    'job_id': job_id,
                    'job_name': getattr(job, 'name', f'Job {job_id[:8]}'),
                    'image_count': getattr(job, 'total_images', 1),
                    'image_groups': getattr(job, 'image_groups', 1),
                    'progress': getattr(job, 'progress', 0),
                    'output_format': getattr(job, 'output_format', 'JPEG'),
                    'output_bitdepth': getattr(job, 'output_bitdepth', '8-bit'),
                    'assigned_client': getattr(job, 'assigned_client', 'Unassigned'),
                    'created_at': getattr(job, 'created_time', time.time())
                }
                jobs_list.append(job_data)
            
            response = {
                'jobs': jobs_list,
                'total_jobs': len(jobs_list),
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error getting current jobs: {str(e)}")
    
    def _handle_queued_jobs(self):
        """Handle queued jobs request for web interface."""
        try:
            jobs_list = []
            
            # Note: Queue inspection is limited in Python's Queue class
            # For a production system, you'd want to maintain a separate list of queued jobs
            # For now, we'll show estimated queue information
            queue_size = self.server_instance.job_queue.qsize()
            
            # Create placeholder entries for queued jobs
            for i in range(min(queue_size, 10)):  # Show up to 10 queued jobs
                job_data = {
                    'job_id': f'queued_{i+1}',
                    'job_name': f'Queued Job {i+1}',
                    'image_count': 'Unknown',
                    'image_groups': 1,
                    'output_format': 'JPEG',
                    'output_bitdepth': '8-bit',
                    'priority': 'Normal',
                    'created_at': datetime.now().isoformat()
                }
                jobs_list.append(job_data)
            
            response = {
                'jobs': jobs_list,
                'total_jobs': queue_size,
                'showing': len(jobs_list),
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error getting queued jobs: {str(e)}")
    
    def _handle_job_groups(self):
        """Handle job groups request for web interface."""
        try:
            job_groups_list = []
            
            for group_id, group_data in self.server_instance.job_groups.items():
                group_info = {
                    'group_id': group_id,
                    'name': group_data['name'],
                    'total_images': group_data['total_images'],
                    'completed_count': group_data['completed_count'],
                    'failed_count': group_data['failed_count'],
                    'active_count': group_data['active_count'],
                    'queued_count': group_data['queued_count'],
                    'status': group_data['status'],
                    'created_time': group_data['created_time'],
                    'progress_percentage': round((group_data['completed_count'] / group_data['total_images']) * 100, 1) if group_data['total_images'] > 0 else 0
                }
                job_groups_list.append(group_info)
            
            # Sort by creation time (newest first)
            job_groups_list.sort(key=lambda x: x['created_time'], reverse=True)
            
            response = {
                'job_groups': job_groups_list,
                'total_groups': len(job_groups_list),
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error getting job groups: {str(e)}")
    
    def _handle_completed_jobs(self):
        """Handle completed jobs request for web interface."""
        try:
            # Get recent completed jobs data from the server
            recent_completed = getattr(self.server_instance, 'recent_completed_jobs', [])
            
            # Get the last 5 completed jobs with processing time information
            jobs_list = []
            for job_data in recent_completed[-5:]:  # Get last 5 jobs
                job_info = {
                    'job_id': job_data.get('job_id', 'Unknown'),
                    'job_name': job_data.get('job_name', f"Job {job_data.get('job_id', '')[:8]}"),
                    'image_name': job_data.get('image_name', 'Unknown'),
                    'output_format': job_data.get('output_format', 'JPEG'),
                    'assigned_client': job_data.get('assigned_client', 'Unknown'),
                    'completed_at': job_data.get('completed_at', time.time()),
                    'start_time': job_data.get('start_time', 0),
                    'end_time': job_data.get('end_time', 0),
                    'processing_time': job_data.get('processing_time', 0),
                    'processing_time_formatted': self._format_processing_time(job_data.get('processing_time', 0))
                }
                jobs_list.append(job_info)
            
            # Reverse to show most recent first
            jobs_list.reverse()
            
            response = {
                'jobs': jobs_list,
                'total_jobs': len(jobs_list),
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error getting completed jobs: {str(e)}")
    
    def _handle_completed_groups(self):
        """Handle completed job groups request for web interface."""
        try:
            # Get recent completed groups data from the server
            recent_completed = getattr(self.server_instance, 'recent_completed_groups', [])
            
            # Get the last 5 completed groups with processing time information
            groups_list = []
            for group_data in recent_completed[-5:]:  # Get last 5 groups
                group_info = {
                    'group_id': group_data.get('group_id', 'Unknown'),
                    'group_name': group_data.get('group_name', f"Group {group_data.get('group_id', '')[:8]}"),
                    'total_images': group_data.get('total_images', 0),
                    'completed_images': group_data.get('completed_images', 0),
                    'failed_images': group_data.get('failed_images', 0),
                    'status': group_data.get('status', 'completed'),
                    'created_time': group_data.get('created_time', time.time()),
                    'completed_time': group_data.get('completed_time', time.time()),
                    'processing_time': group_data.get('processing_time', 0),
                    'avg_time_per_image': group_data.get('avg_time_per_image', 0),
                    'processing_time_formatted': self._format_processing_time(group_data.get('processing_time', 0)),
                    'avg_time_formatted': self._format_processing_time(group_data.get('avg_time_per_image', 0))
                }
                groups_list.append(group_info)
            
            # Reverse to show most recent first
            groups_list.reverse()
            
            response = {
                'groups': groups_list,
                'total_groups': len(groups_list),
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error getting completed groups: {str(e)}")

    def _format_processing_time(self, seconds):
        """Format processing time in human-readable format."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def _handle_job_group_detail(self, group_id: str):
        """Handle detailed job group information request."""
        try:
            if group_id not in self.server_instance.job_groups:
                self._send_error(404, f"Job group {group_id} not found")
                return
            
            group_data = self.server_instance.job_groups[group_id]
            
            # Get detailed information about each image job
            image_jobs = []
            for job_id in group_data['image_jobs']:
                job_info = {
                    'job_id': job_id,
                    'status': 'queued',  # default
                    'assigned_client': None,
                    'progress': 0
                }
                
                if job_id in self.server_instance.active_jobs:
                    job = self.server_instance.active_jobs[job_id]
                    job_info.update({
                        'status': 'active',
                        'filename': job.image_path.split('/')[-1] if hasattr(job, 'image_path') else 'Unknown',
                        'assigned_client': getattr(job, 'assigned_client', None),
                        'progress': getattr(job, 'progress', 0)
                    })
                elif job_id in self.server_instance.completed_jobs:
                    result = self.server_instance.completed_jobs[job_id]
                    job_info.update({
                        'status': 'completed',
                        'filename': result.get('image_path', 'Unknown').split('/')[-1],
                        'completed_at': result.get('completed_at', 'unknown'),
                        'progress': 100
                    })
                elif job_id in self.server_instance.failed_jobs:
                    result = self.server_instance.failed_jobs[job_id]
                    job_info.update({
                        'status': 'failed',
                        'filename': result.get('image_path', 'Unknown').split('/')[-1],
                        'error': result.get('error', 'Unknown error'),
                        'failed_at': result.get('failed_at', 'unknown')
                    })
                
                image_jobs.append(job_info)
            
            response = {
                'group_id': group_id,
                'name': group_data['name'],
                'total_images': group_data['total_images'],
                'completed_count': group_data['completed_count'],
                'failed_count': group_data['failed_count'],
                'active_count': group_data['active_count'],
                'queued_count': group_data['queued_count'],
                'status': group_data['status'],
                'created_time': group_data['created_time'],
                'image_jobs': image_jobs,
                'timestamp': datetime.now().isoformat()
            }
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error getting job group detail: {str(e)}")
    
    def _handle_client_restart(self, client_id: str):
        """Handle client restart request."""
        try:
            if client_id not in self.server_instance.clients:
                self._send_error(404, f"Client {client_id} not found")
                return
            
            client_info = self.server_instance.clients[client_id]
            
            # Send actual restart command to client using the new method
            success = self.server_instance.restart_client(client_id)
            
            if success:
                response = {
                    'message': f'Restart command sent to client {client_id}',
                    'client_id': client_id,
                    'client_address': client_info.address,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
                logging.info(f"Successfully sent restart command to client {client_id} ({client_info.address})")
            else:
                response = {
                    'message': f'Failed to restart client {client_id}',
                    'client_id': client_id,
                    'client_address': client_info.address,
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }
                logging.error(f"Failed to send restart command to client {client_id}")
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error restarting client: {str(e)}")
    
    def _handle_client_thread_count(self, client_id: str):
        """Handle client thread count update request."""
        try:
            # Read the new thread count from request body
            body = self._read_json_body()
            new_thread_count = body.get('thread_count')
            
            if new_thread_count is None:
                self._send_error(400, "Missing thread_count in request body")
                return
            
            # Validate thread count
            try:
                new_thread_count = int(new_thread_count)
                if new_thread_count < 1 or new_thread_count > 32:
                    self._send_error(400, f"Invalid thread count: {new_thread_count}. Must be between 1 and 32.")
                    return
            except (ValueError, TypeError):
                self._send_error(400, f"Invalid thread count value: {new_thread_count}")
                return
            
            if client_id not in self.server_instance.clients:
                self._send_error(404, f"Client {client_id} not found")
                return
            
            client_info = self.server_instance.clients[client_id]
            
            # Send thread count change command to client
            success = self.server_instance.send_thread_count_update(client_id, new_thread_count)
            
            if success:
                # Update the stored thread count
                client_info.thread_count = new_thread_count
                
                response = {
                    'message': f'Thread count updated for client {client_id}',
                    'client_id': client_id,
                    'new_thread_count': new_thread_count,
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
                logging.info(f"Successfully updated thread count for client {client_id} to {new_thread_count}")
            else:
                response = {
                    'message': f'Failed to update thread count for client {client_id}',
                    'client_id': client_id,
                    'requested_thread_count': new_thread_count,
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }
                logging.error(f"Failed to update thread count for client {client_id}")
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error updating client thread count: {str(e)}")
    
    def _handle_stop_job_group(self, group_id: str):
        """Handle stop job group request."""
        try:
            if group_id not in self.server_instance.job_groups:
                self._send_error(404, f"Job group {group_id} not found")
                return
            
            # Call the server's stop_job_group method
            result = self.server_instance.stop_job_group(group_id)
            
            if result['success']:
                response = {
                    'message': result.get('message', f'Job group {group_id} stopped successfully'),
                    'group_id': group_id,
                    'group_name': result.get('group_name', 'Unknown'),
                    'jobs_removed_from_queue': result.get('jobs_removed_from_queue', 0),
                    'active_jobs_cancelled': result.get('active_jobs_cancelled', 0),
                    'total_jobs_stopped': result.get('total_jobs_stopped', 0),
                    'success': True,
                    'timestamp': datetime.now().isoformat()
                }
                removed = result.get('jobs_removed_from_queue', 0)
                cancelled = result.get('active_jobs_cancelled', 0)
                logging.info(f"Successfully stopped job group {group_id}: {removed} removed, {cancelled} cancelled")
            else:
                response = {
                    'message': f'Failed to stop job group {group_id}',
                    'group_id': group_id,
                    'error': result.get('error', 'Unknown error'),
                    'success': False,
                    'timestamp': datetime.now().isoformat()
                }
                logging.error(f"Failed to stop job group {group_id}: {result.get('error', 'Unknown error')}")
            
            self._send_json_response(response)
            
        except Exception as e:
            self._send_error(500, f"Error stopping job group: {str(e)}")
    
    def log_message(self, format, *args):
        """Override to use our logging system."""
        pass  # Suppress default HTTP server logging


class APIServerWrapper:
    """Wrapper for the HTTP API server."""
    
    def __init__(self, processing_server: ProcessingServer, host: str = "0.0.0.0", port: int = 8889):
        self.processing_server = processing_server
        self.host = host
        self.port = port
        self.httpd = None
        self.server_thread = None
        self.running = False
    
    def start(self):
        """Start the API server."""
        try:
            # Create handler class with processing server instance
            handler_class = lambda *args, **kwargs: ProcessingServerAPI(self.processing_server, *args, **kwargs)
            
            self.httpd = HTTPServer((self.host, self.port), handler_class)
            self.running = True
            
            self.server_thread = threading.Thread(target=self._run_server, daemon=True)
            self.server_thread.start()
            
            return True
        except Exception as e:
            logging.error(f"Failed to start API server: {e}")
            return False
    
    def stop(self):
        """Stop the API server."""
        self.running = False
        if self.httpd:
            self.httpd.shutdown()
            self.httpd.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=5)
    
    def _run_server(self):
        """Run the HTTP server."""
        try:
            logging.info(f"API server started on http://{self.host}:{self.port}")
            self.httpd.serve_forever()
        except Exception as e:
            if self.running:  # Only log if we didn't intentionally stop
                logging.error(f"API server error: {e}")


class StandaloneServer:
    """Main standalone server application."""
    
    def __init__(self, host: str = "0.0.0.0", tcp_port: int = 8888, api_port: int = 8889, log_level: str = "INFO"):
        self.host = host
        self.tcp_port = tcp_port
        self.api_port = api_port
        self.log_level = log_level
        
        # Setup logging
        self._setup_logging()
        
        # Initialize servers
        self.processing_server = None
        self.api_server = None
        self.running = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _setup_logging(self):
        """Setup logging configuration."""
        log_level_map = {
            'DEBUG': LogLevel.DEBUG,
            'INFO': LogLevel.INFO,
            'WARNING': LogLevel.WARNING,
            'ERROR': LogLevel.ERROR
        }
        
        # Setup Python logging
        logging.basicConfig(
            level=getattr(logging, self.log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('standalone_server.log')
            ]
        )
        
        # Get log level for our network processor
        self.network_log_level = log_level_map.get(self.log_level, LogLevel.INFO)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logging.info(f"Received signal {signum}, shutting down...")
        self.stop()
        sys.exit(0)
    
    def start(self):
        """Start the standalone server."""
        logging.info("Starting Scan Space Processing Server...")
        logging.info(f"TCP Server: {self.host}:{self.tcp_port}")
        logging.info(f"API Server: {self.host}:{self.api_port}")
        
        try:
            # Start processing server
            self.processing_server = ProcessingServer(
                host=self.host,
                port=self.tcp_port,
                log_callback=self._log_callback,
                log_level=self.network_log_level
            )
            
            # Record start time for uptime calculations
            self.processing_server.start_time = time.time()
            
            if not self.processing_server.start():
                logging.error("Failed to start processing server")
                return False
            
            # Start API server
            self.api_server = APIServerWrapper(
                self.processing_server,
                host=self.host,
                port=self.api_port
            )
            
            if not self.api_server.start():
                logging.error("Failed to start API server")
                self.processing_server.stop()
                return False
            
            self.running = True
            logging.info("Standalone server started successfully")
            
            # Print API endpoints
            self._print_api_info()
            
            return True
            
        except Exception as e:
            logging.error(f"Error starting server: {e}")
            return False
    
    def stop(self):
        """Stop the standalone server."""
        if not self.running:
            return
        
        logging.info("Stopping standalone server...")
        self.running = False
        
        if self.api_server:
            self.api_server.stop()
        
        if self.processing_server:
            self.processing_server.stop()
        
        logging.info("Standalone server stopped")
    
    def run(self):
        """Run the server until stopped."""
        if not self.start():
            return False
        
        try:
            # Keep the main thread alive
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            logging.info("Keyboard interrupt received")
        finally:
            self.stop()
        
        return True
    
    def _log_callback(self, message: str):
        """Callback for processing server logs."""
        # Route processing server logs through Python logging
        if "[ERROR]" in message:
            logging.error(message)
        elif "[WARNING]" in message:
            logging.warning(message)
        elif "[DEBUG]" in message:
            logging.debug(message)
        else:
            logging.info(message)
    
    def _print_api_info(self):
        """Print API endpoint information."""
        base_url = f"http://{self.host}:{self.api_port}"
        
        print("\n" + "="*60)
        print("SCAN SPACE PROCESSING SERVER")
        print("="*60)
        print(f"TCP Server (clients): {self.host}:{self.tcp_port}")
        print(f"HTTP API Server: {base_url}")
        print(f"Web Monitor Interface: {base_url}/")
        print("\nAPI Endpoints:")
        print(f"  GET  {base_url}/                - Web monitoring interface")
        print(f"  GET  {base_url}/api/status      - Server status")
        print(f"  GET  {base_url}/api/clients     - Connected clients")
        print(f"  GET  {base_url}/api/jobs        - Jobs overview")
        print(f"  GET  {base_url}/api/jobs/current - Current running jobs")
        print(f"  GET  {base_url}/api/jobs/queued - Queued jobs")
        print(f"  GET  {base_url}/api/job/[ID]    - Individual job status")
        print(f"  POST {base_url}/api/jobs/submit - Submit processing jobs")
        print(f"  POST {base_url}/api/jobs/clear  - Clear job queues")
        print(f"  POST {base_url}/api/client/[ID]/restart - Restart client")
        print("\nLog file: standalone_server.log")
        print("Press Ctrl+C to stop the server")
        print("="*60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Standalone Processing Server for Scan Space Image Processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python standalone_server.py
  python standalone_server.py --host 0.0.0.0 --port 8888 --api-port 8889
  python standalone_server.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        '--host',
        default='0.0.0.0',
        help='Host IP address to bind to (default: 0.0.0.0)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=8888,
        help='TCP port for client connections (default: 8888)'
    )
    
    parser.add_argument(
        '--api-port',
        type=int,
        default=8889,
        help='HTTP port for API server (default: 8889)'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level (default: INFO)'
    )
    
    args = parser.parse_args()
    
    # Create and run server
    server = StandaloneServer(
        host=args.host,
        tcp_port=args.port,
        api_port=args.api_port,
        log_level=args.log_level
    )
    
    success = server.run()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()