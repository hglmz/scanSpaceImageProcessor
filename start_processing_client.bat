@echo off
REM Batch script to start the imare processing worker client
REM Edit the IP and port to configure for your network

echo Starting Scan Space Standalone Processing Client...
echo.

REM Start the client with default settings
python run_headless_processing_client.py --host 192.168.1.102 --port 8888 --threads 8

echo.
echo Client stopped. Press any key to exit...
pause > nul