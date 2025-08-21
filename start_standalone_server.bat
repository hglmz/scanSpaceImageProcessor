@echo off
REM Batch script to start the standalone processing server
REM This makes it easy to launch the server with default settings

echo Starting Scan Space Standalone Processing Server...
echo.

REM Start the server with default settings
python standalone_server.py --host 0.0.0.0 --port 8888 --api-port 8889 --log-level INFO

echo.
echo Server stopped. Press any key to exit...
pause > nul