@echo off
echo Starting Galaxy Image Enhancer API server...
echo Server will be available at http://localhost:8000
echo Press Ctrl+C to stop the server
echo.
python -m uvicorn main:app --reload --host localhost --port 8000
pause