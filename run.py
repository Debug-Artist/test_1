"""
================================================================================
Application Launcher Script
File: run.py
Description: Simple entry point to launch the Uvicorn ASGI server locally.
================================================================================
"""

import uvicorn
import webbrowser
import threading
import time

def open_browser():
    time.sleep(1.2)
    webbrowser.open("http://127.0.0.1:8000")

if __name__ == "__main__":
    print("\n=======================================================")
    print(" AGRONOMIC MOMENTUM OPTIMIZATION PLATFORM")
    print(" Vector Calculus & Yield Prediction Research Console")
    print("=======================================================")
    print(" Launching local server at: http://127.0.0.1:8000")
    print(" Press Ctrl+C to terminate the application.\n")
    
    # Automatically open default web browser
    threading.Thread(target=open_browser, daemon=True).start()
    
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
