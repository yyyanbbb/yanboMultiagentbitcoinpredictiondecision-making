#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
N8N API Server Launcher

Simply run this file to start the N8N API server:
    python run_n8n_server.py

The server will start at http://localhost:5000
"""

import os
import sys
import subprocess
from pathlib import Path


def main():
    print("=" * 60)
    print("  N8N API Server for Bitcoin Investment Decision System")
    print("=" * 60)
    print()
    
    # Get the directory of this script
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    
    # Change to project root directory
    os.chdir(project_root)
    print(f"[*] Working directory: {project_root}")
    
    # Check and install required packages
    print("[*] Checking required packages...")
    try:
        import flask
        import flask_cors
        print("[OK] Flask and Flask-CORS are installed")
    except ImportError:
        print("[*] Installing required packages...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask", "flask-cors", "-q"])
        print("[OK] Packages installed")
    
    # Add project root to path
    sys.path.insert(0, str(project_root))
    sys.path.insert(0, str(script_dir))
    
    print()
    print("[*] Starting API server...")
    print("[*] Server URL: http://localhost:5000")
    print("[*] Health check: http://localhost:5000/api/health")
    print()
    print("Press Ctrl+C to stop the server")
    print("=" * 60)
    print()
    
    # Run the API server as a subprocess
    api_server_path = script_dir / "n8n_api_server.py"
    subprocess.run([sys.executable, str(api_server_path)])


if __name__ == "__main__":
    main()
