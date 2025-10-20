#!/usr/bin/env python3
"""
Run Script untuk 3D Mesh Editor
Computer Graphics - Universitas Indonesia

Usage: python run.py
"""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    """Main entry point"""
    print("=" * 50)
    print("3D MESH EDITOR")
    print("Computer Graphics - Universitas Indonesia")
    print("=" * 50)
    print()
    print("Starting application...\n")

    # Import and run the main application
    try:
        from mesh_editor import MeshEditorApp

        app = MeshEditorApp()
        app.run()

    except Exception as e:
        print(f"\nError starting application: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()