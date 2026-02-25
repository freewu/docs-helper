import os
import subprocess
import shutil
import sys
from pathlib import Path
from version import __version__

def package_application():
    """Package the application using PyInstaller"""
    
    # Define paths
    main_script = "main.py"
    dist_dir = "dist"
    build_dir = "build"
    
    # Clean previous builds
    if os.path.exists(dist_dir):
        shutil.rmtree(dist_dir)
        print(f"Removed old {dist_dir} directory")
    
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
        print(f"Removed old {build_dir} directory")
    
    # Import version to create version-specific executable name
    from version import __version__
    
    # PyInstaller command with appropriate options for a GUI application
    cmd = [
        "pyinstaller",
        "--onefile",           # Create a single executable file
        "--windowed",          # Don't show console (for GUI apps)
        f"--name=DocsHelper-{__version__}",   # Name of the executable with version
        "--add-data=data;data" if os.path.exists("data") else "", # Include data directory if it exists
        main_script
    ]
    
    # Remove empty string from command if data directory doesn't exist
    cmd = [arg for arg in cmd if arg != ""]
    
    print("Starting packaging process...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Execute PyInstaller
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("Packaging completed successfully!")
        print(result.stdout)
        
        # Show the created executable with version
        exe_name = f"DocsHelper-{__version__}.exe" if sys.platform.startswith("win") else f"DocsHelper-{__version__}"
        exe_path = os.path.join(dist_dir, exe_name)
        
        if os.path.exists(exe_path):
            file_size = os.path.getsize(exe_path) / (1024 * 1024)  # Size in MB
            print(f"\nExecutable created: {exe_path}")
            print(f"File size: {file_size:.2f} MB")
        else:
            print(f"Warning: Expected executable not found at {exe_path}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error during packaging: {e}")
        print(f"Error output: {e.stderr}")
        return False
    except FileNotFoundError:
        print("PyInstaller not found. Please install it with: pip install pyinstaller")
        return False

def main():
    print("Docs Helper Application Packaging Tool")
    print("=" * 40)
    
    if not os.path.exists("main.py"):
        print("Error: main.py not found in current directory")
        return
    
    success = package_application()
    
    if success:
        print("\nPackaging process completed successfully!")
        print("Find your executable in the 'dist' folder.")
    else:
        print("\nPackaging process failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()