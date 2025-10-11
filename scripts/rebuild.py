#!/usr/bin/env python3
"""
Rebuild the virtual environment for the Clenow Momentum project.

This script removes the existing virtual environment and creates a fresh one
with all dependencies installed. Useful for:
- Resolving dependency conflicts
- Clean environment setup
- Pre-commit environment preparation
"""

import subprocess
import sys
from pathlib import Path
import shutil


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"🔄 {description}...")
    
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        
        # Filter out the pyvenv.cfg warning
        if result.stdout:
            output_lines = result.stdout.strip().split('\n')
            filtered_output = [
                line for line in output_lines 
                if "failed to locate pyvenv.cfg" not in line.lower()
            ]
            if filtered_output:
                print('\n'.join(filtered_output))
        
        # Also check stderr for the warning
        if result.stderr and "failed to locate pyvenv.cfg" not in result.stderr.lower():
            print(result.stderr, file=sys.stderr)
        
        return True
        
    except subprocess.CalledProcessError as e:
        # Filter the pyvenv.cfg warning from error output too
        error_msg = e.stderr if e.stderr else str(e)
        if "failed to locate pyvenv.cfg" not in error_msg.lower():
            print(f"❌ Failed: {error_msg}", file=sys.stderr)
        return False


def rebuild_environment():
    """Rebuild the Python virtual environment."""
    
    # Check if we're in the project root
    if not Path("pyproject.toml").exists():
        print("❌ Error: pyproject.toml not found. Please run from project root.")
        sys.exit(1)
    
    # Check if we're being run with uv run (which creates .venv automatically)
    if "uv run" in " ".join(sys.orig_argv) if hasattr(sys, 'orig_argv') else False:
        print("⚠️  Warning: Don't use 'uv run' to run this script!")
        print("   uv run creates a .venv before running, defeating the purpose.")
        print("\n   Instead, use:")
        print("   • python scripts/rebuild.py")
        print("   • py scripts/rebuild.py")
        sys.exit(1)
    
    # Remove existing .venv (even if corrupted/incomplete)
    venv_path = Path(".venv")
    pyvenv_cfg = venv_path / "pyvenv.cfg"
    
    # Check if .venv exists in any form
    if venv_path.exists() or venv_path.is_dir():
        if not pyvenv_cfg.exists():
            print("⚠️  Found incomplete/corrupted .venv directory (missing pyvenv.cfg)")
        print("🧹 Removing existing virtual environment...")
        
        # First, try to kill any Python processes using this venv
        if sys.platform == "win32":
            # Try to force remove using Windows commands
            remove_commands = [
                # First try normal removal
                f'powershell -Command "if (Test-Path {venv_path}) {{Remove-Item -Recurse -Force {venv_path} -ErrorAction SilentlyContinue}}"',
                # If that fails, use cmd's rmdir
                f'cmd /c "rmdir /s /q {venv_path} 2>nul"',
            ]
            
            removed = False
            for cmd in remove_commands:
                try:
                    subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
                    if not venv_path.exists():
                        removed = True
                        break
                except:
                    continue
            
            # If still exists, try Python's shutil as last resort
            if not removed and venv_path.exists():
                try:
                    shutil.rmtree(venv_path, ignore_errors=True)
                    removed = not venv_path.exists()
                except:
                    pass
            
            if not removed and venv_path.exists():
                print("⚠️  Could not remove .venv completely. This might be because:")
                print("   • VS Code or another editor is using the Python interpreter")
                print("   • A terminal has the virtual environment activated")
                print("\n   Please close all Python processes and terminals, then try again.")
                print("\n   Alternative: Run this command in PowerShell as admin:")
                print(f"   Remove-Item -Recurse -Force {venv_path}")
                sys.exit(1)
            else:
                print("✓ Old environment removed")
        else:
            # Unix-like systems
            try:
                shutil.rmtree(venv_path)
                print("✓ Old environment removed")
            except Exception as e:
                print(f"❌ Failed to remove .venv: {e}")
                sys.exit(1)
    
    # Double-check that .venv is really gone
    if venv_path.exists():
        print("⚠️  Warning: .venv directory still exists after removal attempt")
        print("   Attempting one final cleanup...")
        try:
            import time
            time.sleep(1)  # Give Windows time to release file handles
            if sys.platform == "win32":
                subprocess.run('cmd /c "rmdir /s /q .venv 2>nul"', shell=True)
        except:
            pass
    
    # Create new virtual environment
    if not run_command("uv venv --python 3.13", "Creating virtual environment with Python 3.13"):
        sys.exit(1)
    
    # Install dependencies
    if not run_command("uv sync --dev", "Installing dependencies"):
        sys.exit(1)
    
    print("\n✅ Environment successfully rebuilt!")
    print("\nYou can now:")
    print("  • Press F5 in VS Code to start debugging")
    print("  • Run: uv run python main.py")
    print("  • Activate manually: .venv\\Scripts\\activate (Windows)")
    
    # Check Python version in the new environment
    try:
        result = subprocess.run(
            ".venv\\Scripts\\python --version",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"\n📌 Environment Python: {result.stdout.strip()}")
    except:
        pass


if __name__ == "__main__":
    rebuild_environment()