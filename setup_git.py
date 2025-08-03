"""
Git Setup Script for Enhanced Dia TTS

This script helps you set up git and push your enhanced system to GitHub.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and print the result."""
    print(f"🔧 {description}")
    print(f"   Running: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=".")
        
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout.strip():
                print(f"   Output: {result.stdout.strip()}")
        else:
            print(f"   ❌ Error: {result.stderr.strip()}")
            return False
    except Exception as e:
        print(f"   ❌ Exception: {e}")
        return False
    
    return True

def check_git_installed():
    """Check if git is installed."""
    try:
        result = subprocess.run("git --version", shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ Git is installed: {result.stdout.strip()}")
            return True
        else:
            print("❌ Git is not installed. Please install git first.")
            return False
    except:
        print("❌ Git is not installed. Please install git first.")
        return False

def check_required_files():
    """Check if all required files are present."""
    required_files = [
        "enhanced_voice_library.py",
        "runpod_handler.py",
        "audio_processing.py",
        "config_enhanced.py",
        "runpod_client.py",
        "Dockerfile.runpod",
        "runpod_requirements.txt",
        "README_ENHANCED.md"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    else:
        print("✅ All required files present")
        return True

def setup_git_repo():
    """Set up git repository and push to GitHub."""
    
    # Repository URL
    repo_url = "https://github.com/703deuce/dia.git"
    
    print("🚀 Setting up Git repository for Enhanced Dia TTS")
    print("=" * 60)
    
    # Check if git is installed
    if not check_git_installed():
        return False
    
    # Check required files
    if not check_required_files():
        print("\n💡 Make sure you've run the setup scripts first:")
        print("   python setup_runpod.py")
        return False
    
    # Initialize git if not already done
    if not os.path.exists(".git"):
        if not run_command("git init", "Initializing git repository"):
            return False
    
    # Configure git user (if not already set)
    print("\n🔧 Checking git configuration...")
    result = subprocess.run("git config user.name", shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        name = input("Enter your git username: ")
        run_command(f'git config user.name "{name}"', "Setting git username")
    
    result = subprocess.run("git config user.email", shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        email = input("Enter your git email: ")
        run_command(f'git config user.email "{email}"', "Setting git email")
    
    # Add remote origin
    print("\n🌐 Setting up remote repository...")
    result = subprocess.run("git remote get-url origin", shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        if not run_command(f"git remote add origin {repo_url}", "Adding remote origin"):
            return False
    else:
        print(f"   ✅ Remote origin already set to: {result.stdout.strip()}")
    
    # Create main branch if needed
    result = subprocess.run("git branch --show-current", shell=True, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        if not run_command("git checkout -b main", "Creating main branch"):
            return False
    
    # Add files
    print("\n📁 Adding files to git...")
    if not run_command("git add .", "Adding all files"):
        return False
    
    # Create commit
    print("\n💾 Creating commit...")
    commit_message = "Enhanced Dia TTS System with RunPod Serverless Support\n\nFeatures:\n- Intelligent text chunking with speaker awareness\n- Advanced voice library with DAC token caching\n- RunPod serverless handler (no FastAPI overhead)\n- Memory optimization and VRAM reduction\n- Audio post-processing pipeline\n- OpenAI-compatible API\n- Comprehensive deployment tools"
    
    if not run_command(f'git commit -m "{commit_message}"', "Creating initial commit"):
        # Check if there are changes to commit
        result = subprocess.run("git status --porcelain", shell=True, capture_output=True, text=True)
        if not result.stdout.strip():
            print("   ℹ️  No changes to commit")
        else:
            return False
    
    # Push to GitHub
    print("\n🚀 Pushing to GitHub...")
    if not run_command("git push -u origin main", "Pushing to GitHub"):
        return False
    
    print("\n" + "=" * 60)
    print("🎉 Successfully pushed to GitHub!")
    print(f"📁 Repository: {repo_url}")
    print("\n🎯 Next Steps:")
    print("1. Your enhanced TTS system is now on GitHub")
    print("2. You can use this repository URL in RunPod")
    print("3. RunPod will clone and build your Docker image")
    print("4. Deploy using the repository in RunPod serverless")
    
    return True

def main():
    """Main function."""
    if not os.path.exists("enhanced_voice_library.py"):
        print("❌ Please run this script from the dia/ directory")
        sys.exit(1)
    
    success = setup_git_repo()
    if not success:
        print("\n❌ Git setup failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()