"""
RunPod Setup and Deployment Helper

This script helps you set up and deploy the Enhanced Dia TTS system to RunPod serverless.
"""

import os
import json
import subprocess
import sys
from pathlib import Path
import yaml

def check_dependencies():
    """Check if all required dependencies are available."""
    print("🔍 Checking dependencies...")
    
    required_files = [
        "enhanced_voice_library.py",
        "audio_processing.py", 
        "config_enhanced.py",
        "runpod_handler.py",
        "runpod_requirements.txt",
        "Dockerfile.runpod"
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
    
    print("✅ All required files present")
    return True

def create_config_file():
    """Create optimized configuration for RunPod."""
    print("⚙️ Creating RunPod configuration...")
    
    config = {
        "model": {
            "repo_id": "ttj/dia-1.6b-safetensors",
            "weights_filename": "dia-v0_1_bf16.safetensors",
            "config_filename": "config.json",
            "cache_dir": "/app/model_cache"
        },
        "voice_library": {
            "library_path": "/app/voice_library",
            "max_cache_size": 5,  # Smaller for serverless
            "preload_voices": []
        },
        "generation": {
            "defaults": {
                "cfg_scale": 3.0,
                "temperature": 1.3,
                "top_p": 0.95,
                "cfg_filter_top_k": 35,
                "seed": 42,
                "chunk_size": 120,
                "enable_chunking": True
            },
            "limits": {
                "max_text_length": 10000,
                "max_generation_time": 300,
                "concurrent_requests": 1  # Serverless handles concurrency
            }
        },
        "performance": {
            "memory_optimization": True,
            "cuda_memory_fraction": 0.95,  # Use most available memory
            "enable_memory_mapping": True,
            "garbage_collection_threshold": 3
        },
        "audio_processing": {
            "sample_rate": 44100,
            "enable_enhancement": True,
            "enhancement_preset": "balanced"
        }
    }
    
    with open("runpod_config.yaml", "w") as f:
        yaml.safe_dump(config, f, indent=2)
    
    print("✅ Created runpod_config.yaml")

def create_voice_setup_script():
    """Create script to add voices during Docker build."""
    print("🎵 Creating voice setup script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Voice setup script for RunPod deployment.
Add your voices here to pre-process them during Docker build.
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.append('/app')

from enhanced_voice_library import EnhancedVoiceLibrary
from dia.model import Dia

def setup_voices():
    """Setup voices during Docker build."""
    print("🎵 Setting up voices for RunPod deployment...")
    
    # Initialize system
    print("Loading Dia model...")
    dia_model = Dia.from_huggingface(
        repo_id=os.getenv("DIA_MODEL_REPO_ID", "ttj/dia-1.6b-safetensors"),
        weights_filename=os.getenv("DIA_MODEL_WEIGHTS_FILENAME", "dia-v0_1_bf16.safetensors"),
        cache_dir="/app/model_cache"
    )
    
    print("Initializing voice library...")
    library = EnhancedVoiceLibrary("/app/voice_library")
    
    # Add voices from voices directory
    voices_dir = Path("/app/voices")
    if voices_dir.exists():
        for voice_file in voices_dir.glob("*.wav"):
            voice_id = voice_file.stem
            txt_file = voices_dir / f"{voice_id}.txt"
            
            if txt_file.exists():
                with open(txt_file, 'r') as f:
                    transcript = f.read().strip()
                
                print(f"Adding voice: {voice_id}")
                try:
                    profile = library.add_voice_from_audio(
                        dia_model, 
                        voice_id, 
                        str(voice_file), 
                        transcript,
                        metadata={"source": "docker_build"}
                    )
                    print(f"✅ {voice_id}: {profile.processing_time:.2f}s")
                except Exception as e:
                    print(f"❌ {voice_id}: {e}")
            else:
                print(f"⚠️  Skipping {voice_id} - no transcript file")
    
    print(f"🎉 Voice setup complete! Added {len(library.index)} voices")

if __name__ == "__main__":
    setup_voices()
'''
    
    with open("setup_voices.py", "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod("setup_voices.py", 0o755)
    
    print("✅ Created setup_voices.py")

def create_docker_build_script():
    """Create Docker build script."""
    print("🐳 Creating Docker build script...")
    
    script_content = '''#!/bin/bash
# Build script for RunPod Enhanced Dia TTS

set -e

echo "🐳 Building Enhanced Dia TTS for RunPod..."

# Configuration
IMAGE_NAME="enhanced-dia-tts"
TAG="latest"
REGISTRY="${DOCKER_REGISTRY:-}"  # Set this to your registry

# Build the image
echo "📦 Building Docker image..."
docker build -f Dockerfile.runpod -t ${IMAGE_NAME}:${TAG} .

# Tag for registry if specified
if [ ! -z "$REGISTRY" ]; then
    echo "🏷️  Tagging for registry: $REGISTRY"
    docker tag ${IMAGE_NAME}:${TAG} ${REGISTRY}/${IMAGE_NAME}:${TAG}
    
    echo "📤 Pushing to registry..."
    docker push ${REGISTRY}/${IMAGE_NAME}:${TAG}
    
    echo "✅ Image pushed to: ${REGISTRY}/${IMAGE_NAME}:${TAG}"
else
    echo "✅ Image built: ${IMAGE_NAME}:${TAG}"
    echo "💡 Set DOCKER_REGISTRY environment variable to push to registry"
fi

echo "🎉 Build complete!"
echo ""
echo "Next steps:"
echo "1. Push image to a registry accessible by RunPod"
echo "2. Create RunPod serverless endpoint"
echo "3. Use image: ${REGISTRY:-}${IMAGE_NAME}:${TAG}"
'''
    
    with open("build_docker.sh", "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod("build_docker.sh", 0o755)
    
    print("✅ Created build_docker.sh")

def create_test_script():
    """Create local test script."""
    print("🧪 Creating test script...")
    
    script_content = '''#!/usr/bin/env python3
"""
Local test script for RunPod handler.
"""

import sys
import os

# Add current directory to path
sys.path.append('.')

def test_handler():
    """Test the RunPod handler locally."""
    print("🧪 Testing RunPod handler locally...")
    
    # Import and test
    try:
        from runpod_handler import handler, initialize_system
        
        # Initialize system
        print("🚀 Initializing system...")
        initialize_system()
        
        # Test health check
        print("🏥 Testing health check...")
        health_event = {
            "input": {
                "endpoint": "system",
                "data": {"type": "health"}
            }
        }
        
        result = handler(health_event)
        if result.get("success"):
            print("✅ Health check passed")
            print(f"   Model loaded: {result['status']['model_loaded']}")
            print(f"   Voices: {result['status']['voice_count']}")
        else:
            print(f"❌ Health check failed: {result}")
            return False
        
        # Test simple generation
        print("🎯 Testing simple generation...")
        tts_event = {
            "input": {
                "endpoint": "openai",
                "data": {
                    "input": "This is a test of the local handler.",
                    "voice": "S1",
                    "response_format": "wav"
                }
            }
        }
        
        result = handler(tts_event)
        if result.get("success"):
            print("✅ Generation test passed")
            print(f"   Generation time: {result['generation_time']:.2f}s")
            print(f"   Audio length: {result['audio_length']} samples")
        else:
            print(f"❌ Generation test failed: {result}")
            return False
        
        print("🎉 All tests passed! Handler is ready for RunPod.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_handler()
    sys.exit(0 if success else 1)
'''
    
    with open("test_local.py", "w") as f:
        f.write(script_content)
    
    # Make it executable
    os.chmod("test_local.py", 0o755)
    
    print("✅ Created test_local.py")

def create_voices_directory():
    """Create example voices directory structure."""
    print("📁 Creating voices directory structure...")
    
    voices_dir = Path("voices")
    voices_dir.mkdir(exist_ok=True)
    
    # Create example files
    example_files = {
        "narrator.txt": "[S1] This is a professional narrator voice with clear articulation and steady pacing suitable for audiobooks and documentaries.",
        "character_male.txt": "[S1] This is a male character voice with expressive delivery and emotional range for storytelling and dialogue.",
        "character_female.txt": "[S1] This is a female character voice with warm tone and natural inflection perfect for conversational content.",
        "announcer.txt": "[S1] This is an announcer voice with authoritative presence and clear pronunciation ideal for presentations and broadcasts."
    }
    
    for filename, content in example_files.items():
        file_path = voices_dir / filename
        if not file_path.exists():
            with open(file_path, "w") as f:
                f.write(content)
    
    # Create README
    readme_content = """# Voices Directory

Place your voice audio files (.wav) and corresponding transcript files (.txt) here.

## Format Requirements

1. **Audio files**: WAV format, 44.1kHz, mono or stereo
2. **Transcript files**: Plain text with speaker tags

## Example Structure

```
voices/
├── narrator.wav          # Audio file
├── narrator.txt          # Transcript: "[S1] Your transcript here..."
├── character_male.wav
├── character_male.txt
└── ...
```

## Speaker Tags

- `[S1]` - Primary speaker
- `[S2]` - Secondary speaker  
- Use consistent tags in your transcripts

## Tips

- Keep audio files 5-20 seconds long
- Ensure high audio quality (minimal background noise)
- Transcripts should match the audio exactly
- Use descriptive filenames (narrator, character_male, etc.)

These voices will be pre-processed during Docker build for fast serverless access.
"""
    
    with open(voices_dir / "README.md", "w") as f:
        f.write(readme_content)
    
    print("✅ Created voices directory with examples")

def print_deployment_instructions():
    """Print deployment instructions."""
    print("\n" + "="*60)
    print("🚀 RunPod Deployment Instructions")
    print("="*60)
    
    instructions = """
1. **Add Your Voices** (Optional but recommended):
   - Place .wav audio files in the `voices/` directory
   - Create corresponding .txt transcript files
   - Follow the format in voices/README.md

2. **Build Docker Image**:
   ```bash
   # Set your registry (DockerHub, GHCR, etc.)
   export DOCKER_REGISTRY=your-registry.com/your-username
   
   # Build and push
   ./build_docker.sh
   ```

3. **Test Locally** (Optional):
   ```bash
   # Install dependencies
   pip install -r runpod_requirements.txt
   
   # Run local test
   ./test_local.py
   ```

4. **Deploy to RunPod**:
   - Go to https://www.runpod.io/serverless
   - Create new endpoint
   - Use your Docker image
   - Recommended settings:
     * GPU: RTX 4090 or A100
     * Memory: 24GB+ VRAM
     * Container Disk: 20GB
     * Execution Timeout: 300s

5. **Test Your Deployment**:
   ```python
   from runpod_client import RunPodTTSClient
   
   client = RunPodTTSClient(
       endpoint_url="https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync",
       api_key="YOUR_API_KEY"
   )
   
   # Test generation
   audio_bytes = client.text_to_speech("Hello world!", voice="S1")
   client.save_audio(audio_bytes, "test.wav")
   ```

6. **Use in Production**:
   - See runpod_example.py for complete usage examples
   - Monitor costs and performance in RunPod dashboard
   - Scale automatically based on demand

🎉 Your Enhanced Dia TTS system will be ready for serverless deployment!
"""
    
    print(instructions)

def main():
    """Main setup function."""
    print("🚀 RunPod Enhanced Dia TTS Setup")
    print("="*50)
    
    # Check if we're in the right directory
    if not os.path.exists("enhanced_voice_library.py"):
        print("❌ Please run this script from the dia/ directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Missing dependencies. Please ensure all files are present.")
        sys.exit(1)
    
    # Create all necessary files
    create_config_file()
    create_voice_setup_script()
    create_docker_build_script()
    create_test_script()
    create_voices_directory()
    
    print("\n✅ Setup complete!")
    
    # Print instructions
    print_deployment_instructions()

if __name__ == "__main__":
    main()