# 🚀 RunPod Serverless Deployment Guide

**Deploy Enhanced Dia TTS on RunPod Serverless with optimized cold starts and direct handler functions.**

## 🎯 Why This Approach?

✅ **No FastAPI server overhead** - Direct function calls  
✅ **Optimized cold starts** - Model loads once per worker  
✅ **Memory efficient** - Perfect for RunPod GPU instances  
✅ **Cost effective** - Pay only for inference time  
✅ **Auto-scaling** - Handles traffic spikes automatically  

---

## 🚀 Quick Deployment

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -f Dockerfile.runpod -t your-registry/enhanced-dia-tts:latest .

# Push to your container registry
docker push your-registry/enhanced-dia-tts:latest
```

### 2. Deploy on RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/serverless)
2. Create new endpoint
3. Use your Docker image: `your-registry/enhanced-dia-tts:latest`
4. Configure:
   - **GPU**: RTX 4090 or A100 (recommended)
   - **Memory**: 24GB+ VRAM
   - **Container Disk**: 20GB
   - **Execution Timeout**: 300 seconds

### 3. Environment Variables (Optional)

```bash
DIA_MODEL_REPO_ID=ttj/dia-1.6b-safetensors
DIA_MODEL_WEIGHTS_FILENAME=dia-v0_1_bf16.safetensors
VOICE_CACHE_SIZE=5
MEMORY_OPTIMIZATION=true
DEFAULT_SEED=42
```

---

## 📡 API Usage

### OpenAI-Compatible Endpoint

```python
import requests
import base64

# Your RunPod endpoint URL
endpoint_url = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
headers = {"Authorization": "Bearer YOUR_API_KEY"}

# OpenAI-compatible request
payload = {
    "input": {
        "endpoint": "openai",
        "data": {
            "input": "Hello, this is a test of the enhanced TTS system with intelligent chunking.",
            "voice": "S1",  # or your voice_id
            "response_format": "wav",
            "speed": 1.0,
            "seed": 42
        }
    }
}

response = requests.post(endpoint_url, json=payload, headers=headers)
result = response.json()

if result["status"] == "COMPLETED":
    # Decode base64 audio
    audio_base64 = result["output"]["audio_base64"]
    audio_bytes = base64.b64decode(audio_base64)
    
    # Save audio file
    with open("output.wav", "wb") as f:
        f.write(audio_bytes)
    
    print(f"✅ Generated in {result['output']['generation_time']:.2f}s")
else:
    print(f"❌ Error: {result}")
```

### Custom TTS Endpoint (Full Control)

```python
# Custom request with all parameters
payload = {
    "input": {
        "endpoint": "custom",
        "data": {
            "text": "This is a longer text that will be intelligently chunked while maintaining perfect voice consistency across all segments using the enhanced voice library system.",
            "voice_mode": "voice_library",  # or "single_s1", "single_s2", "dialogue"
            "voice_id": "narrator_deep",    # Your voice ID
            "enable_chunking": True,
            "chunk_size": 120,
            "cfg_scale": 3.0,
            "temperature": 1.3,
            "top_p": 0.95,
            "cfg_filter_top_k": 35,
            "seed": 42,
            "output_format": "wav"
        }
    }
}

response = requests.post(endpoint_url, json=payload, headers=headers)
result = response.json()

if result["status"] == "COMPLETED":
    output = result["output"]
    print(f"✅ Generated {output['generation_info']['chunks_processed']} chunks")
    print(f"⏱️  Total time: {output['generation_time']:.2f}s")
    print(f"🎵 Audio duration: {output['generation_info']['audio_duration_seconds']:.1f}s")
```

### Voice Management

```python
# List available voices
payload = {
    "input": {
        "endpoint": "voice",
        "data": {
            "action": "list"
        }
    }
}

response = requests.post(endpoint_url, json=payload, headers=headers)
voices = response.json()["output"]["voices"]
print(f"Available voices: {[v['voice_id'] for v in voices]}")

# Add a new voice (requires audio file in container)
payload = {
    "input": {
        "endpoint": "voice",
        "data": {
            "action": "add",
            "voice_id": "my_narrator",
            "audio_path": "/app/audio/narrator.wav",  # Path in container
            "reference_text": "[S1] This is a sample narration with professional tone.",
            "metadata": {"style": "narrative", "gender": "male"}
        }
    }
}
```

### System Information

```python
# Health check
payload = {
    "input": {
        "endpoint": "system",
        "data": {
            "type": "health"
        }
    }
}

response = requests.post(endpoint_url, json=payload, headers=headers)
health = response.json()["output"]["status"]
print(f"System healthy: {health['healthy']}")
print(f"Voices available: {health['voice_count']}")

# Performance stats
payload = {
    "input": {
        "endpoint": "system",
        "data": {
            "type": "stats"
        }
    }
}
```

---

## 🎵 Voice Library Setup

### Pre-build Voices (Recommended)

Create voices during Docker build for fastest cold starts:

```dockerfile
# Add to Dockerfile.runpod
COPY voices/ /app/voices/
RUN python3 -c "
from enhanced_voice_library import EnhancedVoiceLibrary
from dia.model import Dia
import os

# Initialize system
dia_model = Dia.from_huggingface('ttj/dia-1.6b-safetensors', cache_dir='/app/model_cache')
library = EnhancedVoiceLibrary('/app/voice_library')

# Add all voices in voices directory
for voice_file in os.listdir('/app/voices'):
    if voice_file.endswith('.wav'):
        voice_id = voice_file.replace('.wav', '')
        txt_file = f'/app/voices/{voice_id}.txt'
        if os.path.exists(txt_file):
            with open(txt_file, 'r') as f:
                transcript = f.read().strip()
            library.add_voice_from_audio(dia_model, voice_id, f'/app/voices/{voice_file}', transcript)
            print(f'Added voice: {voice_id}')
"
```

### Runtime Voice Addition

```python
# Function to add voice at runtime
def add_voice_to_runpod(voice_id, audio_base64, reference_text):
    """Add voice by uploading audio as base64"""
    
    # First, decode and save the audio file
    import tempfile
    import base64
    
    audio_bytes = base64.b64decode(audio_base64)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        temp_path = tmp_file.name
    
    # Add voice via API
    payload = {
        "input": {
            "endpoint": "voice",
            "data": {
                "action": "add",
                "voice_id": voice_id,
                "audio_path": temp_path,
                "reference_text": reference_text
            }
        }
    }
    
    response = requests.post(endpoint_url, json=payload, headers=headers)
    return response.json()
```

---

## ⚡ Performance Optimization

### Cold Start Optimization

The handler is designed for fast cold starts:

1. **Model caching**: Models download once per worker
2. **Voice preloading**: Frequently used voices cached in memory
3. **Memory optimization**: Efficient VRAM usage
4. **Lazy loading**: Only load components when needed

### Expected Performance

| Scenario | Cold Start | Warm Generation |
|----------|------------|-----------------|
| **First request** | ~15-30s | - |
| **Short text (<200 chars)** | - | ~2-4s |
| **Long text (1000+ chars)** | - | ~8-15s |
| **Cached voice** | - | ~1-3s |

### Memory Usage

- **Model**: ~6-7GB VRAM (BF16)
- **Voice cache**: ~500MB per cached voice
- **Processing**: ~1-2GB temporary

---

## 🛠️ Advanced Configuration

### Environment Variables

```bash
# Model Configuration
DIA_MODEL_REPO_ID=ttj/dia-1.6b-safetensors
DIA_MODEL_WEIGHTS_FILENAME=dia-v0_1_bf16.safetensors
DIA_MODEL_CONFIG_FILENAME=config.json
MODEL_CACHE_DIR=/app/model_cache

# Voice Library
VOICE_LIBRARY_PATH=/app/voice_library
VOICE_CACHE_SIZE=5
PRELOAD_VOICES=narrator,character1,character2

# Performance
MEMORY_OPTIMIZATION=true
CUDA_MEMORY_FRACTION=0.95
ENABLE_PROFILING=false

# Generation Defaults
DEFAULT_CFG_SCALE=3.0
DEFAULT_TEMPERATURE=1.3
DEFAULT_TOP_P=0.95
DEFAULT_SEED=42
DEFAULT_ENABLE_CHUNKING=true
DEFAULT_CHUNK_SIZE=120

# Limits
MAX_TEXT_LENGTH=10000
MAX_GENERATION_TIME=300
```

### Custom Configuration File

Create `enhanced_tts_config.yaml` in your container:

```yaml
model:
  repo_id: "ttj/dia-1.6b-safetensors"
  weights_filename: "dia-v0_1_bf16.safetensors"
  cache_dir: "/app/model_cache"

voice_library:
  library_path: "/app/voice_library"
  max_cache_size: 5
  preload_voices: ["narrator", "character1"]

generation:
  defaults:
    enable_chunking: true
    chunk_size: 120
    seed: 42
  limits:
    max_text_length: 10000

performance:
  memory_optimization: true
  cuda_memory_fraction: 0.95
```

---

## 🧪 Testing

### Local Testing

```bash
# Test the handler locally
python runpod_handler.py
```

### RunPod Testing

```python
import requests

# Test basic functionality
def test_runpod_endpoint(endpoint_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    
    # Test health check
    payload = {
        "input": {
            "endpoint": "system",
            "data": {"type": "health"}
        }
    }
    
    response = requests.post(endpoint_url, json=payload, headers=headers)
    health = response.json()
    
    if health["status"] == "COMPLETED":
        print("✅ Health check passed")
        return True
    else:
        print(f"❌ Health check failed: {health}")
        return False

# Test generation
def test_generation(endpoint_url, api_key):
    headers = {"Authorization": f"Bearer {api_key}"}
    
    payload = {
        "input": {
            "endpoint": "openai",
            "data": {
                "input": "This is a test of the RunPod serverless TTS system.",
                "voice": "S1"
            }
        }
    }
    
    response = requests.post(endpoint_url, json=payload, headers=headers)
    result = response.json()
    
    if result["status"] == "COMPLETED":
        print(f"✅ Generation successful in {result['output']['generation_time']:.2f}s")
        return True
    else:
        print(f"❌ Generation failed: {result}")
        return False
```

---

## 🔧 Troubleshooting

### Common Issues

**1. Cold Start Timeout**
```bash
# Increase timeout in RunPod settings
# Or reduce model size:
DIA_MODEL_WEIGHTS_FILENAME=dia-v0_1_bf16.safetensors  # Instead of full precision
```

**2. Memory Issues**
```bash
# Reduce memory usage:
VOICE_CACHE_SIZE=3
CUDA_MEMORY_FRACTION=0.8
```

**3. Model Download Issues**
```bash
# Pre-download in Dockerfile:
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('ttj/dia-1.6b-safetensors', 'dia-v0_1_bf16.safetensors', cache_dir='/app/model_cache')"
```

**4. Voice Not Found**
```python
# Check available voices:
payload = {
    "input": {
        "endpoint": "voice",
        "data": {"action": "list"}
    }
}
```

### Error Handling

The handler includes comprehensive error handling:

```python
# All responses include error information
{
    "success": false,
    "error": "Detailed error message",
    "traceback": "Full traceback for debugging"
}
```

---

## 📊 Cost Optimization

### RunPod Pricing Tips

1. **Use BF16 models** - 50% less VRAM = smaller instances
2. **Optimize voice cache** - Reduce cold starts
3. **Batch requests** - Process multiple texts together
4. **Use appropriate GPU** - RTX 4090 vs A100 based on needs

### Expected Costs

| GPU | VRAM | Cost/min | Best For |
|-----|------|----------|----------|
| RTX 4090 | 24GB | ~$0.40 | Production |
| RTX 3090 | 24GB | ~$0.30 | Development |
| A100 | 40GB | ~$1.20 | High throughput |

---

## 🚀 Production Tips

1. **Pre-build voices** in Docker image for fastest cold starts
2. **Use health checks** to monitor system status
3. **Implement retries** for cold start scenarios
4. **Monitor costs** with RunPod analytics
5. **Scale based on demand** using RunPod auto-scaling

---

**Your enhanced TTS system is now ready for RunPod serverless deployment! 🎉**

The handler approach eliminates server overhead while providing all the advanced features of your voice library system with intelligent chunking and memory optimization.