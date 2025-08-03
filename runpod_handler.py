"""
RunPod Serverless Handler for Enhanced Dia TTS

This handler is optimized for RunPod serverless deployment:
- Direct function handling (no FastAPI server)
- Optimized cold start initialization
- Memory efficient operation
- OpenAI-compatible request format
"""

import os
import io
import json
import time
import logging
import traceback
from typing import Dict, Any, Optional
import base64

import torch
import numpy as np
import soundfile as sf
import runpod

# Import our enhanced TTS system
from enhanced_voice_library import (
    EnhancedVoiceLibrary, 
    EnhancedDiaWithVoiceLibrary,
    PerformanceMonitor
)
from audio_processing import AudioProcessor, quick_enhance_audio
from config_enhanced import get_config

# Configure logging for RunPod
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state for model (initialized once per worker)
enhanced_dia = None
voice_library = None
audio_processor = None
model_loaded = False
initialization_time = None

def initialize_system():
    """Initialize the TTS system (called once per worker instance)."""
    global enhanced_dia, voice_library, audio_processor, model_loaded, initialization_time
    
    if model_loaded:
        return
    
    start_time = time.time()
    logger.info("🚀 Initializing Enhanced Dia TTS for RunPod serverless...")
    
    try:
        # Get configuration
        config = get_config()
        
        # Setup directories
        library_path = config.get("voice_library", "library_path", "./voice_library")
        os.makedirs(library_path, exist_ok=True)
        os.makedirs(os.path.join(library_path, "profiles"), exist_ok=True)
        os.makedirs(os.path.join(library_path, "tokens"), exist_ok=True)
        
        # Configure PyTorch for serverless
        torch.set_grad_enabled(False)
        if torch.cuda.is_available():
            # Use most of available CUDA memory for serverless
            torch.cuda.set_per_process_memory_fraction(0.95)
            # Clear any existing cache
            torch.cuda.empty_cache()
        
        # Load Dia model
        from dia.model import Dia
        model_config = config.get("model")
        
        logger.info(f"Loading Dia model: {model_config['repo_id']}")
        dia_model = Dia.from_huggingface(
            repo_id=model_config["repo_id"],
            config_filename=model_config["config_filename"],
            weights_filename=model_config["weights_filename"],
            cache_dir=model_config["cache_dir"]
        )
        
        # Initialize voice library
        voice_library = EnhancedVoiceLibrary(library_path)
        voice_library._max_cache_size = config.get("voice_library", "max_cache_size", 5)  # Smaller for serverless
        
        # Initialize audio processor
        audio_processor = AudioProcessor(
            sample_rate=config.get("audio_processing", "sample_rate", 44100)
        )
        
        # Create enhanced wrapper
        enhanced_dia = EnhancedDiaWithVoiceLibrary(dia_model, voice_library)
        enhanced_dia.default_params.update(config.get("generation", "defaults", {}))
        
        # Preload voices if specified
        preload_voices = config.get("voice_library", "preload_voices", [])
        for voice_id in preload_voices:
            try:
                voice_library.load_voice(voice_id)
                logger.info(f"Preloaded voice: {voice_id}")
            except Exception as e:
                logger.warning(f"Failed to preload voice {voice_id}: {e}")
        
        model_loaded = True
        initialization_time = time.time() - start_time
        
        logger.info(f"✅ System initialized in {initialization_time:.2f}s")
        logger.info(f"🎵 Available voices: {len(voice_library.index)}")
        
        if torch.cuda.is_available():
            memory_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"💾 CUDA memory allocated: {memory_gb:.1f} GB")
        
    except Exception as e:
        logger.error(f"❌ Initialization failed: {e}")
        logger.error(traceback.format_exc())
        raise

def encode_audio_to_base64(audio_array: np.ndarray, sample_rate: int, format: str = "wav") -> str:
    """Encode audio array to base64 string."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_array, sample_rate, format="WAV")
    buffer.seek(0)
    audio_bytes = buffer.read()
    return base64.b64encode(audio_bytes).decode('utf-8')

def handle_openai_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle OpenAI-compatible TTS requests."""
    try:
        # Extract parameters with defaults
        text = input_data.get("input", "")
        voice = input_data.get("voice", "S1")
        response_format = input_data.get("response_format", "wav")
        speed = input_data.get("speed", 1.0)
        seed = input_data.get("seed", enhanced_dia.default_params.get("seed", 42))
        
        if not text:
            return {"error": "Missing 'input' text parameter"}
        
        logger.info(f"🎯 OpenAI request: voice={voice}, format={response_format}, speed={speed}")
        
        monitor = PerformanceMonitor()
        monitor.start("total_generation")
        
        # Determine voice mode and generate
        if voice.lower() in ["s1", "s2", "dialogue"]:
            # Handle built-in voices
            if voice.lower() == "s1":
                formatted_text = f"[S1] {text}"
            elif voice.lower() == "s2":
                formatted_text = f"[S2] {text}"
            else:
                formatted_text = text  # Assume dialogue format already present
            
            audio_result = enhanced_dia.optimized_model.generate_optimized(
                text=formatted_text,
                seed=seed
            )
            generation_info = {"voice_mode": voice.lower(), "chunking_enabled": False}
            
        else:
            # Assume it's a voice ID from library
            if voice not in voice_library.index:
                return {"error": f"Voice '{voice}' not found. Available: {list(voice_library.index.keys())}"}
            
            audio_result, generation_info = enhanced_dia.generate_with_voice(
                voice_id=voice,
                text=text,
                seed=seed
            )
        
        # Apply speed adjustment if needed
        if speed != 1.0:
            target_length = int(len(audio_result) / speed)
            audio_result = np.resize(audio_result, target_length)
        
        total_time = monitor.end("total_generation")
        
        # Encode audio
        audio_base64 = encode_audio_to_base64(
            audio_result, 
            enhanced_dia.default_params.get("sample_rate", 44100),
            response_format
        )
        
        return {
            "success": True,
            "audio_base64": audio_base64,
            "format": response_format,
            "generation_time": total_time,
            "audio_length": len(audio_result),
            "generation_info": generation_info
        }
        
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}

def handle_custom_request(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle custom TTS requests with full parameter control."""
    try:
        # Extract parameters
        text = input_data.get("text", "")
        voice_mode = input_data.get("voice_mode", "single_s1")
        voice_id = input_data.get("voice_id")
        output_format = input_data.get("output_format", "wav")
        
        # Generation parameters
        cfg_scale = input_data.get("cfg_scale", enhanced_dia.default_params.get("cfg_scale", 3.0))
        temperature = input_data.get("temperature", enhanced_dia.default_params.get("temperature", 1.3))
        top_p = input_data.get("top_p", enhanced_dia.default_params.get("top_p", 0.95))
        cfg_filter_top_k = input_data.get("cfg_filter_top_k", enhanced_dia.default_params.get("cfg_filter_top_k", 35))
        seed = input_data.get("seed", enhanced_dia.default_params.get("seed", 42))
        
        # Chunking parameters
        enable_chunking = input_data.get("enable_chunking", enhanced_dia.default_params.get("enable_chunking", True))
        chunk_size = input_data.get("chunk_size", enhanced_dia.default_params.get("chunk_size", 120))
        
        if not text:
            return {"error": "Missing 'text' parameter"}
        
        logger.info(f"🎯 Custom request: mode={voice_mode}, voice_id={voice_id}, chunking={enable_chunking}")
        
        monitor = PerformanceMonitor()
        monitor.start("total_generation")
        
        # Generate based on mode
        if voice_mode == "voice_library":
            if not voice_id:
                return {"error": "voice_id required for voice_library mode"}
            
            if voice_id not in voice_library.index:
                return {"error": f"Voice '{voice_id}' not found. Available: {list(voice_library.index.keys())}"}
            
            # Use voice library with all parameters
            audio_result, generation_info = enhanced_dia.generate_with_voice(
                voice_id=voice_id,
                text=text,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                seed=seed,
                chunk_size=chunk_size,
                enable_chunking=enable_chunking
            )
            
        else:
            # Direct generation for dialogue, single_s1, single_s2
            if voice_mode == "dialogue":
                formatted_text = text
            elif voice_mode == "single_s1":
                formatted_text = f"[S1] {text}"
            elif voice_mode == "single_s2":
                formatted_text = f"[S2] {text}"
            else:
                return {"error": f"Invalid voice_mode: {voice_mode}"}
            
            audio_result = enhanced_dia.optimized_model.generate_optimized(
                text=formatted_text,
                seed=seed,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k
            )
            
            generation_info = {
                "voice_mode": voice_mode,
                "chunking_enabled": False,
                "chunks_processed": 1
            }
        
        total_time = monitor.end("total_generation")
        
        # Add performance info
        generation_info["total_request_time"] = total_time
        generation_info["audio_duration_seconds"] = len(audio_result) / 44100
        
        # Encode audio
        audio_base64 = encode_audio_to_base64(audio_result, 44100, output_format)
        
        return {
            "success": True,
            "audio_base64": audio_base64,
            "format": output_format,
            "generation_time": total_time,
            "audio_length": len(audio_result),
            "generation_info": generation_info
        }
        
    except Exception as e:
        logger.error(f"Custom request failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}

def handle_voice_management(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle voice management operations."""
    try:
        action = input_data.get("action", "list")
        
        if action == "list":
            voices = []
            for voice_id, voice_data in voice_library.index.items():
                voice_info = voice_data.copy()
                voice_info["cached"] = voice_id in voice_library._loaded_profiles
                voices.append(voice_info)
            
            return {
                "success": True,
                "voices": voices,
                "total_voices": len(voices)
            }
        
        elif action == "add":
            voice_id = input_data.get("voice_id")
            audio_path = input_data.get("audio_path")
            reference_text = input_data.get("reference_text")
            metadata = input_data.get("metadata", {})
            
            if not all([voice_id, audio_path, reference_text]):
                return {"error": "Missing required parameters: voice_id, audio_path, reference_text"}
            
            if not os.path.exists(audio_path):
                return {"error": f"Audio file not found: {audio_path}"}
            
            profile = voice_library.add_voice_from_audio(
                enhanced_dia.optimized_model.dia_model,
                voice_id,
                audio_path,
                reference_text,
                metadata
            )
            
            return {
                "success": True,
                "message": f"Voice '{voice_id}' added successfully",
                "processing_time": profile.processing_time,
                "voice_info": profile.to_dict()
            }
        
        elif action == "info":
            voice_id = input_data.get("voice_id")
            if not voice_id:
                return {"error": "Missing voice_id parameter"}
            
            if voice_id not in voice_library.index:
                return {"error": f"Voice '{voice_id}' not found"}
            
            voice_info = voice_library.index[voice_id].copy()
            voice_info["cached"] = voice_id in voice_library._loaded_profiles
            
            return {
                "success": True,
                "voice_info": voice_info
            }
        
        else:
            return {"error": f"Invalid action: {action}. Supported: list, add, info"}
            
    except Exception as e:
        logger.error(f"Voice management failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}

def handle_system_info(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Handle system information requests."""
    try:
        request_type = input_data.get("type", "health")
        
        if request_type == "health":
            status = {
                "healthy": model_loaded,
                "model_loaded": model_loaded,
                "initialization_time": initialization_time,
                "voice_count": len(voice_library.index) if voice_library else 0,
                "cached_voices": len(voice_library._loaded_profiles) if voice_library else 0
            }
            
            if torch.cuda.is_available():
                status["cuda_memory_allocated"] = torch.cuda.memory_allocated() / 1024**3
                status["cuda_memory_reserved"] = torch.cuda.memory_reserved() / 1024**3
            
            return {"success": True, "status": status}
        
        elif request_type == "stats":
            if not model_loaded:
                return {"error": "System not initialized"}
            
            stats = enhanced_dia.get_system_stats()
            stats["initialization_time"] = initialization_time
            
            return {"success": True, "stats": stats}
        
        else:
            return {"error": f"Invalid request type: {request_type}. Supported: health, stats"}
            
    except Exception as e:
        logger.error(f"System info request failed: {e}")
        return {"error": str(e), "traceback": traceback.format_exc()}

def handler(event):
    """
    Main RunPod serverless handler.
    
    Expected input format:
    {
        "input": {
            "endpoint": "openai|custom|voice|system",
            "data": {...}  # Endpoint-specific data
        }
    }
    """
    try:
        # Initialize system if not already done
        if not model_loaded:
            initialize_system()
        
        # Extract input data
        input_data = event.get("input", {})
        endpoint = input_data.get("endpoint", "openai")
        data = input_data.get("data", {})
        
        logger.info(f"🎯 Handler called: endpoint={endpoint}")
        
        # Route to appropriate handler
        if endpoint == "openai":
            # OpenAI-compatible endpoint
            return handle_openai_request(data)
        
        elif endpoint == "custom":
            # Custom TTS endpoint
            return handle_custom_request(data)
        
        elif endpoint == "voice":
            # Voice management
            return handle_voice_management(data)
        
        elif endpoint == "system":
            # System information
            return handle_system_info(data)
        
        else:
            return {"error": f"Invalid endpoint: {endpoint}. Supported: openai, custom, voice, system"}
    
    except Exception as e:
        logger.error(f"Handler failed: {e}")
        logger.error(traceback.format_exc())
        return {
            "error": str(e),
            "traceback": traceback.format_exc(),
            "handler_error": True
        }

if __name__ == "__main__":
    # For local testing
    print("🧪 Testing RunPod handler locally...")
    
    # Test initialization
    initialize_system()
    
    # Test OpenAI request
    test_event = {
        "input": {
            "endpoint": "openai",
            "data": {
                "input": "Hello, this is a test of the RunPod serverless TTS system.",
                "voice": "S1",
                "response_format": "wav"
            }
        }
    }
    
    result = handler(test_event)
    if result.get("success"):
        print("✅ Test successful!")
        print(f"Generation time: {result['generation_time']:.2f}s")
        print(f"Audio length: {result['audio_length']} samples")
    else:
        print("❌ Test failed!")
        print(f"Error: {result.get('error')}")

# Start the RunPod serverless handler
runpod.serverless.start({"handler": handler})