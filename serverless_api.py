"""
Serverless-Optimized FastAPI Server for Enhanced Dia TTS

This module provides a lean, serverless-ready API server with:
- OpenAI-compatible endpoints
- Voice library integration
- Memory optimization
- Configuration management
- No UI dependencies
"""

import os
import io
import json
import time
import uuid
import logging
from contextlib import asynccontextmanager
from typing import Optional, Literal, Dict, Any, List, Union
from pathlib import Path

import torch
import numpy as np
import soundfile as sf
from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn

from dia.model import Dia
from enhanced_voice_library import (
    EnhancedVoiceLibrary, 
    EnhancedDiaWithVoiceLibrary,
    PerformanceMonitor
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
enhanced_dia = None
voice_library = None
model_loaded = False

class TTSConfig:
    """Configuration management for the TTS API."""
    
    def __init__(self):
        self.model_config = {
            "repo_id": os.getenv("DIA_MODEL_REPO_ID", "ttj/dia-1.6b-safetensors"),
            "weights_filename": os.getenv("DIA_MODEL_WEIGHTS_FILENAME", "dia-v0_1_bf16.safetensors"),
            "config_filename": os.getenv("DIA_MODEL_CONFIG_FILENAME", "config.json"),
            "cache_dir": os.getenv("MODEL_CACHE_DIR", "./model_cache")
        }
        
        self.voice_library_config = {
            "library_path": os.getenv("VOICE_LIBRARY_PATH", "./voice_library"),
            "max_cache_size": int(os.getenv("VOICE_CACHE_SIZE", "10"))
        }
        
        self.generation_defaults = {
            "cfg_scale": float(os.getenv("DEFAULT_CFG_SCALE", "3.0")),
            "temperature": float(os.getenv("DEFAULT_TEMPERATURE", "1.3")),
            "top_p": float(os.getenv("DEFAULT_TOP_P", "0.95")),
            "cfg_filter_top_k": int(os.getenv("DEFAULT_CFG_FILTER_TOP_K", "35")),
            "seed": int(os.getenv("DEFAULT_SEED", "42")),
            "chunk_size": int(os.getenv("DEFAULT_CHUNK_SIZE", "120")),
            "enable_chunking": os.getenv("DEFAULT_ENABLE_CHUNKING", "true").lower() == "true"
        }
        
        self.server_config = {
            "host": os.getenv("HOST", "0.0.0.0"),
            "port": int(os.getenv("PORT", "8000")),
            "sample_rate": int(os.getenv("SAMPLE_RATE", "44100"))
        }

config = TTSConfig()

# Pydantic models for API
class OpenAITTSRequest(BaseModel):
    """OpenAI-compatible TTS request."""
    model: str = Field(default="dia-tts", description="Model identifier")
    input: str = Field(..., description="Text to convert to speech")
    voice: str = Field(default="S1", description="Voice to use (S1, S2, dialogue, or voice_id)")
    response_format: Literal["mp3", "opus", "aac", "flac", "wav"] = Field(default="mp3")
    speed: float = Field(default=1.0, ge=0.25, le=4.0, description="Playback speed")
    seed: Optional[int] = Field(default=None, description="Seed for reproducible output")

class CustomTTSRequest(BaseModel):
    """Custom TTS request with full parameter control."""
    text: str = Field(..., description="Text to convert to speech")
    voice_mode: Literal["dialogue", "single_s1", "single_s2", "voice_library"] = Field(
        default="single_s1", description="Voice generation mode"
    )
    voice_id: Optional[str] = Field(default=None, description="Voice ID from library (for voice_library mode)")
    output_format: Literal["wav", "mp3", "opus"] = Field(default="wav")
    
    # Generation parameters
    cfg_scale: float = Field(default=3.0, ge=1.0, le=10.0)
    temperature: float = Field(default=1.3, ge=0.1, le=2.0)
    top_p: float = Field(default=0.95, ge=0.1, le=1.0)
    cfg_filter_top_k: int = Field(default=35, ge=1, le=100)
    seed: Optional[int] = Field(default=None, description="Seed for reproducible output")
    
    # Chunking parameters
    enable_chunking: bool = Field(default=True, description="Enable text chunking for long inputs")
    chunk_size: int = Field(default=120, ge=50, le=500, description="Target chunk size in characters")

class TTSResponse(BaseModel):
    """Response model for TTS generation info."""
    success: bool
    message: str
    generation_info: Optional[Dict] = None
    performance_stats: Optional[Dict] = None

class VoiceInfo(BaseModel):
    """Voice information model."""
    voice_id: str
    reference_text: str
    duration_tokens: int
    usage_count: int
    cached: bool
    metadata: Dict

class SystemStats(BaseModel):
    """System statistics model."""
    model_loaded: bool
    voice_library_stats: Dict
    memory_info: Dict
    uptime_seconds: float

# Audio encoding utilities
def encode_audio(audio_array: np.ndarray, sample_rate: int, format: str) -> bytes:
    """Encode audio array to specified format."""
    buffer = io.BytesIO()
    
    if format.lower() in ["wav"]:
        sf.write(buffer, audio_array, sample_rate, format="WAV")
    elif format.lower() in ["mp3"]:
        # For MP3, we'll use wav for now (requires additional dependencies for true MP3)
        sf.write(buffer, audio_array, sample_rate, format="WAV")
    elif format.lower() in ["opus"]:
        # For Opus, we'll use wav for now (requires additional dependencies for true Opus)
        sf.write(buffer, audio_array, sample_rate, format="WAV")
    else:
        sf.write(buffer, audio_array, sample_rate, format="WAV")
    
    buffer.seek(0)
    return buffer.read()

def get_content_type(format: str) -> str:
    """Get content type for audio format."""
    format_map = {
        "wav": "audio/wav",
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac"
    }
    return format_map.get(format.lower(), "audio/wav")

# Model loading
async def load_model():
    """Load the Dia model and initialize voice library."""
    global enhanced_dia, voice_library, model_loaded
    
    if model_loaded:
        return
    
    try:
        logger.info("Loading Dia model...")
        
        # Initialize Dia model
        dia_model = Dia.from_huggingface(
            repo_id=config.model_config["repo_id"],
            config_filename=config.model_config["config_filename"],
            weights_filename=config.model_config["weights_filename"],
            cache_dir=config.model_config["cache_dir"]
        )
        
        # Initialize voice library
        logger.info("Initializing voice library...")
        voice_library = EnhancedVoiceLibrary(
            library_path=config.voice_library_config["library_path"]
        )
        voice_library._max_cache_size = config.voice_library_config["max_cache_size"]
        
        # Create enhanced wrapper
        enhanced_dia = EnhancedDiaWithVoiceLibrary(dia_model, voice_library)
        enhanced_dia.default_params.update(config.generation_defaults)
        
        model_loaded = True
        logger.info("Model and voice library loaded successfully!")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model_loaded = False
        raise

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    # Startup
    start_time = time.time()
    app.state.start_time = start_time
    
    try:
        await load_model()
        logger.info("API server ready!")
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        # In serverless, we might want to fail fast
        raise
    
    yield
    
    # Shutdown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("API server shutdown")

# Create FastAPI app
app = FastAPI(
    title="Enhanced Dia TTS API",
    description="Serverless-optimized TTS API with voice library and chunking support",
    version="1.0.0",
    lifespan=lifespan
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy" if model_loaded else "loading",
        "model_loaded": model_loaded,
        "timestamp": time.time()
    }

# System stats endpoint
@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    uptime = time.time() - app.state.start_time
    stats = enhanced_dia.get_system_stats()
    
    return SystemStats(
        model_loaded=model_loaded,
        voice_library_stats=stats["voice_library"],
        memory_info=stats["memory_info"],
        uptime_seconds=uptime
    )

# Voice management endpoints
@app.get("/voices", response_model=List[VoiceInfo])
async def list_voices():
    """List available voices."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    voices_data = enhanced_dia.list_voices()
    return [VoiceInfo(**voice_data) for voice_data in voices_data]

@app.post("/voices/{voice_id}")
async def add_voice(
    voice_id: str,
    audio_file_path: str,
    reference_text: str,
    metadata: Optional[Dict] = None
):
    """Add a new voice to the library."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not os.path.exists(audio_file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    try:
        profile = voice_library.add_voice_from_audio(
            enhanced_dia.optimized_model.dia_model,
            voice_id,
            audio_file_path,
            reference_text,
            metadata
        )
        
        return {
            "success": True,
            "message": f"Voice '{voice_id}' added successfully",
            "processing_time": profile.processing_time
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# OpenAI-compatible endpoint
@app.post("/v1/audio/speech")
async def openai_speech_endpoint(request: OpenAITTSRequest):
    """OpenAI-compatible speech synthesis endpoint."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    monitor = PerformanceMonitor()
    monitor.start("total_request")
    
    try:
        # Map voice parameter
        voice_mode = "single_s1"
        voice_id = None
        
        if request.voice.lower() == "s1":
            voice_mode = "single_s1"
        elif request.voice.lower() == "s2":
            voice_mode = "single_s2"
        elif request.voice.lower() == "dialogue":
            voice_mode = "dialogue"
        else:
            # Assume it's a voice ID from library
            voice_mode = "voice_library"
            voice_id = request.voice
            
            # Check if voice exists
            if voice_id not in voice_library.index:
                raise HTTPException(
                    status_code=404, 
                    detail=f"Voice '{voice_id}' not found in library"
                )
        
        # Generate audio
        if voice_mode == "voice_library":
            audio_result, generation_info = enhanced_dia.generate_with_voice(
                voice_id=voice_id,
                text=request.input,
                seed=request.seed if request.seed is not None else config.generation_defaults["seed"]
            )
        else:
            # Handle non-library modes (S1, S2, dialogue)
            # For now, we'll use a simple approach
            audio_result = enhanced_dia.optimized_model.generate_optimized(
                text=f"[{voice_mode.upper()}] {request.input}",
                seed=request.seed if request.seed is not None else config.generation_defaults["seed"]
            )
            generation_info = {"voice_mode": voice_mode}
        
        # Apply speed adjustment if needed
        if request.speed != 1.0:
            # Simple speed adjustment (could be improved with proper resampling)
            target_length = int(len(audio_result) / request.speed)
            audio_result = np.resize(audio_result, target_length)
        
        # Encode audio
        encoded_audio = encode_audio(
            audio_result, 
            config.server_config["sample_rate"], 
            request.response_format
        )
        
        total_time = monitor.end("total_request")
        
        # Log performance
        logger.info(
            f"OpenAI request completed in {total_time:.2f}s: "
            f"voice={request.voice}, format={request.response_format}, "
            f"input_length={len(request.input)}, output_length={len(audio_result)}"
        )
        
        return StreamingResponse(
            io.BytesIO(encoded_audio),
            media_type=get_content_type(request.response_format),
            headers={
                "Content-Disposition": f"attachment; filename=speech.{request.response_format}",
                "X-Generation-Time": str(total_time),
                "X-Audio-Length": str(len(audio_result))
            }
        )
        
    except Exception as e:
        logger.error(f"OpenAI request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Custom TTS endpoint
@app.post("/tts")
async def custom_tts_endpoint(request: CustomTTSRequest):
    """Custom TTS endpoint with full parameter control."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    monitor = PerformanceMonitor()
    monitor.start("total_request")
    
    try:
        # Generate audio based on mode
        if request.voice_mode == "voice_library":
            if not request.voice_id:
                raise HTTPException(status_code=400, detail="voice_id required for voice_library mode")
            
            if request.voice_id not in voice_library.index:
                raise HTTPException(status_code=404, detail=f"Voice '{request.voice_id}' not found")
            
            # Use voice library
            audio_result, generation_info = enhanced_dia.generate_with_voice(
                voice_id=request.voice_id,
                text=request.text,
                cfg_scale=request.cfg_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                cfg_filter_top_k=request.cfg_filter_top_k,
                seed=request.seed if request.seed is not None else config.generation_defaults["seed"],
                chunk_size=request.chunk_size,
                enable_chunking=request.enable_chunking
            )
        else:
            # Direct generation
            formatted_text = f"[{request.voice_mode.replace('single_', '').upper()}] {request.text}"
            
            audio_result = enhanced_dia.optimized_model.generate_optimized(
                text=formatted_text,
                seed=request.seed if request.seed is not None else config.generation_defaults["seed"],
                cfg_scale=request.cfg_scale,
                temperature=request.temperature,
                top_p=request.top_p,
                cfg_filter_top_k=request.cfg_filter_top_k
            )
            
            generation_info = {
                "voice_mode": request.voice_mode,
                "chunking_enabled": False,
                "chunks_processed": 1
            }
        
        # Encode audio
        encoded_audio = encode_audio(
            audio_result,
            config.server_config["sample_rate"],
            request.output_format
        )
        
        total_time = monitor.end("total_request")
        
        # Add performance info
        generation_info["total_request_time"] = total_time
        generation_info["audio_duration_seconds"] = len(audio_result) / config.server_config["sample_rate"]
        
        # Log performance
        logger.info(
            f"Custom TTS completed in {total_time:.2f}s: "
            f"mode={request.voice_mode}, voice_id={request.voice_id}, "
            f"chunking={request.enable_chunking}, chunks={generation_info.get('chunks_processed', 1)}"
        )
        
        return StreamingResponse(
            io.BytesIO(encoded_audio),
            media_type=get_content_type(request.output_format),
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.{request.output_format}",
                "X-Generation-Info": json.dumps(generation_info),
                "X-Generation-Time": str(total_time)
            }
        )
        
    except Exception as e:
        logger.error(f"Custom TTS request failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Text analysis endpoint
@app.post("/analyze")
async def analyze_text(text: str, chunk_size: int = 120):
    """Analyze text for chunking without generating audio."""
    from enhanced_voice_library import TextChunker
    
    chunker = TextChunker()
    chunks = chunker.chunk_text_by_sentences(text, chunk_size)
    
    return {
        "original_length": len(text),
        "num_chunks": len(chunks),
        "chunks": chunks,
        "estimated_generation_time": len(chunks) * 2.5,  # rough estimate
        "recommendations": {
            "enable_chunking": len(text) > chunk_size * 2,
            "optimal_chunk_size": min(max(len(text) // 5, 80), 200) if len(text) > 500 else chunk_size
        }
    }

# Configuration endpoint
@app.get("/config")
async def get_config():
    """Get current configuration."""
    return {
        "generation_defaults": config.generation_defaults,
        "server_config": config.server_config,
        "model_config": {k: v for k, v in config.model_config.items() if "cache" not in k.lower()}
    }

@app.post("/config")
async def update_config(new_config: Dict[str, Any]):
    """Update configuration (runtime changes only)."""
    try:
        if "generation_defaults" in new_config:
            config.generation_defaults.update(new_config["generation_defaults"])
            if enhanced_dia:
                enhanced_dia.default_params.update(config.generation_defaults)
        
        return {"success": True, "message": "Configuration updated"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    # For local development
    uvicorn.run(
        "serverless_api:app",
        host=config.server_config["host"],
        port=config.server_config["port"],
        reload=False,
        log_level="info"
    )