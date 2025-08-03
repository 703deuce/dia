"""
Deployment and Integration Script for Enhanced Dia TTS

This script provides utilities for deploying the enhanced TTS system including:
- Model initialization and optimization
- Voice library setup
- Performance monitoring
- Serverless deployment helpers
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml

# Local imports
from config_enhanced import get_config, validate_config
from enhanced_voice_library import EnhancedVoiceLibrary, EnhancedDiaWithVoiceLibrary
from audio_processing import AudioProcessor, quick_enhance_audio
from serverless_api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DeploymentManager:
    """Manages deployment and initialization of the TTS system."""
    
    def __init__(self):
        self.config = get_config()
        self.dia_model = None
        self.enhanced_dia = None
        self.voice_library = None
        self.audio_processor = None
        
    @validate_config
    def initialize_system(self):
        """Initialize the complete TTS system."""
        logger.info("🚀 Initializing Enhanced Dia TTS System...")
        
        # Step 1: Setup directories
        self._setup_directories()
        
        # Step 2: Configure PyTorch
        self._configure_pytorch()
        
        # Step 3: Load model
        self._load_model()
        
        # Step 4: Initialize voice library
        self._initialize_voice_library()
        
        # Step 5: Setup audio processor
        self._setup_audio_processor()
        
        # Step 6: Create enhanced wrapper
        self._create_enhanced_wrapper()
        
        # Step 7: Optimization
        self._optimize_system()
        
        logger.info("✅ System initialization complete!")
        return self.enhanced_dia
    
    def _setup_directories(self):
        """Setup required directories."""
        directories = [
            self.config.get("model", "cache_dir"),
            self.config.get("voice_library", "library_path"),
            os.path.join(self.config.get("voice_library", "library_path"), "profiles"),
            os.path.join(self.config.get("voice_library", "library_path"), "tokens"),
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
    
    def _configure_pytorch(self):
        """Configure PyTorch for optimal performance."""
        config = self.config.get("performance", {})
        
        # Set CUDA memory fraction
        if torch.cuda.is_available() and config.get("cuda_memory_fraction", 0.9) < 1.0:
            torch.cuda.set_per_process_memory_fraction(config["cuda_memory_fraction"])
            logger.info(f"Set CUDA memory fraction to {config['cuda_memory_fraction']}")
        
        # Enable memory mapping
        if config.get("enable_memory_mapping", True):
            torch.backends.cuda.enable_memory_efficient_attention = True
            logger.debug("Enabled memory efficient attention")
        
        # Configure for inference
        torch.set_grad_enabled(False)
        
        # Set number of threads for CPU inference
        if not torch.cuda.is_available():
            torch.set_num_threads(min(4, os.cpu_count() or 1))
            logger.info(f"Set PyTorch threads to {torch.get_num_threads()}")
    
    def _load_model(self):
        """Load the Dia model with optimization."""
        from dia.model import Dia
        
        model_config = self.config.get("model")
        logger.info(f"Loading Dia model from {model_config['repo_id']}...")
        
        start_time = time.time()
        
        try:
            self.dia_model = Dia.from_huggingface(
                repo_id=model_config["repo_id"],
                config_filename=model_config["config_filename"],
                weights_filename=model_config["weights_filename"],
                cache_dir=model_config["cache_dir"]
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Model loaded successfully in {load_time:.2f}s")
            
            # Log model info
            device = getattr(self.dia_model, 'device', 'unknown')
            logger.info(f"Model device: {device}")
            
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 / 1024
                logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    def _initialize_voice_library(self):
        """Initialize the voice library."""
        library_config = self.config.get("voice_library")
        logger.info("Initializing voice library...")
        
        self.voice_library = EnhancedVoiceLibrary(
            library_path=library_config["library_path"]
        )
        self.voice_library._max_cache_size = library_config["max_cache_size"]
        
        # Preload voices if specified
        preload_voices = library_config.get("preload_voices", [])
        if preload_voices:
            logger.info(f"Preloading {len(preload_voices)} voices...")
            for voice_id in preload_voices:
                try:
                    self.voice_library.load_voice(voice_id)
                    logger.debug(f"Preloaded voice: {voice_id}")
                except Exception as e:
                    logger.warning(f"Failed to preload voice {voice_id}: {e}")
        
        logger.info(f"✅ Voice library initialized with {len(self.voice_library.index)} voices")
    
    def _setup_audio_processor(self):
        """Setup audio processor."""
        audio_config = self.config.get("audio_processing")
        self.audio_processor = AudioProcessor(
            sample_rate=audio_config["sample_rate"]
        )
        logger.info("✅ Audio processor initialized")
    
    def _create_enhanced_wrapper(self):
        """Create the enhanced Dia wrapper."""
        self.enhanced_dia = EnhancedDiaWithVoiceLibrary(
            self.dia_model, 
            self.voice_library
        )
        
        # Update default parameters from config
        generation_defaults = self.config.get("generation", "defaults")
        self.enhanced_dia.default_params.update(generation_defaults)
        
        logger.info("✅ Enhanced Dia wrapper created")
    
    def _optimize_system(self):
        """Apply system optimizations."""
        performance_config = self.config.get("performance", {})
        
        # Memory optimization
        if performance_config.get("memory_optimization", True):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            import gc
            gc.collect()
            logger.debug("Applied memory optimization")
        
        # Model compilation (if enabled)
        if self.config.get("model", "compile_model", False):
            try:
                compile_mode = self.config.get("model", "torch_compile_mode", "default")
                # Note: This would require torch.compile compatible model
                logger.info(f"Model compilation with mode '{compile_mode}' requested")
                # self.dia_model = torch.compile(self.dia_model, mode=compile_mode)
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information."""
        info = {
            "system": {
                "python_version": sys.version,
                "torch_version": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
            },
            "model": {
                "loaded": self.dia_model is not None,
                "device": str(getattr(self.dia_model, 'device', 'unknown')),
                "repo_id": self.config.get("model", "repo_id"),
                "weights_file": self.config.get("model", "weights_filename")
            },
            "voice_library": {
                "path": self.config.get("voice_library", "library_path"),
                "voice_count": len(self.voice_library.index) if self.voice_library else 0,
                "cached_voices": len(self.voice_library._loaded_profiles) if self.voice_library else 0
            },
            "memory": {}
        }
        
        if torch.cuda.is_available():
            info["memory"]["cuda_allocated"] = torch.cuda.memory_allocated() / 1024**3
            info["memory"]["cuda_reserved"] = torch.cuda.memory_reserved() / 1024**3
            info["memory"]["cuda_max_allocated"] = torch.cuda.max_memory_allocated() / 1024**3
        
        return info
    
    def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        status = {
            "healthy": True,
            "issues": [],
            "timestamp": time.time()
        }
        
        # Check model
        if not self.dia_model:
            status["healthy"] = False
            status["issues"].append("Model not loaded")
        
        # Check voice library
        if not self.voice_library:
            status["healthy"] = False
            status["issues"].append("Voice library not initialized")
        
        # Check CUDA memory (if available)
        if torch.cuda.is_available():
            memory_percent = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_percent > 0.9:
                status["issues"].append("High CUDA memory usage")
        
        # Check disk space
        library_path = self.config.get("voice_library", "library_path")
        if os.path.exists(library_path):
            import shutil
            total, used, free = shutil.disk_usage(library_path)
            free_percent = free / total
            if free_percent < 0.1:  # Less than 10% free
                status["issues"].append("Low disk space")
        
        if status["issues"]:
            status["healthy"] = False
        
        return status


def create_example_config():
    """Create an example configuration file."""
    config_path = "enhanced_tts_config.yaml"
    
    example_config = {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 1
        },
        "model": {
            "repo_id": "ttj/dia-1.6b-safetensors",
            "weights_filename": "dia-v0_1_bf16.safetensors",
            "cache_dir": "./model_cache"
        },
        "voice_library": {
            "library_path": "./voice_library",
            "max_cache_size": 10
        },
        "generation": {
            "defaults": {
                "cfg_scale": 3.0,
                "temperature": 1.3,
                "seed": 42,
                "enable_chunking": True
            }
        },
        "audio_processing": {
            "enable_enhancement": True,
            "enhancement_preset": "balanced"
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.safe_dump(example_config, f, indent=2)
    
    logger.info(f"Created example config at {config_path}")
    return config_path


def main():
    """Main deployment script."""
    parser = argparse.ArgumentParser(description="Enhanced Dia TTS Deployment")
    parser.add_argument("--init", action="store_true", help="Initialize the system")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--create-config", action="store_true", help="Create example config")
    parser.add_argument("--health-check", action="store_true", help="Run health check")
    parser.add_argument("--info", action="store_true", help="Show system info")
    parser.add_argument("--serve", action="store_true", help="Start the API server")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    # Set config file if provided
    if args.config:
        os.environ["CONFIG_FILE"] = args.config
    
    # Create example config
    if args.create_config:
        create_example_config()
        return
    
    # Initialize deployment manager
    manager = DeploymentManager()
    
    # Initialize system
    if args.init or args.serve or args.health_check or args.info:
        try:
            manager.initialize_system()
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            sys.exit(1)
    
    # Show system info
    if args.info:
        info = manager.get_system_info()
        print("\n🔍 System Information:")
        print(yaml.dump(info, indent=2))
    
    # Health check
    if args.health_check:
        health = manager.health_check()
        print(f"\n🏥 Health Check: {'✅ Healthy' if health['healthy'] else '❌ Issues Found'}")
        if health['issues']:
            for issue in health['issues']:
                print(f"  - {issue}")
    
    # Start server
    if args.serve:
        logger.info(f"🌐 Starting API server at {args.host}:{args.port}")
        
        # Update global state for the API
        import serverless_api
        serverless_api.enhanced_dia = manager.enhanced_dia
        serverless_api.voice_library = manager.voice_library
        serverless_api.model_loaded = True
        
        # Start server
        import uvicorn
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            log_level="info",
            access_log=True
        )


if __name__ == "__main__":
    main()