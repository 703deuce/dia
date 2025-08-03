"""
Enhanced Configuration Management for Dia TTS API

This module provides centralized configuration management with:
- Environment variable support
- Runtime configuration updates
- Performance tuning
- Serverless optimizations
"""

import os
import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """Centralized configuration management."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or os.getenv("CONFIG_FILE", "enhanced_tts_config.yaml")
        self.config = self._load_default_config()
        self._load_from_file()
        self._load_from_env()
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            "server": {
                "host": "0.0.0.0",
                "port": 8000,
                "workers": 1,
                "timeout": 300,
                "max_request_size": 100 * 1024 * 1024,  # 100MB
                "cors_enabled": True,
                "cors_origins": ["*"]
            },
            "model": {
                "repo_id": "ttj/dia-1.6b-safetensors",
                "weights_filename": "dia-v0_1_bf16.safetensors",
                "config_filename": "config.json",
                "cache_dir": "./model_cache",
                "device": "auto",  # auto, cpu, cuda, mps
                "dtype": "bfloat16",  # float32, float16, bfloat16
                "compile_model": False,
                "torch_compile_mode": "default"
            },
            "voice_library": {
                "library_path": "./voice_library",
                "max_cache_size": 10,
                "preload_voices": [],  # List of voice IDs to preload
                "auto_cleanup": True,
                "backup_enabled": False,
                "backup_interval_hours": 24
            },
            "generation": {
                "defaults": {
                    "cfg_scale": 3.0,
                    "temperature": 1.3,
                    "top_p": 0.95,
                    "cfg_filter_top_k": 35,
                    "seed": 42,
                    "chunk_size": 120,
                    "enable_chunking": True,
                    "max_chunk_count": 50
                },
                "limits": {
                    "max_text_length": 10000,
                    "max_generation_time": 300,
                    "concurrent_requests": 3
                }
            },
            "audio_processing": {
                "sample_rate": 44100,
                "enable_enhancement": True,
                "enhancement_preset": "balanced",  # minimal, balanced, aggressive
                "enable_trimming": True,
                "enable_normalization": True,
                "enable_filtering": True,
                "enable_silence_reduction": True,
                "enable_click_removal": False,
                "output_formats": ["wav", "mp3", "opus"]
            },
            "performance": {
                "memory_optimization": True,
                "cuda_memory_fraction": 0.9,
                "enable_memory_mapping": True,
                "garbage_collection_threshold": 5,
                "cache_cleanup_interval": 300,
                "enable_profiling": False
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,  # Set to file path to enable file logging
                "max_file_size": 10 * 1024 * 1024,  # 10MB
                "backup_count": 5,
                "enable_request_logging": True,
                "enable_performance_logging": True
            },
            "security": {
                "enable_rate_limiting": True,
                "rate_limit_requests": 60,
                "rate_limit_window": 60,
                "enable_api_key": False,
                "api_key": None,
                "allowed_ips": [],  # Empty list means all IPs allowed
                "enable_request_validation": True
            },
            "monitoring": {
                "enable_metrics": True,
                "metrics_endpoint": "/metrics",
                "enable_health_checks": True,
                "health_check_interval": 30,
                "enable_alerts": False,
                "alert_webhook": None
            }
        }
    
    def _load_from_file(self):
        """Load configuration from YAML file."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    file_config = yaml.safe_load(f)
                    self._deep_merge(self.config, file_config)
                logger.info(f"Loaded configuration from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config file {self.config_file}: {e}")
    
    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            # Server
            "HOST": ("server", "host"),
            "PORT": ("server", "port", int),
            
            # Model
            "DIA_MODEL_REPO_ID": ("model", "repo_id"),
            "DIA_MODEL_WEIGHTS_FILENAME": ("model", "weights_filename"),
            "DIA_MODEL_CONFIG_FILENAME": ("model", "config_filename"),
            "MODEL_CACHE_DIR": ("model", "cache_dir"),
            "MODEL_DEVICE": ("model", "device"),
            "MODEL_DTYPE": ("model", "dtype"),
            
            # Voice Library
            "VOICE_LIBRARY_PATH": ("voice_library", "library_path"),
            "VOICE_CACHE_SIZE": ("voice_library", "max_cache_size", int),
            
            # Generation Defaults
            "DEFAULT_CFG_SCALE": ("generation", "defaults", "cfg_scale", float),
            "DEFAULT_TEMPERATURE": ("generation", "defaults", "temperature", float),
            "DEFAULT_TOP_P": ("generation", "defaults", "top_p", float),
            "DEFAULT_CFG_FILTER_TOP_K": ("generation", "defaults", "cfg_filter_top_k", int),
            "DEFAULT_SEED": ("generation", "defaults", "seed", int),
            "DEFAULT_CHUNK_SIZE": ("generation", "defaults", "chunk_size", int),
            "DEFAULT_ENABLE_CHUNKING": ("generation", "defaults", "enable_chunking", bool),
            
            # Limits
            "MAX_TEXT_LENGTH": ("generation", "limits", "max_text_length", int),
            "MAX_GENERATION_TIME": ("generation", "limits", "max_generation_time", int),
            "CONCURRENT_REQUESTS": ("generation", "limits", "concurrent_requests", int),
            
            # Audio Processing
            "SAMPLE_RATE": ("audio_processing", "sample_rate", int),
            "ENABLE_AUDIO_ENHANCEMENT": ("audio_processing", "enable_enhancement", bool),
            "ENHANCEMENT_PRESET": ("audio_processing", "enhancement_preset"),
            
            # Performance
            "MEMORY_OPTIMIZATION": ("performance", "memory_optimization", bool),
            "CUDA_MEMORY_FRACTION": ("performance", "cuda_memory_fraction", float),
            "ENABLE_PROFILING": ("performance", "enable_profiling", bool),
            
            # Logging
            "LOG_LEVEL": ("logging", "level"),
            "LOG_FILE": ("logging", "file"),
            
            # Security
            "ENABLE_RATE_LIMITING": ("security", "enable_rate_limiting", bool),
            "RATE_LIMIT_REQUESTS": ("security", "rate_limit_requests", int),
            "ENABLE_API_KEY": ("security", "enable_api_key", bool),
            "API_KEY": ("security", "api_key"),
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Parse value type if converter specified
                    if len(config_path) > 2 and callable(config_path[-1]):
                        converter = config_path[-1]
                        config_path = config_path[:-1]
                        
                        if converter == bool:
                            value = value.lower() in ("true", "1", "yes", "on")
                        else:
                            value = converter(value)
                    
                    # Set nested value
                    self._set_nested_value(self.config, config_path, value)
                    logger.debug(f"Set config from env {env_var}: {config_path} = {value}")
                    
                except Exception as e:
                    logger.warning(f"Failed to parse env var {env_var}={value}: {e}")
    
    def _deep_merge(self, target: Dict, source: Dict):
        """Recursively merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge(target[key], value)
            else:
                target[key] = value
    
    def _set_nested_value(self, d: Dict, keys: tuple, value: Any):
        """Set value in nested dictionary."""
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    
    def get(self, *keys, default=None):
        """Get nested configuration value."""
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    def set(self, *keys, value):
        """Set nested configuration value."""
        if len(keys) == 0:
            return
        
        target = self.config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        target[keys[-1]] = value
    
    def update(self, new_config: Dict[str, Any]):
        """Update configuration with new values."""
        self._deep_merge(self.config, new_config)
    
    def save(self, file_path: Optional[str] = None):
        """Save configuration to YAML file."""
        file_path = file_path or self.config_file
        try:
            with open(file_path, 'w') as f:
                yaml.safe_dump(self.config, f, indent=2, default_flow_style=False)
            logger.info(f"Configuration saved to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate server config
        if not isinstance(self.get("server", "port"), int) or not (1 <= self.get("server", "port") <= 65535):
            issues.append("Invalid server port")
        
        # Validate model config
        if not self.get("model", "repo_id"):
            issues.append("Model repo_id is required")
        
        # Validate generation limits
        max_text = self.get("generation", "limits", "max_text_length")
        if not isinstance(max_text, int) or max_text <= 0:
            issues.append("Invalid max_text_length")
        
        # Validate audio processing
        sample_rate = self.get("audio_processing", "sample_rate")
        if not isinstance(sample_rate, int) or sample_rate <= 0:
            issues.append("Invalid sample_rate")
        
        return issues

# Global configuration instance
config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration manager instance."""
    global config_manager
    if config_manager is None:
        config_manager = ConfigManager()
    return config_manager

def reload_config():
    """Reload configuration from file and environment."""
    global config_manager
    config_manager = ConfigManager()
    return config_manager

# Convenience functions
def get_server_config() -> Dict[str, Any]:
    """Get server configuration."""
    return get_config().get("server")

def get_model_config() -> Dict[str, Any]:
    """Get model configuration."""
    return get_config().get("model")

def get_generation_defaults() -> Dict[str, Any]:
    """Get generation default parameters."""
    return get_config().get("generation", "defaults")

def get_audio_processing_config() -> Dict[str, Any]:
    """Get audio processing configuration."""
    return get_config().get("audio_processing")

def get_performance_config() -> Dict[str, Any]:
    """Get performance configuration."""
    return get_config().get("performance")

# Configuration validation decorator
def validate_config(func):
    """Decorator to validate configuration before function execution."""
    def wrapper(*args, **kwargs):
        issues = get_config().validate()
        if issues:
            raise ValueError(f"Configuration validation failed: {', '.join(issues)}")
        return func(*args, **kwargs)
    return wrapper