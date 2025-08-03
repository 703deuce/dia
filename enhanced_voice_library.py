"""
Enhanced Voice Library System for Dia Model with Serverless Optimizations

This module provides advanced functionality including:
- Intelligent text chunking for long-form TTS
- VRAM optimizations and memory management
- Seed control for reproducible output
- Audio post-processing
- Performance monitoring
"""

import json
import os
import pickle
import re
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextChunker:
    """Intelligent text chunking system that preserves linguistic structure."""
    
    @staticmethod
    def split_into_sentences(text: str) -> List[str]:
        """Split text into sentences while handling edge cases."""
        if not text or text.isspace():
            return []
            
        # Handle abbreviations that shouldn't trigger splits
        abbreviations = {
            'dr.', 'mr.', 'mrs.', 'ms.', 'prof.', 'sr.', 'jr.',
            'inc.', 'ltd.', 'corp.', 'vs.', 'etc.', 'e.g.', 'i.e.',
            'p.m.', 'a.m.', 'u.s.', 'u.k.', 'u.n.'
        }
        
        # Replace abbreviations temporarily
        temp_text = text.lower()
        replacements = {}
        for i, abbr in enumerate(abbreviations):
            placeholder = f"__ABBR_{i}__"
            if abbr in temp_text:
                text = text.replace(abbr, placeholder)
                text = text.replace(abbr.upper(), placeholder)
                text = text.replace(abbr.capitalize(), placeholder)
                replacements[placeholder] = abbr
        
        # Split on sentence endings
        sentences = re.split(r'[.!?]+\s+', text)
        
        # Restore abbreviations
        for placeholder, abbr in replacements.items():
            sentences = [s.replace(placeholder, abbr) for s in sentences]
        
        # Clean up and filter
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    @staticmethod
    def preprocess_and_tag_sentences(full_text: str) -> List[Tuple[str, str]]:
        """Parse text and tag each sentence with its speaker."""
        if not full_text or full_text.isspace():
            return []
            
        # Find all speaker tags and their positions
        speaker_pattern = r'\[S[12]\]'
        matches = list(re.finditer(speaker_pattern, full_text))
        
        if not matches:
            # No speaker tags, default to S1
            sentences = TextChunker.split_into_sentences(full_text)
            return [('[S1]', sentence) for sentence in sentences]
        
        tagged_sentences = []
        current_speaker = '[S1]'  # Default speaker
        
        for i, match in enumerate(matches):
            # Get text before this tag (if any)
            start_pos = matches[i-1].end() if i > 0 else 0
            text_before = full_text[start_pos:match.start()].strip()
            
            if text_before:
                sentences = TextChunker.split_into_sentences(text_before)
                for sentence in sentences:
                    tagged_sentences.append((current_speaker, sentence))
            
            # Update current speaker
            current_speaker = match.group()
            
            # Get text after this tag
            end_pos = matches[i+1].start() if i+1 < len(matches) else len(full_text)
            text_after = full_text[match.end():end_pos].strip()
            
            if text_after:
                sentences = TextChunker.split_into_sentences(text_after)
                for sentence in sentences:
                    tagged_sentences.append((current_speaker, sentence))
        
        return tagged_sentences
    
    @staticmethod
    def chunk_text_by_sentences(
        full_text: str,
        chunk_size: int = 120,
        allow_multiple_speakers: bool = False
    ) -> List[str]:
        """Chunk text intelligently based on sentences and speaker boundaries."""
        if not full_text or full_text.isspace():
            return []
            
        if chunk_size <= 0:
            chunk_size = float('inf')
            
        tagged_sentences = TextChunker.preprocess_and_tag_sentences(full_text)
        if not tagged_sentences:
            return []
        
        chunks = []
        current_chunk_sentences = []
        current_chunk_tag = None
        current_chunk_len = 0
        
        for sentence_tag, sentence_text in tagged_sentences:
            sentence_len = len(sentence_text)
            start_new_chunk = False
            
            # Start new chunk if no current tag
            if current_chunk_tag is None:
                start_new_chunk = True
            # Force new chunk on speaker change (unless allowing multiple speakers)
            elif not allow_multiple_speakers and sentence_tag != current_chunk_tag:
                logger.debug(f"New chunk due to speaker change: {current_chunk_tag} -> {sentence_tag}")
                start_new_chunk = True
            # Check length limit
            elif current_chunk_len + sentence_len + 1 > chunk_size:
                logger.debug(f"New chunk due to length: {current_chunk_len} + {sentence_len} > {chunk_size}")
                start_new_chunk = True
            
            if start_new_chunk:
                # Finalize current chunk
                if current_chunk_sentences:
                    chunk_text = " ".join(current_chunk_sentences)
                    formatted_chunk = f"{current_chunk_tag} {chunk_text}"
                    chunks.append(formatted_chunk)
                
                # Start new chunk
                current_chunk_sentences = [sentence_text]
                current_chunk_tag = sentence_tag
                current_chunk_len = sentence_len
            else:
                # Add to current chunk
                current_chunk_sentences.append(sentence_text)
                current_chunk_len += sentence_len + 1  # +1 for space
        
        # Finalize last chunk
        if current_chunk_sentences:
            chunk_text = " ".join(current_chunk_sentences)
            formatted_chunk = f"{current_chunk_tag} {chunk_text}"
            chunks.append(formatted_chunk)
        
        logger.info(f"Chunked text into {len(chunks)} segments")
        return chunks


class PerformanceMonitor:
    """Monitor performance and memory usage."""
    
    def __init__(self):
        self.timestamps = {}
        self.gpu_memory_before = None
        
    def start(self, operation: str):
        """Start timing an operation."""
        self.timestamps[operation] = time.time()
        if torch.cuda.is_available():
            self.gpu_memory_before = torch.cuda.memory_allocated()
    
    def end(self, operation: str) -> float:
        """End timing and return duration."""
        if operation in self.timestamps:
            duration = time.time() - self.timestamps[operation]
            
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_delta = memory_after - self.gpu_memory_before if self.gpu_memory_before else 0
                logger.debug(f"{operation}: {duration:.2f}s, Memory change: {memory_delta/1024/1024:.1f}MB")
            else:
                logger.debug(f"{operation}: {duration:.2f}s")
                
            return duration
        return 0.0


class EnhancedVoiceProfile:
    """Enhanced voice profile with metadata and performance info."""
    
    def __init__(
        self,
        voice_id: str,
        reference_text: str,
        dac_tokens: torch.Tensor,
        original_audio_path: str,
        metadata: Optional[Dict] = None,
        processing_time: float = 0.0
    ):
        self.voice_id = voice_id
        self.reference_text = reference_text
        self.dac_tokens = dac_tokens
        self.original_audio_path = original_audio_path
        self.metadata = metadata or {}
        self.processing_time = processing_time
        
        # Performance metrics
        self.duration_tokens = dac_tokens.shape[0]
        self.num_channels = dac_tokens.shape[1]
        self.created_at = time.time()
        self.usage_count = 0
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "voice_id": self.voice_id,
            "reference_text": self.reference_text,
            "original_audio_path": self.original_audio_path,
            "duration_tokens": self.duration_tokens,
            "num_channels": self.num_channels,
            "processing_time": self.processing_time,
            "created_at": self.created_at,
            "usage_count": self.usage_count,
            "metadata": self.metadata
        }
    
    def increment_usage(self):
        """Track usage for analytics."""
        self.usage_count += 1
    
    def __repr__(self):
        return f"EnhancedVoiceProfile(id='{self.voice_id}', tokens={self.duration_tokens}, usage={self.usage_count})"


class OptimizedDiaModel:
    """Wrapper for Dia model with memory optimizations."""
    
    def __init__(self, dia_model):
        self.dia_model = dia_model
        self.device = dia_model.device
        
    def generate_optimized(
        self,
        text: str,
        audio_prompt: Optional[torch.Tensor] = None,
        seed: int = 42,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        cfg_filter_top_k: int = 35,
        verbose: bool = False
    ) -> np.ndarray:
        """Generate with memory optimization."""
        
        # Clear cache before generation
        self.clear_cache()
        
        # Set seed for reproducibility
        if seed >= 0:
            torch.manual_seed(seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(seed)
        
        # Generate
        try:
            result = self.dia_model.generate(
                text=text,
                audio_prompt=audio_prompt,
                seed=seed,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                cfg_filter_top_k=cfg_filter_top_k,
                verbose=verbose
            )
            
            # Clear cache after generation
            self.clear_cache()
            
            return result
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.clear_cache()
            raise
    
    def clear_cache(self):
        """Aggressive memory cleanup."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def reset_state(self):
        """Reset model state if available."""
        if hasattr(self.dia_model, 'reset_state'):
            self.dia_model.reset_state()


class EnhancedVoiceLibrary:
    """Enhanced voice library with chunking and optimization features."""
    
    def __init__(self, library_path: str = "voice_library"):
        self.library_path = Path(library_path)
        self.library_path.mkdir(exist_ok=True)
        
        # Subdirectories
        self.profiles_dir = self.library_path / "profiles"
        self.tokens_dir = self.library_path / "tokens"
        self.profiles_dir.mkdir(exist_ok=True)
        self.tokens_dir.mkdir(exist_ok=True)
        
        # In-memory cache with LRU-like behavior
        self._loaded_profiles: Dict[str, EnhancedVoiceProfile] = {}
        self._max_cache_size = 10  # Limit cache size for serverless
        
        # Load index
        self.index_file = self.library_path / "index.json"
        self._load_index()
        
        # Performance tracking
        self.performance_stats = {
            "total_generations": 0,
            "cache_hits": 0,
            "cache_misses": 0
        }
    
    def _load_index(self):
        """Load voice profiles index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Save voice profiles index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def _manage_cache_size(self):
        """Manage cache size for memory efficiency."""
        if len(self._loaded_profiles) > self._max_cache_size:
            # Remove least recently used (simple approach)
            oldest_key = min(
                self._loaded_profiles.keys(),
                key=lambda k: self._loaded_profiles[k].created_at
            )
            del self._loaded_profiles[oldest_key]
            logger.debug(f"Removed {oldest_key} from cache due to size limit")
    
    def add_voice_from_audio(
        self,
        dia_model,
        voice_id: str,
        audio_path: str,
        reference_text: str,
        metadata: Optional[Dict] = None
    ) -> EnhancedVoiceProfile:
        """Add voice with performance tracking."""
        if voice_id in self.index:
            raise ValueError(f"Voice ID '{voice_id}' already exists")
        
        monitor = PerformanceMonitor()
        monitor.start("voice_processing")
        
        logger.info(f"Processing voice '{voice_id}'...")
        
        # Encode audio
        dac_tokens = dia_model.load_audio(audio_path)
        processing_time = monitor.end("voice_processing")
        
        # Create enhanced profile
        profile = EnhancedVoiceProfile(
            voice_id=voice_id,
            reference_text=reference_text,
            dac_tokens=dac_tokens,
            original_audio_path=audio_path,
            metadata=metadata,
            processing_time=processing_time
        )
        
        # Save profile
        self._save_voice_profile(profile)
        self.index[voice_id] = profile.to_dict()
        self._save_index()
        
        # Cache with size management
        self._loaded_profiles[voice_id] = profile
        self._manage_cache_size()
        
        logger.info(f"Voice '{voice_id}' processed in {processing_time:.2f}s")
        return profile
    
    def _save_voice_profile(self, profile: EnhancedVoiceProfile):
        """Save profile with optimized storage."""
        # Save metadata as JSON
        profile_file = self.profiles_dir / f"{profile.voice_id}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        
        # Save tokens with compression
        tokens_file = self.tokens_dir / f"{profile.voice_id}.pt"
        torch.save(profile.dac_tokens, tokens_file)
    
    def load_voice(self, voice_id: str) -> EnhancedVoiceProfile:
        """Load voice with cache management."""
        if voice_id in self._loaded_profiles:
            self.performance_stats["cache_hits"] += 1
            return self._loaded_profiles[voice_id]
        
        self.performance_stats["cache_misses"] += 1
        
        if voice_id not in self.index:
            raise ValueError(f"Voice ID '{voice_id}' not found")
        
        # Load from disk
        profile_file = self.profiles_dir / f"{voice_id}.json"
        tokens_file = self.tokens_dir / f"{voice_id}.pt"
        
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
        
        dac_tokens = torch.load(tokens_file, map_location='cpu')
        
        profile = EnhancedVoiceProfile(
            voice_id=profile_data["voice_id"],
            reference_text=profile_data["reference_text"],
            dac_tokens=dac_tokens,
            original_audio_path=profile_data["original_audio_path"],
            metadata=profile_data.get("metadata", {}),
            processing_time=profile_data.get("processing_time", 0.0)
        )
        
        # Update from saved data
        profile.usage_count = profile_data.get("usage_count", 0)
        profile.created_at = profile_data.get("created_at", time.time())
        
        # Cache with size management
        self._loaded_profiles[voice_id] = profile
        self._manage_cache_size()
        
        return profile
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        cache_hit_rate = 0.0
        total_requests = self.performance_stats["cache_hits"] + self.performance_stats["cache_misses"]
        if total_requests > 0:
            cache_hit_rate = self.performance_stats["cache_hits"] / total_requests
        
        return {
            **self.performance_stats,
            "cache_hit_rate": cache_hit_rate,
            "cached_voices": len(self._loaded_profiles),
            "total_voices": len(self.index)
        }


class EnhancedDiaWithVoiceLibrary:
    """Enhanced Dia wrapper with chunking, optimization, and advanced features."""
    
    def __init__(self, dia_model, voice_library: EnhancedVoiceLibrary):
        self.optimized_model = OptimizedDiaModel(dia_model)
        self.voice_library = voice_library
        self.chunker = TextChunker()
        
        # Default generation parameters
        self.default_params = {
            "cfg_scale": 3.0,
            "temperature": 1.3,
            "top_p": 0.95,
            "cfg_filter_top_k": 35,
            "seed": 42,
            "chunk_size": 120,
            "enable_chunking": True
        }
    
    def generate_with_voice(
        self,
        voice_id: str,
        text: str,
        **generation_params
    ) -> Tuple[np.ndarray, Dict]:
        """Generate with enhanced features and monitoring."""
        
        # Merge parameters
        params = {**self.default_params, **generation_params}
        
        monitor = PerformanceMonitor()
        monitor.start("total_generation")
        
        # Load voice profile
        profile = self.voice_library.load_voice(voice_id)
        profile.increment_usage()
        
        # Determine if chunking is needed
        should_chunk = (
            params["enable_chunking"] and 
            len(text) > params["chunk_size"] * 2
        )
        
        if should_chunk:
            return self._generate_chunked(profile, text, params, monitor)
        else:
            return self._generate_single(profile, text, params, monitor)
    
    def _generate_single(
        self, 
        profile: EnhancedVoiceProfile, 
        text: str, 
        params: Dict,
        monitor: PerformanceMonitor
    ) -> Tuple[np.ndarray, Dict]:
        """Generate single chunk."""
        
        # Prepare input
        combined_text = profile.reference_text + " " + text
        
        # Generate
        monitor.start("model_generation")
        audio_result = self.optimized_model.generate_optimized(
            text=combined_text,
            audio_prompt=profile.dac_tokens,
            seed=params["seed"],
            cfg_scale=params["cfg_scale"],
            temperature=params["temperature"],
            top_p=params["top_p"],
            cfg_filter_top_k=params["cfg_filter_top_k"],
            verbose=False
        )
        generation_time = monitor.end("model_generation")
        
        total_time = monitor.end("total_generation")
        
        # Performance info
        performance_info = {
            "chunks_processed": 1,
            "total_generation_time": total_time,
            "model_generation_time": generation_time,
            "chunking_enabled": False,
            "voice_id": profile.voice_id,
            "seed_used": params["seed"]
        }
        
        return audio_result, performance_info
    
    def _generate_chunked(
        self, 
        profile: EnhancedVoiceProfile, 
        text: str, 
        params: Dict,
        monitor: PerformanceMonitor
    ) -> Tuple[np.ndarray, Dict]:
        """Generate with intelligent chunking."""
        
        # Chunk the text
        monitor.start("text_chunking")
        chunks = self.chunker.chunk_text_by_sentences(
            text, 
            chunk_size=params["chunk_size"]
        )
        chunking_time = monitor.end("text_chunking")
        
        if not chunks:
            raise ValueError("Text chunking produced no valid chunks")
        
        logger.info(f"Processing {len(chunks)} chunks for voice '{profile.voice_id}'")
        
        # Generate each chunk
        audio_segments = []
        chunk_times = []
        
        # Progress bar for multiple chunks
        with tqdm(total=len(chunks), desc="Generating chunks", disable=len(chunks)==1) as pbar:
            for i, chunk in enumerate(chunks):
                monitor.start(f"chunk_{i}")
                
                # For first chunk, include reference text
                if i == 0:
                    input_text = profile.reference_text + " " + chunk
                else:
                    input_text = chunk
                
                try:
                    chunk_audio = self.optimized_model.generate_optimized(
                        text=input_text,
                        audio_prompt=profile.dac_tokens,
                        seed=params["seed"],
                        cfg_scale=params["cfg_scale"],
                        temperature=params["temperature"],
                        top_p=params["top_p"],
                        cfg_filter_top_k=params["cfg_filter_top_k"],
                        verbose=False
                    )
                    
                    if chunk_audio is not None and chunk_audio.size > 0:
                        audio_segments.append(chunk_audio)
                        chunk_time = monitor.end(f"chunk_{i}")
                        chunk_times.append(chunk_time)
                        
                        logger.debug(f"Chunk {i+1}/{len(chunks)} completed in {chunk_time:.2f}s")
                    else:
                        logger.warning(f"Chunk {i+1} produced no audio")
                        
                except Exception as e:
                    logger.error(f"Error generating chunk {i+1}: {e}")
                    continue
                
                pbar.update(1)
        
        if not audio_segments:
            raise RuntimeError("No audio segments were successfully generated")
        
        # Concatenate segments
        monitor.start("audio_concatenation")
        final_audio = np.concatenate(audio_segments, axis=0)
        concatenation_time = monitor.end("audio_concatenation")
        
        total_time = monitor.end("total_generation")
        
        # Performance info
        performance_info = {
            "chunks_processed": len(chunks),
            "total_generation_time": total_time,
            "chunking_time": chunking_time,
            "concatenation_time": concatenation_time,
            "average_chunk_time": np.mean(chunk_times) if chunk_times else 0,
            "chunking_enabled": True,
            "voice_id": profile.voice_id,
            "seed_used": params["seed"],
            "audio_length_seconds": len(final_audio) / 44100  # Assuming 44.1kHz
        }
        
        return final_audio, performance_info
    
    def list_voices(self) -> List[Dict]:
        """List all voices with usage statistics."""
        voices = []
        for voice_id, voice_data in self.voice_library.index.items():
            voice_info = voice_data.copy()
            voice_info["cached"] = voice_id in self.voice_library._loaded_profiles
            voices.append(voice_info)
        return voices
    
    def get_system_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        return {
            "voice_library": self.voice_library.get_performance_stats(),
            "memory_info": {
                "cuda_available": torch.cuda.is_available(),
                "cuda_memory_allocated": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                "cuda_memory_reserved": torch.cuda.memory_reserved() if torch.cuda.is_available() else 0,
            }
        }