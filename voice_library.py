"""
Voice Library System for Dia Model

This module provides functionality to pre-process and cache reference audio voices,
eliminating the need to re-encode the same voice for multiple generations.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np


class VoiceProfile:
    """
    Represents a cached voice profile with all necessary data for voice cloning.
    """
    
    def __init__(
        self,
        voice_id: str,
        reference_text: str,
        dac_tokens: torch.Tensor,
        original_audio_path: str,
        metadata: Optional[Dict] = None
    ):
        """
        Initialize a voice profile.
        
        Args:
            voice_id: Unique identifier for this voice
            reference_text: The transcript of the reference audio
            dac_tokens: DAC-encoded audio tokens [T, C]
            original_audio_path: Path to the original audio file
            metadata: Optional metadata (duration, speaker info, etc.)
        """
        self.voice_id = voice_id
        self.reference_text = reference_text
        self.dac_tokens = dac_tokens  # Shape: [T, 9]
        self.original_audio_path = original_audio_path
        self.metadata = metadata or {}
        
        # Calculate derived properties
        self.duration_tokens = dac_tokens.shape[0]
        self.num_channels = dac_tokens.shape[1]
        
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization (without tensor data)."""
        return {
            "voice_id": self.voice_id,
            "reference_text": self.reference_text,
            "original_audio_path": self.original_audio_path,
            "duration_tokens": self.duration_tokens,
            "num_channels": self.num_channels,
            "metadata": self.metadata
        }
    
    def __repr__(self):
        return f"VoiceProfile(id='{self.voice_id}', tokens={self.duration_tokens}, text='{self.reference_text[:50]}...')"


class VoiceLibrary:
    """
    Manages a library of cached voice profiles for efficient voice cloning.
    """
    
    def __init__(self, library_path: str = "voice_library"):
        """
        Initialize the voice library.
        
        Args:
            library_path: Directory to store the voice library files
        """
        self.library_path = Path(library_path)
        self.library_path.mkdir(exist_ok=True)
        
        # Subdirectories for organization
        self.profiles_dir = self.library_path / "profiles"
        self.tokens_dir = self.library_path / "tokens"
        self.profiles_dir.mkdir(exist_ok=True)
        self.tokens_dir.mkdir(exist_ok=True)
        
        # Cache for loaded profiles
        self._loaded_profiles: Dict[str, VoiceProfile] = {}
        
        # Load existing profiles index
        self.index_file = self.library_path / "index.json"
        self._load_index()
    
    def _load_index(self):
        """Load the voice profiles index."""
        if self.index_file.exists():
            with open(self.index_file, 'r') as f:
                self.index = json.load(f)
        else:
            self.index = {}
    
    def _save_index(self):
        """Save the voice profiles index."""
        with open(self.index_file, 'w') as f:
            json.dump(self.index, f, indent=2)
    
    def add_voice_from_audio(
        self, 
        dia_model, 
        voice_id: str, 
        audio_path: str, 
        reference_text: str,
        metadata: Optional[Dict] = None
    ) -> VoiceProfile:
        """
        Process and add a new voice to the library.
        
        Args:
            dia_model: Initialized Dia model instance
            voice_id: Unique identifier for this voice
            audio_path: Path to the reference audio file
            reference_text: Transcript of the reference audio
            metadata: Optional metadata
            
        Returns:
            VoiceProfile: The created voice profile
        """
        if voice_id in self.index:
            raise ValueError(f"Voice ID '{voice_id}' already exists in library")
        
        print(f"Processing voice '{voice_id}'...")
        
        # Encode the reference audio using Dia model
        print("  - Encoding audio with DAC...")
        dac_tokens = dia_model.load_audio(audio_path)  # Returns [T, C]
        
        print(f"  - Processed {dac_tokens.shape[0]} tokens across {dac_tokens.shape[1]} channels")
        
        # Create voice profile
        profile = VoiceProfile(
            voice_id=voice_id,
            reference_text=reference_text,
            dac_tokens=dac_tokens,
            original_audio_path=audio_path,
            metadata=metadata
        )
        
        # Save the profile
        self._save_voice_profile(profile)
        
        # Update index
        self.index[voice_id] = profile.to_dict()
        self._save_index()
        
        # Cache in memory
        self._loaded_profiles[voice_id] = profile
        
        print(f"  - Voice '{voice_id}' added to library successfully!")
        return profile
    
    def _save_voice_profile(self, profile: VoiceProfile):
        """Save voice profile data to disk."""
        # Save profile metadata as JSON
        profile_file = self.profiles_dir / f"{profile.voice_id}.json"
        with open(profile_file, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)
        
        # Save DAC tokens as pickle (preserves tensor format)
        tokens_file = self.tokens_dir / f"{profile.voice_id}.pt"
        torch.save(profile.dac_tokens, tokens_file)
    
    def load_voice(self, voice_id: str) -> VoiceProfile:
        """
        Load a voice profile from the library.
        
        Args:
            voice_id: ID of the voice to load
            
        Returns:
            VoiceProfile: The loaded voice profile
        """
        if voice_id in self._loaded_profiles:
            return self._loaded_profiles[voice_id]
        
        if voice_id not in self.index:
            raise ValueError(f"Voice ID '{voice_id}' not found in library")
        
        # Load profile metadata
        profile_file = self.profiles_dir / f"{voice_id}.json"
        with open(profile_file, 'r') as f:
            profile_data = json.load(f)
        
        # Load DAC tokens
        tokens_file = self.tokens_dir / f"{voice_id}.pt"
        dac_tokens = torch.load(tokens_file)
        
        # Create profile object
        profile = VoiceProfile(
            voice_id=profile_data["voice_id"],
            reference_text=profile_data["reference_text"],
            dac_tokens=dac_tokens,
            original_audio_path=profile_data["original_audio_path"],
            metadata=profile_data["metadata"]
        )
        
        # Cache in memory
        self._loaded_profiles[voice_id] = profile
        return profile
    
    def get_cached_audio_prompt(self, voice_id: str) -> Tuple[torch.Tensor, str]:
        """
        Get the cached DAC tokens and reference text for a voice.
        
        Args:
            voice_id: ID of the voice
            
        Returns:
            Tuple of (dac_tokens, reference_text)
        """
        profile = self.load_voice(voice_id)
        return profile.dac_tokens, profile.reference_text
    
    def list_voices(self) -> List[Dict]:
        """List all voices in the library."""
        return list(self.index.values())
    
    def remove_voice(self, voice_id: str):
        """Remove a voice from the library."""
        if voice_id not in self.index:
            raise ValueError(f"Voice ID '{voice_id}' not found in library")
        
        # Remove files
        profile_file = self.profiles_dir / f"{voice_id}.json"
        tokens_file = self.tokens_dir / f"{voice_id}.pt"
        
        if profile_file.exists():
            profile_file.unlink()
        if tokens_file.exists():
            tokens_file.unlink()
        
        # Remove from index and cache
        del self.index[voice_id]
        if voice_id in self._loaded_profiles:
            del self._loaded_profiles[voice_id]
        
        self._save_index()
        print(f"Voice '{voice_id}' removed from library")
    
    def get_voice_info(self, voice_id: str) -> Dict:
        """Get information about a voice."""
        if voice_id not in self.index:
            raise ValueError(f"Voice ID '{voice_id}' not found in library")
        return self.index[voice_id]


class DiaWithVoiceLibrary:
    """
    Enhanced Dia model wrapper that integrates with the voice library system.
    """
    
    def __init__(self, dia_model, voice_library: VoiceLibrary):
        self.dia_model = dia_model
        self.voice_library = voice_library
    
    def generate_with_voice(
        self,
        voice_id: str,
        new_text: str,
        max_tokens: int = 3072,
        cfg_scale: float = 3.0,
        temperature: float = 1.8,
        top_p: float = 0.90,
        cfg_filter_top_k: int = 50,
        use_torch_compile: bool = False,
        verbose: bool = False
    ) -> np.ndarray:
        """
        Generate audio using a cached voice from the library.
        
        Args:
            voice_id: ID of the voice to use
            new_text: Text to generate in that voice
            Other args: Same as Dia.generate()
            
        Returns:
            Generated audio as numpy array
        """
        # Load cached voice data
        cached_tokens, reference_text = self.voice_library.get_cached_audio_prompt(voice_id)
        
        if verbose:
            print(f"Using cached voice '{voice_id}' with {cached_tokens.shape[0]} reference tokens")
        
        # Combine reference text with new text
        combined_text = reference_text + new_text
        
        # Generate using the cached audio prompt
        return self.dia_model.generate(
            text=combined_text,
            audio_prompt=cached_tokens,  # Use cached DAC tokens directly
            max_tokens=max_tokens,
            cfg_scale=cfg_scale,
            temperature=temperature,
            top_p=top_p,
            cfg_filter_top_k=cfg_filter_top_k,
            use_torch_compile=use_torch_compile,
            verbose=verbose
        )
    
    def add_voice_to_library(
        self,
        voice_id: str,
        audio_path: str,
        reference_text: str,
        metadata: Optional[Dict] = None
    ) -> VoiceProfile:
        """Add a new voice to the library."""
        return self.voice_library.add_voice_from_audio(
            self.dia_model, voice_id, audio_path, reference_text, metadata
        )