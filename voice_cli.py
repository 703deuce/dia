#!/usr/bin/env python3
"""
Voice Library CLI Tool for Dia

A command-line interface for managing voice libraries, adding voices,
and generating audio with cached voices.
"""

import argparse
import sys
import os
from pathlib import Path
from typing import Optional

from dia.model import Dia
from voice_library import VoiceLibrary, DiaWithVoiceLibrary


class VoiceCLI:
    def __init__(self, library_path: str = "voice_library"):
        self.library_path = library_path
        self.voice_lib = VoiceLibrary(library_path)
        self._dia_model = None
        self._dia_with_voices = None
    
    @property
    def dia_model(self):
        """Lazy load the Dia model."""
        if self._dia_model is None:
            print("Loading Dia model...")
            self._dia_model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
            self._dia_with_voices = DiaWithVoiceLibrary(self._dia_model, self.voice_lib)
            print("✓ Model loaded")
        return self._dia_model
    
    @property
    def dia_with_voices(self):
        """Get the Dia model with voice library integration."""
        _ = self.dia_model  # Trigger lazy loading
        return self._dia_with_voices
    
    def list_voices(self):
        """List all voices in the library."""
        voices = self.voice_lib.list_voices()
        
        if not voices:
            print("No voices found in library.")
            print(f"Library path: {self.voice_lib.library_path}")
            return
        
        print(f"Voice Library ({len(voices)} voices):")
        print(f"Path: {self.voice_lib.library_path}")
        print("-" * 60)
        
        for voice in voices:
            print(f"ID: {voice['voice_id']}")
            print(f"  Reference: '{voice['reference_text'][:50]}...'")
            print(f"  Tokens: {voice['duration_tokens']}")
            print(f"  Audio: {voice['original_audio_path']}")
            if voice['metadata']:
                print(f"  Metadata: {voice['metadata']}")
            print()
    
    def add_voice(self, voice_id: str, audio_path: str, reference_text: str, 
                  speaker: Optional[str] = None):
        """Add a new voice to the library."""
        if not os.path.exists(audio_path):
            print(f"Error: Audio file not found: {audio_path}")
            return False
        
        if not reference_text.strip():
            print("Error: Reference text cannot be empty")
            return False
        
        try:
            metadata = {}
            if speaker:
                metadata['speaker'] = speaker
            
            profile = self.dia_with_voices.add_voice_to_library(
                voice_id=voice_id,
                audio_path=audio_path,
                reference_text=reference_text,
                metadata=metadata
            )
            
            print(f"✓ Voice '{voice_id}' added successfully!")
            print(f"  Tokens: {profile.duration_tokens}")
            print(f"  Channels: {profile.num_channels}")
            return True
            
        except ValueError as e:
            print(f"Error: {e}")
            return False
        except Exception as e:
            print(f"Error adding voice: {e}")
            return False
    
    def remove_voice(self, voice_id: str):
        """Remove a voice from the library."""
        try:
            self.voice_lib.remove_voice(voice_id)
            print(f"✓ Voice '{voice_id}' removed successfully!")
            return True
        except ValueError as e:
            print(f"Error: {e}")
            return False
    
    def voice_info(self, voice_id: str):
        """Show detailed information about a voice."""
        try:
            profile = self.voice_lib.load_voice(voice_id)
            print(f"Voice Profile: {voice_id}")
            print("-" * 40)
            print(f"Reference Text: {profile.reference_text}")
            print(f"Duration (tokens): {profile.duration_tokens}")
            print(f"Channels: {profile.num_channels}")
            print(f"Original Audio: {profile.original_audio_path}")
            print(f"Metadata: {profile.metadata}")
            return True
        except ValueError as e:
            print(f"Error: {e}")
            return False
    
    def generate(self, voice_id: str, text: str, output_path: str,
                cfg_scale: float = 4.0, temperature: float = 1.8, 
                top_p: float = 0.9, verbose: bool = False):
        """Generate audio using a cached voice."""
        try:
            if verbose:
                print(f"Generating audio with voice '{voice_id}'...")
                print(f"Text: {text}")
            
            output_audio = self.dia_with_voices.generate_with_voice(
                voice_id=voice_id,
                new_text=text,
                cfg_scale=cfg_scale,
                temperature=temperature,
                top_p=top_p,
                verbose=verbose
            )
            
            self.dia_model.save_audio(output_path, output_audio)
            print(f"✓ Audio saved to: {output_path}")
            return True
            
        except ValueError as e:
            print(f"Error: {e}")
            return False
        except Exception as e:
            print(f"Error generating audio: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="Dia Voice Library CLI")
    parser.add_argument("--library", "-l", default="voice_library", 
                       help="Path to voice library directory")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # List voices
    list_parser = subparsers.add_parser("list", help="List all voices in library")
    
    # Add voice
    add_parser = subparsers.add_parser("add", help="Add a new voice to library")
    add_parser.add_argument("voice_id", help="Unique ID for the voice")
    add_parser.add_argument("audio_path", help="Path to reference audio file")
    add_parser.add_argument("reference_text", help="Transcript of the reference audio")
    add_parser.add_argument("--speaker", help="Speaker name (optional)")
    
    # Remove voice
    remove_parser = subparsers.add_parser("remove", help="Remove voice from library")
    remove_parser.add_argument("voice_id", help="ID of voice to remove")
    
    # Voice info
    info_parser = subparsers.add_parser("info", help="Show voice information")
    info_parser.add_argument("voice_id", help="ID of voice to show info for")
    
    # Generate audio
    gen_parser = subparsers.add_parser("generate", help="Generate audio with cached voice")
    gen_parser.add_argument("voice_id", help="ID of voice to use")
    gen_parser.add_argument("text", help="Text to generate")
    gen_parser.add_argument("output_path", help="Output audio file path")
    gen_parser.add_argument("--cfg-scale", type=float, default=4.0, help="CFG scale")
    gen_parser.add_argument("--temperature", type=float, default=1.8, help="Temperature")
    gen_parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    gen_parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = VoiceCLI(args.library)
    
    if args.command == "list":
        cli.list_voices()
    
    elif args.command == "add":
        cli.add_voice(args.voice_id, args.audio_path, args.reference_text, args.speaker)
    
    elif args.command == "remove":
        cli.remove_voice(args.voice_id)
    
    elif args.command == "info":
        cli.voice_info(args.voice_id)
    
    elif args.command == "generate":
        cli.generate(
            args.voice_id, args.text, args.output_path,
            args.cfg_scale, args.temperature, args.top_p, args.verbose
        )


if __name__ == "__main__":
    main()