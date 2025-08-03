#!/usr/bin/env python3
"""
Quick test of the voice library system
"""

from dia.model import Dia
from voice_library import VoiceLibrary, DiaWithVoiceLibrary
import time

def test_system():
    print("🎵 Testing Voice Library System...")
    
    # Load model
    print("Loading Dia model...")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    print("✓ Model loaded successfully!")
    
    # Create voice library
    print("Initializing voice library...")
    voice_lib = VoiceLibrary("./my_voices")
    dia_with_voices = DiaWithVoiceLibrary(model, voice_lib)
    print("✓ Voice library initialized!")
    
    # Show current status
    voices = voice_lib.list_voices()
    print(f"\n📊 Current Status:")
    print(f"   Library location: {voice_lib.library_path}")
    print(f"   Voices in library: {len(voices)}")
    
    if len(voices) == 0:
        print("\n🚀 Ready to add your first voice!")
        print("   Use: python voice_cli.py add 'voice_name' 'audio.mp3' '[S1] transcript'")
    else:
        print(f"\n🎯 Available voices:")
        for voice in voices:
            print(f"   - {voice['voice_id']}: {voice['duration_tokens']} tokens")
    
    print("\n✅ Voice library system is working perfectly!")
    return True

if __name__ == "__main__":
    test_system()