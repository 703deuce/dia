#!/usr/bin/env python3
"""
Demonstration of the Voice Library System for Dia

This script shows how to:
1. Create a voice library
2. Add voices from audio files
3. Use cached voices for fast generation
4. Compare performance with/without caching
"""

import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dia.model import Dia
from voice_library import VoiceLibrary, DiaWithVoiceLibrary


def main():
    print("=== Dia Voice Library Demo ===\n")
    
    # Initialize Dia model
    print("1. Loading Dia model...")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    print("   ✓ Model loaded\n")
    
    # Initialize voice library
    print("2. Initializing voice library...")
    voice_lib = VoiceLibrary("./my_voice_library")
    dia_with_voices = DiaWithVoiceLibrary(model, voice_lib)
    print("   ✓ Voice library ready\n")
    
    # Check if we have existing voices
    voices = voice_lib.list_voices()
    if len(voices) == 0:
        print("3. No voices found in library. Let's add one...")
        
        # You need to have a reference audio file
        # For demo, we'll create one using the simple example first
        demo_audio_path = "demo_voice.mp3"
        demo_text = "[S1] Hello, this is a demo voice for the voice library system. [S2] This voice will be cached for future use."
        
        if not os.path.exists(demo_audio_path):
            print("   Creating demo audio file...")
            demo_audio = model.generate(demo_text, verbose=True)
            model.save_audio(demo_audio_path, demo_audio)
            print(f"   ✓ Demo audio saved to {demo_audio_path}")
        
        # Add voice to library
        print("   Adding voice to library...")
        start_time = time.time()
        
        voice_profile = dia_with_voices.add_voice_to_library(
            voice_id="demo_speaker",
            audio_path=demo_audio_path,
            reference_text=demo_text,
            metadata={
                "speaker": "Demo Speaker",
                "language": "English",
                "created_by": "voice_library_demo.py"
            }
        )
        
        add_time = time.time() - start_time
        print(f"   ✓ Voice added in {add_time:.2f} seconds")
        print(f"   ✓ Profile: {voice_profile}\n")
    else:
        print(f"3. Found {len(voices)} existing voices in library:")
        for voice in voices:
            print(f"   - {voice['voice_id']}: '{voice['reference_text'][:50]}...'")
        print()
    
    # Demonstrate fast generation with cached voice
    print("4. Testing voice generation performance...")
    
    test_texts = [
        "[S1] This is the first test sentence using our cached voice.",
        "[S2] Here's another test with a different speaker tag.",
        "[S1] And this is a third test to show consistency across generations."
    ]
    
    voice_id = "demo_speaker" if "demo_speaker" in [v['voice_id'] for v in voice_lib.list_voices()] else voices[0]['voice_id']
    
    print(f"\nUsing voice: {voice_id}")
    
    for i, test_text in enumerate(test_texts, 1):
        print(f"\n--- Generation {i} ---")
        print(f"Text: {test_text}")
        
        start_time = time.time()
        
        # Generate using cached voice (fast!)
        output_audio = dia_with_voices.generate_with_voice(
            voice_id=voice_id,
            new_text=test_text,
            verbose=True,
            cfg_scale=4.0
        )
        
        generation_time = time.time() - start_time
        print(f"Generation time: {generation_time:.2f} seconds")
        
        # Save output
        output_path = f"voice_lib_output_{i}.mp3"
        model.save_audio(output_path, output_audio)
        print(f"Saved: {output_path}")
    
    # Show library statistics
    print("\n5. Voice Library Statistics:")
    print(f"   Total voices: {len(voice_lib.list_voices())}")
    print(f"   Library path: {voice_lib.library_path}")
    
    for voice in voice_lib.list_voices():
        profile = voice_lib.load_voice(voice['voice_id'])
        print(f"   - {voice['voice_id']}:")
        print(f"     * Tokens: {profile.duration_tokens}")
        print(f"     * Channels: {profile.num_channels}")
        print(f"     * Reference: '{profile.reference_text[:40]}...'")
    
    print("\n=== Demo Complete ===")
    print("\nBenefits of using Voice Library:")
    print("• No re-encoding of reference audio (saves 1-3 seconds per generation)")
    print("• Consistent voice across multiple generations")
    print("• Easy voice management and organization")
    print("• Reusable voice profiles across sessions")


def performance_comparison():
    """Compare performance between direct audio loading vs cached voices."""
    print("\n=== Performance Comparison ===")
    
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    voice_lib = VoiceLibrary("./my_voice_library")
    dia_with_voices = DiaWithVoiceLibrary(model, voice_lib)
    
    test_text = "[S1] This is a performance test."
    
    if len(voice_lib.list_voices()) > 0:
        voice_id = voice_lib.list_voices()[0]['voice_id']
        profile = voice_lib.load_voice(voice_id)
        
        # Test 1: Using cached voice (fast)
        print("Test 1: Using cached voice...")
        start_time = time.time()
        output1 = dia_with_voices.generate_with_voice(voice_id, test_text)
        cached_time = time.time() - start_time
        print(f"Cached voice time: {cached_time:.2f} seconds")
        
        # Test 2: Using direct audio file (slow)
        print("\nTest 2: Using direct audio file...")
        start_time = time.time()
        output2 = model.generate(
            profile.reference_text + test_text,
            audio_prompt=profile.original_audio_path
        )
        direct_time = time.time() - start_time
        print(f"Direct audio time: {direct_time:.2f} seconds")
        
        speedup = direct_time / cached_time
        print(f"\nSpeedup: {speedup:.1f}x faster with cached voice!")
        print(f"Time saved: {direct_time - cached_time:.2f} seconds per generation")


if __name__ == "__main__":
    main()
    
    # Uncomment to run performance comparison
    # performance_comparison()