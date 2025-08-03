#!/usr/bin/env python3
"""
Quick Voice Library Setup Example

This script demonstrates how to quickly set up and use a voice library
for efficient voice cloning with the Dia model.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dia.model import Dia
from voice_library import VoiceLibrary, DiaWithVoiceLibrary


def create_sample_voices():
    """Create sample voices for demonstration."""
    print("=== Setting Up Voice Library ===\n")
    
    # Load Dia model
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    voice_lib = VoiceLibrary("./sample_voices")
    dia_with_voices = DiaWithVoiceLibrary(model, voice_lib)
    
    # Sample voice configurations
    sample_voices = [
        {
            "voice_id": "narrator",
            "text": "[S1] Welcome to the voice library system. This narrator voice will be used for announcements and introductions.",
            "metadata": {"type": "narrator", "style": "formal"}
        },
        {
            "voice_id": "casual_speaker", 
            "text": "[S2] Hey there! I'm a more casual speaker. You can use my voice for informal conversations and friendly chats.",
            "metadata": {"type": "conversational", "style": "casual"}
        }
    ]
    
    # Create sample audio files and add to library
    for voice_config in sample_voices:
        voice_id = voice_config["voice_id"]
        reference_text = voice_config["text"]
        audio_path = f"{voice_id}_reference.mp3"
        
        print(f"Creating voice: {voice_id}")
        
        # Generate reference audio if it doesn't exist
        if not os.path.exists(audio_path):
            print(f"  - Generating reference audio...")
            reference_audio = model.generate(reference_text, cfg_scale=3.0)
            model.save_audio(audio_path, reference_audio)
            print(f"  - Saved reference audio: {audio_path}")
        
        # Add to voice library
        print(f"  - Adding to voice library...")
        try:
            dia_with_voices.add_voice_to_library(
                voice_id=voice_id,
                audio_path=audio_path,
                reference_text=reference_text,
                metadata=voice_config["metadata"]
            )
            print(f"  ✓ Voice '{voice_id}' added successfully!")
        except ValueError as e:
            print(f"  - Voice already exists: {e}")
        
        print()
    
    return dia_with_voices


def demonstrate_fast_generation():
    """Demonstrate fast voice generation using cached voices."""
    print("=== Fast Voice Generation Demo ===\n")
    
    # Set up voice library
    voice_lib = VoiceLibrary("./sample_voices")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    dia_with_voices = DiaWithVoiceLibrary(model, voice_lib)
    
    # Test texts for different scenarios
    test_scenarios = [
        {
            "voice": "narrator",
            "text": "[S1] Today we'll be exploring the capabilities of artificial intelligence in speech synthesis.",
            "description": "Formal narration"
        },
        {
            "voice": "casual_speaker",
            "text": "[S2] That sounds really cool! I can't wait to try it out myself.",
            "description": "Casual response"
        },
        {
            "voice": "narrator", 
            "text": "[S1] The technology behind this system uses advanced neural networks and audio codecs.",
            "description": "Technical explanation"
        }
    ]
    
    print("Generating audio with cached voices (FAST):\n")
    
    total_start = time.time()
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Scenario {i}: {scenario['description']}")
        print(f"Voice: {scenario['voice']}")
        print(f"Text: {scenario['text']}")
        
        start_time = time.time()
        
        # Generate using cached voice - this is FAST!
        output_audio = dia_with_voices.generate_with_voice(
            voice_id=scenario['voice'],
            new_text=scenario['text'],
            cfg_scale=4.0,
            verbose=False
        )
        
        generation_time = time.time() - start_time
        output_path = f"fast_output_{i}.mp3"
        model.save_audio(output_path, output_audio)
        
        print(f"✓ Generated in {generation_time:.2f}s → {output_path}")
        print()
    
    total_time = time.time() - total_start
    print(f"Total time for 3 generations: {total_time:.2f} seconds")
    print(f"Average per generation: {total_time/3:.2f} seconds")


def show_library_stats():
    """Show voice library statistics."""
    print("=== Voice Library Statistics ===\n")
    
    voice_lib = VoiceLibrary("./sample_voices")
    voices = voice_lib.list_voices()
    
    print(f"Library location: {voice_lib.library_path}")
    print(f"Total voices: {len(voices)}\n")
    
    for voice in voices:
        profile = voice_lib.load_voice(voice['voice_id'])
        print(f"Voice: {voice['voice_id']}")
        print(f"  - Reference: '{profile.reference_text[:60]}...'")
        print(f"  - Tokens: {profile.duration_tokens}")
        print(f"  - Audio file: {profile.original_audio_path}")
        print(f"  - Metadata: {profile.metadata}")
        print()


def usage_examples():
    """Show code examples for using the voice library."""
    print("=== Usage Examples ===\n")
    
    example_code = '''
# Basic usage with voice library
from dia.model import Dia
from voice_library import VoiceLibrary, DiaWithVoiceLibrary

# Load model and voice library
model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
voice_lib = VoiceLibrary("./my_voices")
dia_with_voices = DiaWithVoiceLibrary(model, voice_lib)

# Add a new voice (one-time setup)
dia_with_voices.add_voice_to_library(
    voice_id="my_voice",
    audio_path="reference_audio.mp3", 
    reference_text="[S1] This is my reference speech..."
)

# Generate audio with cached voice (FAST!)
output = dia_with_voices.generate_with_voice(
    voice_id="my_voice",
    new_text="[S1] Hello, this is new text in my voice!",
    cfg_scale=4.0
)

# Save the output
model.save_audio("output.mp3", output)
'''
    
    print("Python code example:")
    print(example_code)
    
    print("\nCLI usage examples:")
    print("# Add a voice to library")
    print('python voice_cli.py add my_voice "audio.mp3" "[S1] Reference text here"')
    print()
    print("# Generate audio with cached voice")
    print('python voice_cli.py generate my_voice "[S1] New text" output.mp3')
    print()
    print("# List all voices")
    print('python voice_cli.py list')


def main():
    print("🎵 Dia Voice Library Quick Setup 🎵\n")
    
    try:
        # Step 1: Create sample voices
        dia_with_voices = create_sample_voices()
        
        # Step 2: Demonstrate fast generation
        demonstrate_fast_generation()
        
        # Step 3: Show library stats
        show_library_stats()
        
        # Step 4: Show usage examples
        usage_examples()
        
        print("=== Setup Complete! ===")
        print("\nNext steps:")
        print("1. Try the CLI tool: python voice_cli.py list")
        print("2. Add your own voices with real audio files")
        print("3. Generate audio instantly with cached voices!")
        print("\nBenefits:")
        print("• 3-5x faster generation (no audio re-encoding)")
        print("• Consistent voice quality")
        print("• Easy voice management")
        print("• Reusable across sessions")
        
    except Exception as e:
        print(f"Error during setup: {e}")
        print("Make sure you have the Dia model and dependencies installed.")


if __name__ == "__main__":
    main()