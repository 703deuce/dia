#!/usr/bin/env python3
"""
Create a demo voice for testing the voice library system
"""

from dia.model import Dia
from voice_library import VoiceLibrary, DiaWithVoiceLibrary

def create_demo_voice():
    print("🎵 Creating Demo Voice for Voice Library...")
    
    # Load Dia model
    print("1. Loading Dia model...")
    model = Dia.from_pretrained("nari-labs/Dia-1.6B-0626", compute_dtype="float16")
    print("   ✓ Model loaded!")
    
    # Create demo reference audio
    print("2. Generating demo reference audio...")
    demo_text = "[S1] Hello, I am a demo speaker for the voice library system. This voice will be cached for instant reuse."
    
    audio = model.generate(demo_text, cfg_scale=3.0, verbose=True)
    model.save_audio("demo_reference.mp3", audio)
    print("   ✓ Demo reference audio created: demo_reference.mp3")
    
    # Initialize voice library
    print("3. Setting up voice library...")
    voice_lib = VoiceLibrary("./my_voice_library")
    dia_with_voices = DiaWithVoiceLibrary(model, voice_lib)
    print("   ✓ Voice library initialized!")
    
    # Add the demo voice to library
    print("4. Adding demo voice to library...")
    try:
        profile = dia_with_voices.add_voice_to_library(
            voice_id="demo_speaker",
            audio_path="demo_reference.mp3",
            reference_text=demo_text,
            metadata={"speaker": "Demo Speaker", "purpose": "tutorial"}
        )
        print(f"   ✓ Voice added! Profile: {profile.voice_id} ({profile.duration_tokens} tokens)")
    except ValueError as e:
        print(f"   - Voice already exists: {e}")
    
    # Test the cached voice generation
    print("5. Testing fast generation with cached voice...")
    test_text = "[S1] This is a test of the fast voice generation using the cached voice library!"
    
    import time
    start_time = time.time()
    
    output = dia_with_voices.generate_with_voice(
        voice_id="demo_speaker",
        new_text=test_text,
        cfg_scale=4.0,
        verbose=True
    )
    
    generation_time = time.time() - start_time
    model.save_audio("fast_demo_output.mp3", output)
    
    print(f"   ✓ Generated in {generation_time:.2f} seconds!")
    print(f"   ✓ Output saved: fast_demo_output.mp3")
    
    # Show library status
    print("6. Voice library status:")
    voices = voice_lib.list_voices()
    print(f"   - Total voices: {len(voices)}")
    print(f"   - Library location: {voice_lib.library_path}")
    
    for voice in voices:
        print(f"   - {voice['voice_id']}: {voice['duration_tokens']} tokens")
    
    print("\n🎉 Demo Complete! Your voice library is ready!")
    print("\nNext steps:")
    print("• Try: python voice_cli.py list")
    print("• Add your own voice: python voice_cli.py add 'my_voice' 'audio.mp3' '[S1] transcript'")
    print("• Generate instantly: python voice_cli.py generate 'demo_speaker' '[S1] New text' 'output.mp3'")

if __name__ == "__main__":
    create_demo_voice()