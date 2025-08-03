"""
Complete RunPod Serverless TTS Example

This example demonstrates how to use the Enhanced Dia TTS system on RunPod serverless
for various use cases including voice cloning, long text generation, and batch processing.
"""

import os
import time
from runpod_client import RunPodTTSClient

# Configuration - Replace with your actual values
ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
API_KEY = "YOUR_API_KEY"

def example_1_basic_tts():
    """Example 1: Basic text-to-speech generation"""
    print("🎯 Example 1: Basic TTS Generation")
    print("=" * 50)
    
    client = RunPodTTSClient(ENDPOINT_URL, API_KEY)
    
    # Simple OpenAI-compatible generation
    text = "Hello! This is a demonstration of the enhanced Dia TTS system running on RunPod serverless."
    
    print(f"📝 Text: {text}")
    print("🎵 Generating speech...")
    
    start_time = time.time()
    audio_bytes = client.text_to_speech(
        text=text,
        voice="S1",
        seed=42  # For reproducible results
    )
    
    generation_time = time.time() - start_time
    
    # Save the audio
    client.save_audio(audio_bytes, "example_1_basic.wav")
    
    print(f"✅ Generated in {generation_time:.2f}s")
    print(f"💾 Saved to example_1_basic.wav ({len(audio_bytes)} bytes)")

def example_2_long_text_chunking():
    """Example 2: Long text with intelligent chunking"""
    print("\n🧠 Example 2: Long Text with Chunking")
    print("=" * 50)
    
    client = RunPodTTSClient(ENDPOINT_URL, API_KEY)
    
    # Long text that will be automatically chunked
    long_text = """
    [S1] Welcome to our advanced text-to-speech demonstration. This system features intelligent chunking that can handle very long texts while maintaining perfect voice consistency across all segments.
    
    [S2] That's incredible! How does it work exactly? I'm really curious about the technical implementation behind this voice consistency feature.
    
    [S1] The system uses pre-processed voice embeddings that are cached and reused across chunks. This means each chunk uses exactly the same voice characteristics, creating seamless audio even for very long documents.
    
    [S2] And it can handle multiple speakers too, right? Like this conversation we're having right now?
    
    [S1] Absolutely! The chunking algorithm is speaker-aware, so it never splits text in the middle of one speaker's turn. Each chunk maintains the correct speaker identity throughout the entire generation process.
    """
    
    print(f"📝 Text length: {len(long_text)} characters")
    print("🧠 Analyzing text...")
    
    # Analyze the text first
    analysis = client.analyze_text(long_text, chunk_size=150)
    print(f"📊 Will create ~{analysis['num_chunks']} chunks")
    print(f"⏱️  Estimated time: {analysis['estimated_generation_time']:.1f}s")
    
    # Generate with custom parameters
    print("🎵 Generating speech with chunking...")
    
    result = client.custom_tts(
        text=long_text,
        voice_mode="dialogue",  # Handles multiple speakers
        enable_chunking=True,
        chunk_size=150,
        seed=42,
        cfg_scale=3.0,
        temperature=1.2
    )
    
    # Save the audio
    client.save_audio(result['audio_bytes'], "example_2_long.wav")
    
    print(f"✅ Generated {result['generation_info']['chunks_processed']} chunks")
    print(f"⏱️  Total time: {result['generation_time']:.2f}s")
    print(f"🎵 Audio duration: {result['generation_info']['audio_duration_seconds']:.1f}s")
    print(f"💾 Saved to example_2_long.wav")

def example_3_voice_library():
    """Example 3: Using voice library for consistent character voices"""
    print("\n🎭 Example 3: Voice Library Usage")
    print("=" * 50)
    
    client = RunPodTTSClient(ENDPOINT_URL, API_KEY)
    
    # Check available voices
    print("🎵 Checking available voices...")
    voices = client.list_voices()
    
    if not voices:
        print("⚠️  No voices in library. Using default voices.")
        voice_to_use = "S1"
        voice_mode = "single_s1"
    else:
        print(f"Found {len(voices)} voices:")
        for voice in voices[:3]:  # Show first 3
            print(f"  - {voice['voice_id']}: {voice.get('usage_count', 0)} uses")
        
        # Use the first available voice
        voice_to_use = voices[0]['voice_id']
        voice_mode = "voice_library"
    
    # Generate multiple texts with the same voice for consistency
    texts = [
        "This is the first sentence using our consistent voice.",
        "Here's another sentence that should sound exactly the same as the first one.",
        "And finally, this third sentence completes our demonstration of voice consistency across multiple generations."
    ]
    
    print(f"\n🎯 Generating {len(texts)} samples with {voice_to_use}...")
    
    for i, text in enumerate(texts, 1):
        result = client.custom_tts(
            text=text,
            voice_mode=voice_mode,
            voice_id=voice_to_use if voice_mode == "voice_library" else None,
            seed=42  # Same seed for consistency
        )
        
        filename = f"example_3_voice_{i}.wav"
        client.save_audio(result['audio_bytes'], filename)
        
        print(f"  ✅ Sample {i}: {result['generation_time']:.2f}s → {filename}")

def example_4_batch_processing():
    """Example 4: Batch processing multiple texts efficiently"""
    print("\n📦 Example 4: Batch Processing")
    print("=" * 50)
    
    client = RunPodTTSClient(ENDPOINT_URL, API_KEY)
    
    # Multiple texts to process
    batch_texts = {
        "intro": "[S1] Welcome to our podcast about artificial intelligence and machine learning.",
        "segment1": "[S2] Today we're discussing the latest advances in text-to-speech technology and how it's changing the way we interact with computers.",
        "segment2": "[S1] One of the most exciting developments is the ability to clone voices and maintain consistency across long-form content.",
        "outro": "[S2] Thank you for listening! Don't forget to subscribe and leave us a review."
    }
    
    print(f"📦 Processing {len(batch_texts)} segments...")
    
    results = {}
    total_time = 0
    total_audio_duration = 0
    
    for segment_id, text in batch_texts.items():
        print(f"\n🎯 Processing {segment_id}...")
        
        start_time = time.time()
        result = client.custom_tts(
            text=text,
            voice_mode="dialogue",
            enable_chunking=True,
            seed=42,  # Consistent seed across all segments
            temperature=1.1  # Slightly lower for more consistent delivery
        )
        
        processing_time = time.time() - start_time
        
        # Save individual segment
        filename = f"example_4_{segment_id}.wav"
        client.save_audio(result['audio_bytes'], filename)
        
        results[segment_id] = result
        total_time += processing_time
        total_audio_duration += result['generation_info']['audio_duration_seconds']
        
        print(f"  ✅ {segment_id}: {processing_time:.2f}s → {filename}")
        print(f"     Audio: {result['generation_info']['audio_duration_seconds']:.1f}s")
    
    print(f"\n📊 Batch Summary:")
    print(f"  - Total processing time: {total_time:.2f}s")
    print(f"  - Total audio duration: {total_audio_duration:.1f}s")
    print(f"  - Real-time factor: {total_audio_duration/total_time:.2f}x")

def example_5_performance_monitoring():
    """Example 5: Performance monitoring and optimization"""
    print("\n📊 Example 5: Performance Monitoring")
    print("=" * 50)
    
    client = RunPodTTSClient(ENDPOINT_URL, API_KEY)
    
    # Check system health
    print("🏥 System Health Check:")
    health = client.health_check()
    
    print(f"  - System healthy: {health['healthy']}")
    print(f"  - Model loaded: {health['model_loaded']}")
    print(f"  - Initialization time: {health.get('initialization_time', 'N/A')}s")
    print(f"  - Available voices: {health['voice_count']}")
    print(f"  - Cached voices: {health['cached_voices']}")
    
    if 'cuda_memory_allocated' in health:
        print(f"  - CUDA memory: {health['cuda_memory_allocated']:.1f} GB")
    
    # Get detailed stats
    print("\n📈 Performance Statistics:")
    try:
        stats = client.get_stats()
        
        voice_stats = stats['voice_library']
        print(f"  - Total generations: {voice_stats['total_generations']}")
        print(f"  - Cache hit rate: {voice_stats['cache_hit_rate']:.1%}")
        
        if 'memory_info' in stats:
            memory = stats['memory_info']
            if memory['cuda_available']:
                print(f"  - CUDA memory allocated: {memory['cuda_memory_allocated']:.1f} GB")
    
    except Exception as e:
        print(f"  ⚠️  Could not retrieve detailed stats: {e}")
    
    # Performance test
    print("\n⚡ Performance Test:")
    test_text = "This is a performance test to measure generation speed and consistency."
    
    times = []
    for i in range(3):
        start_time = time.time()
        result = client.custom_tts(
            text=test_text,
            voice_mode="single_s1",
            seed=42
        )
        processing_time = time.time() - start_time
        times.append(processing_time)
        
        print(f"  - Run {i+1}: {processing_time:.2f}s")
    
    avg_time = sum(times) / len(times)
    print(f"  - Average: {avg_time:.2f}s")
    print(f"  - Consistency: {min(times):.2f}s - {max(times):.2f}s")

def main():
    """Run all examples"""
    print("🚀 RunPod Enhanced Dia TTS - Complete Examples")
    print("=" * 60)
    
    # Check if configuration is set
    if "YOUR_ENDPOINT_ID" in ENDPOINT_URL or "YOUR_API_KEY" in API_KEY:
        print("⚠️  Please update ENDPOINT_URL and API_KEY with your actual values!")
        print("\nTo get started:")
        print("1. Deploy the enhanced TTS system to RunPod")
        print("2. Get your endpoint URL and API key")
        print("3. Update the configuration at the top of this file")
        print("4. Run this script again")
        return
    
    try:
        # Test connection first
        client = RunPodTTSClient(ENDPOINT_URL, API_KEY)
        health = client.health_check()
        
        if not health['healthy']:
            print("❌ System not healthy. Please check your RunPod deployment.")
            return
        
        print(f"✅ Connected to RunPod! System initialized in {health.get('initialization_time', 'N/A')}s")
        print(f"🎵 {health['voice_count']} voices available")
        
        # Run all examples
        example_1_basic_tts()
        example_2_long_text_chunking()
        example_3_voice_library()
        example_4_batch_processing()
        example_5_performance_monitoring()
        
        print("\n" + "=" * 60)
        print("✅ All examples completed successfully!")
        print("\n🎉 Your RunPod Enhanced Dia TTS system is working perfectly!")
        print("\n💡 Next steps:")
        print("   - Add your own voices to the library")
        print("   - Integrate into your applications")
        print("   - Scale up for production workloads")
        
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Check your endpoint URL and API key")
        print("   - Verify your RunPod deployment is running")
        print("   - Check RunPod logs for any errors")
        print("   - Ensure sufficient GPU memory is available")

if __name__ == "__main__":
    main()