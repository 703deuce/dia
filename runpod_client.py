"""
RunPod Client for Enhanced Dia TTS

Simple client library for interacting with the RunPod serverless TTS endpoint.
"""

import requests
import base64
import io
import time
from typing import Dict, Any, Optional, List
import soundfile as sf

class RunPodTTSClient:
    """Client for RunPod Enhanced Dia TTS API."""
    
    def __init__(self, endpoint_url: str, api_key: str):
        """
        Initialize RunPod TTS client.
        
        Args:
            endpoint_url: RunPod serverless endpoint URL
            api_key: RunPod API key
        """
        self.endpoint_url = endpoint_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
    
    def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Make request to RunPod endpoint."""
        response = requests.post(self.endpoint_url, json=payload, headers=self.headers)
        response.raise_for_status()
        
        result = response.json()
        
        if result["status"] == "COMPLETED":
            return result["output"]
        elif result["status"] == "FAILED":
            raise Exception(f"RunPod request failed: {result.get('error', 'Unknown error')}")
        else:
            raise Exception(f"Unexpected status: {result['status']}")
    
    def text_to_speech(
        self,
        text: str,
        voice: str = "S1",
        response_format: str = "wav",
        speed: float = 1.0,
        seed: Optional[int] = None
    ) -> bytes:
        """
        Convert text to speech using OpenAI-compatible endpoint.
        
        Args:
            text: Text to convert to speech
            voice: Voice to use (S1, S2, dialogue, or voice_id)
            response_format: Audio format (wav, mp3, opus)
            speed: Playback speed (0.25-4.0)
            seed: Random seed for reproducible output
            
        Returns:
            Audio bytes
        """
        payload = {
            "input": {
                "endpoint": "openai",
                "data": {
                    "input": text,
                    "voice": voice,
                    "response_format": response_format,
                    "speed": speed
                }
            }
        }
        
        if seed is not None:
            payload["input"]["data"]["seed"] = seed
        
        result = self._make_request(payload)
        audio_base64 = result["audio_base64"]
        return base64.b64decode(audio_base64)
    
    def custom_tts(
        self,
        text: str,
        voice_mode: str = "single_s1",
        voice_id: Optional[str] = None,
        enable_chunking: bool = True,
        chunk_size: int = 120,
        cfg_scale: float = 3.0,
        temperature: float = 1.3,
        top_p: float = 0.95,
        cfg_filter_top_k: int = 35,
        seed: Optional[int] = None,
        output_format: str = "wav"
    ) -> Dict[str, Any]:
        """
        Generate speech with custom parameters.
        
        Args:
            text: Text to convert to speech
            voice_mode: Voice mode (voice_library, single_s1, single_s2, dialogue)
            voice_id: Voice ID (required for voice_library mode)
            enable_chunking: Enable intelligent chunking for long text
            chunk_size: Target chunk size in characters
            cfg_scale: CFG scale (1.0-10.0)
            temperature: Temperature (0.1-2.0)
            top_p: Top-p sampling (0.1-1.0)
            cfg_filter_top_k: CFG filter top-k (1-100)
            seed: Random seed for reproducible output
            output_format: Output format (wav, mp3, opus)
            
        Returns:
            Dict with audio_bytes and generation_info
        """
        payload = {
            "input": {
                "endpoint": "custom",
                "data": {
                    "text": text,
                    "voice_mode": voice_mode,
                    "enable_chunking": enable_chunking,
                    "chunk_size": chunk_size,
                    "cfg_scale": cfg_scale,
                    "temperature": temperature,
                    "top_p": top_p,
                    "cfg_filter_top_k": cfg_filter_top_k,
                    "output_format": output_format
                }
            }
        }
        
        if voice_id:
            payload["input"]["data"]["voice_id"] = voice_id
        
        if seed is not None:
            payload["input"]["data"]["seed"] = seed
        
        result = self._make_request(payload)
        
        return {
            "audio_bytes": base64.b64decode(result["audio_base64"]),
            "generation_info": result["generation_info"],
            "generation_time": result["generation_time"],
            "audio_length": result["audio_length"]
        }
    
    def list_voices(self) -> List[Dict[str, Any]]:
        """List available voices."""
        payload = {
            "input": {
                "endpoint": "voice",
                "data": {
                    "action": "list"
                }
            }
        }
        
        result = self._make_request(payload)
        return result["voices"]
    
    def add_voice(
        self,
        voice_id: str,
        audio_bytes: bytes,
        reference_text: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Add a new voice to the library.
        
        Args:
            voice_id: Unique identifier for the voice
            audio_bytes: Audio file bytes (WAV format recommended)
            reference_text: Transcript of the audio with speaker tags
            metadata: Optional metadata dictionary
            
        Returns:
            Voice addition result
        """
        # For RunPod, we'd need to handle file upload differently
        # This is a simplified example - in practice, you'd need to:
        # 1. Upload the file to storage accessible by RunPod
        # 2. Pass the storage path to the handler
        
        # Convert audio bytes to base64 for transmission
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        payload = {
            "input": {
                "endpoint": "voice",
                "data": {
                    "action": "add",
                    "voice_id": voice_id,
                    "audio_base64": audio_base64,  # Custom field for audio data
                    "reference_text": reference_text,
                    "metadata": metadata or {}
                }
            }
        }
        
        return self._make_request(payload)
    
    def get_voice_info(self, voice_id: str) -> Dict[str, Any]:
        """Get information about a specific voice."""
        payload = {
            "input": {
                "endpoint": "voice",
                "data": {
                    "action": "info",
                    "voice_id": voice_id
                }
            }
        }
        
        result = self._make_request(payload)
        return result["voice_info"]
    
    def health_check(self) -> Dict[str, Any]:
        """Check system health."""
        payload = {
            "input": {
                "endpoint": "system",
                "data": {
                    "type": "health"
                }
            }
        }
        
        result = self._make_request(payload)
        return result["status"]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        payload = {
            "input": {
                "endpoint": "system",
                "data": {
                    "type": "stats"
                }
            }
        }
        
        result = self._make_request(payload)
        return result["stats"]
    
    def analyze_text(self, text: str, chunk_size: int = 120) -> Dict[str, Any]:
        """
        Analyze text for chunking without generating audio.
        
        Args:
            text: Text to analyze
            chunk_size: Target chunk size
            
        Returns:
            Analysis results
        """
        # This would need to be implemented in the handler
        # For now, we can estimate locally
        from enhanced_voice_library import TextChunker
        
        chunker = TextChunker()
        chunks = chunker.chunk_text_by_sentences(text, chunk_size)
        
        return {
            "original_length": len(text),
            "num_chunks": len(chunks),
            "chunks": chunks,
            "estimated_generation_time": len(chunks) * 2.5,
            "recommendations": {
                "enable_chunking": len(text) > chunk_size * 2,
                "optimal_chunk_size": min(max(len(text) // 5, 80), 200) if len(text) > 500 else chunk_size
            }
        }
    
    def save_audio(self, audio_bytes: bytes, filename: str):
        """Save audio bytes to file."""
        with open(filename, 'wb') as f:
            f.write(audio_bytes)
    
    def play_audio(self, audio_bytes: bytes):
        """Play audio bytes (requires additional dependencies)."""
        try:
            import pygame
            pygame.mixer.init()
            
            # Load audio from bytes
            audio_io = io.BytesIO(audio_bytes)
            pygame.mixer.music.load(audio_io)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except ImportError:
            print("pygame not available for audio playback. Install with: pip install pygame")
        except Exception as e:
            print(f"Playback failed: {e}")


# Convenience functions
def create_client(endpoint_url: str, api_key: str) -> RunPodTTSClient:
    """Create a RunPod TTS client."""
    return RunPodTTSClient(endpoint_url, api_key)

def quick_tts(
    endpoint_url: str,
    api_key: str,
    text: str,
    voice: str = "S1",
    save_path: Optional[str] = None
) -> bytes:
    """Quick text-to-speech generation."""
    client = create_client(endpoint_url, api_key)
    audio_bytes = client.text_to_speech(text, voice)
    
    if save_path:
        client.save_audio(audio_bytes, save_path)
    
    return audio_bytes

# Example usage
if __name__ == "__main__":
    # Example configuration
    ENDPOINT_URL = "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync"
    API_KEY = "YOUR_API_KEY"
    
    print("🧪 Testing RunPod TTS Client...")
    
    try:
        # Create client
        client = create_client(ENDPOINT_URL, API_KEY)
        
        # Health check
        print("🏥 Checking system health...")
        health = client.health_check()
        print(f"✅ System healthy: {health['healthy']}")
        print(f"🎵 Voices available: {health['voice_count']}")
        
        # List voices
        print("\n🎵 Available voices:")
        voices = client.list_voices()
        for voice in voices[:5]:  # Show first 5
            print(f"  - {voice['voice_id']}: {voice.get('metadata', {})}")
        
        # Generate speech
        print("\n🎯 Generating speech...")
        text = "This is a test of the RunPod serverless TTS system with intelligent chunking and voice consistency."
        
        result = client.custom_tts(
            text=text,
            voice_mode="single_s1",
            enable_chunking=True,
            seed=42
        )
        
        print(f"✅ Generated {result['generation_info']['chunks_processed']} chunks")
        print(f"⏱️  Generation time: {result['generation_time']:.2f}s")
        print(f"🎵 Audio duration: {result['generation_info']['audio_duration_seconds']:.1f}s")
        
        # Save audio
        client.save_audio(result['audio_bytes'], 'test_output.wav')
        print("💾 Audio saved to test_output.wav")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("🔧 Make sure to set your actual endpoint URL and API key")