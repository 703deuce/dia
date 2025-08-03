"""
Audio Post-Processing Utilities for Enhanced Dia TTS

This module provides audio enhancement features including:
- Silence trimming and removal
- Noise reduction
- Audio normalization
- Format conversion
- Quality optimization
"""

import numpy as np
import soundfile as sf
import logging
from typing import Tuple, Optional
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

logger = logging.getLogger(__name__)

# Optional imports for advanced processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available. Some audio processing features will be disabled.")

try:
    import scipy.signal
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available. Advanced filtering will be disabled.")

class AudioProcessor:
    """Advanced audio processing utilities."""
    
    def __init__(self, sample_rate: int = 44100):
        self.sample_rate = sample_rate
        
    def trim_silence(
        self, 
        audio: np.ndarray, 
        threshold: float = 0.01,
        frame_length: int = 1024,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        Trim silence from the beginning and end of audio.
        
        Args:
            audio: Input audio array
            threshold: Amplitude threshold for silence detection
            frame_length: Frame length for analysis
            hop_length: Hop length for analysis
            
        Returns:
            Trimmed audio array
        """
        if not LIBROSA_AVAILABLE:
            return self._simple_trim_silence(audio, threshold)
        
        try:
            # Use librosa for more sophisticated trimming
            trimmed_audio, _ = librosa.effects.trim(
                audio,
                top_db=20,  # Trim silence below -20dB
                frame_length=frame_length,
                hop_length=hop_length
            )
            
            return trimmed_audio
            
        except Exception as e:
            logger.warning(f"Librosa trimming failed, using simple method: {e}")
            return self._simple_trim_silence(audio, threshold)
    
    def _simple_trim_silence(self, audio: np.ndarray, threshold: float) -> np.ndarray:
        """Simple silence trimming without librosa."""
        # Find non-silent regions
        non_silent = np.abs(audio) > threshold
        
        if not np.any(non_silent):
            # All audio is silent, return a small segment
            return audio[:int(0.1 * self.sample_rate)]
        
        # Find first and last non-silent samples
        non_silent_indices = np.where(non_silent)[0]
        start_idx = max(0, non_silent_indices[0] - int(0.1 * self.sample_rate))  # Keep 0.1s before
        end_idx = min(len(audio), non_silent_indices[-1] + int(0.1 * self.sample_rate))  # Keep 0.1s after
        
        return audio[start_idx:end_idx]
    
    def normalize_audio(
        self, 
        audio: np.ndarray, 
        target_lufs: float = -23.0,
        peak_normalize: bool = True
    ) -> np.ndarray:
        """
        Normalize audio to target loudness.
        
        Args:
            audio: Input audio array
            target_lufs: Target loudness in LUFS (if available)
            peak_normalize: Whether to peak normalize
            
        Returns:
            Normalized audio array
        """
        if peak_normalize:
            # Simple peak normalization
            peak = np.max(np.abs(audio))
            if peak > 0:
                # Normalize to -3dB to avoid clipping
                target_peak = 0.7
                audio = audio * (target_peak / peak)
        
        # Additional loudness normalization would require pyloudnorm
        # For now, we'll stick with peak normalization
        
        return audio
    
    def remove_clicks_and_pops(self, audio: np.ndarray, threshold: float = 0.9) -> np.ndarray:
        """
        Remove clicks and pops from audio.
        
        Args:
            audio: Input audio array
            threshold: Threshold for click detection
            
        Returns:
            Cleaned audio array
        """
        if not SCIPY_AVAILABLE:
            return audio
        
        try:
            # Detect sudden amplitude changes
            diff = np.abs(np.diff(audio))
            click_indices = np.where(diff > threshold)[0]
            
            # Replace clicks with interpolated values
            cleaned_audio = audio.copy()
            for idx in click_indices:
                if idx > 0 and idx < len(audio) - 1:
                    # Simple linear interpolation
                    cleaned_audio[idx] = (audio[idx-1] + audio[idx+1]) / 2
            
            return cleaned_audio
            
        except Exception as e:
            logger.warning(f"Click removal failed: {e}")
            return audio
    
    def apply_gentle_filtering(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply gentle high-pass and low-pass filtering to improve audio quality.
        
        Args:
            audio: Input audio array
            
        Returns:
            Filtered audio array
        """
        if not SCIPY_AVAILABLE:
            return audio
        
        try:
            # Gentle high-pass filter to remove DC offset and low-frequency noise
            sos_hp = scipy.signal.butter(2, 50, btype='high', fs=self.sample_rate, output='sos')
            filtered_audio = scipy.signal.sosfiltfilt(sos_hp, audio)
            
            # Gentle low-pass filter to remove high-frequency artifacts
            sos_lp = scipy.signal.butter(2, 8000, btype='low', fs=self.sample_rate, output='sos')
            filtered_audio = scipy.signal.sosfiltfilt(sos_lp, filtered_audio)
            
            return filtered_audio
            
        except Exception as e:
            logger.warning(f"Filtering failed: {e}")
            return audio
    
    def reduce_internal_silence(
        self, 
        audio: np.ndarray, 
        max_silence_duration: float = 0.5,
        silence_threshold: float = 0.01
    ) -> np.ndarray:
        """
        Reduce long internal silences while preserving natural pauses.
        
        Args:
            audio: Input audio array
            max_silence_duration: Maximum allowed silence duration in seconds
            silence_threshold: Amplitude threshold for silence detection
            
        Returns:
            Audio with reduced internal silences
        """
        if not LIBROSA_AVAILABLE:
            return self._simple_reduce_silence(audio, max_silence_duration, silence_threshold)
        
        try:
            # Detect silent regions
            silent_mask = np.abs(audio) < silence_threshold
            
            # Find continuous silent regions
            silent_regions = []
            in_silence = False
            silence_start = 0
            
            for i, is_silent in enumerate(silent_mask):
                if is_silent and not in_silence:
                    silence_start = i
                    in_silence = True
                elif not is_silent and in_silence:
                    silent_regions.append((silence_start, i))
                    in_silence = False
            
            # Handle case where audio ends in silence
            if in_silence:
                silent_regions.append((silence_start, len(audio)))
            
            # Reduce long silences
            max_silence_samples = int(max_silence_duration * self.sample_rate)
            output_audio = []
            last_end = 0
            
            for start, end in silent_regions:
                # Add non-silent audio before this silence
                output_audio.append(audio[last_end:start])
                
                # Add reduced silence
                silence_duration = end - start
                if silence_duration > max_silence_samples:
                    # Keep only max_silence_duration of silence
                    reduced_silence = audio[start:start + max_silence_samples]
                    output_audio.append(reduced_silence)
                else:
                    # Keep original short silence
                    output_audio.append(audio[start:end])
                
                last_end = end
            
            # Add remaining audio
            if last_end < len(audio):
                output_audio.append(audio[last_end:])
            
            return np.concatenate(output_audio) if output_audio else audio
            
        except Exception as e:
            logger.warning(f"Silence reduction failed: {e}")
            return audio
    
    def _simple_reduce_silence(
        self, 
        audio: np.ndarray, 
        max_silence_duration: float, 
        silence_threshold: float
    ) -> np.ndarray:
        """Simple silence reduction without librosa."""
        # This is a simplified version
        max_silence_samples = int(max_silence_duration * self.sample_rate)
        
        # Find silent samples
        silent_mask = np.abs(audio) < silence_threshold
        
        # Simple approach: just limit consecutive silent samples
        output_audio = []
        consecutive_silence = 0
        
        for i, (sample, is_silent) in enumerate(zip(audio, silent_mask)):
            if is_silent:
                consecutive_silence += 1
                if consecutive_silence <= max_silence_samples:
                    output_audio.append(sample)
            else:
                consecutive_silence = 0
                output_audio.append(sample)
        
        return np.array(output_audio)
    
    def enhance_audio(
        self,
        audio: np.ndarray,
        enable_trimming: bool = True,
        enable_normalization: bool = True,
        enable_filtering: bool = True,
        enable_silence_reduction: bool = True,
        enable_click_removal: bool = False  # Disabled by default as it can be aggressive
    ) -> Tuple[np.ndarray, Dict]:
        """
        Apply full audio enhancement pipeline.
        
        Args:
            audio: Input audio array
            enable_*: Flags to enable/disable specific processing steps
            
        Returns:
            Tuple of (enhanced_audio, processing_info)
        """
        processing_info = {
            "original_length": len(audio),
            "processing_steps": []
        }
        
        enhanced_audio = audio.copy()
        
        try:
            # 1. Trim silence
            if enable_trimming:
                before_length = len(enhanced_audio)
                enhanced_audio = self.trim_silence(enhanced_audio)
                after_length = len(enhanced_audio)
                
                processing_info["processing_steps"].append({
                    "step": "silence_trimming",
                    "samples_removed": before_length - after_length,
                    "duration_removed_seconds": (before_length - after_length) / self.sample_rate
                })
            
            # 2. Remove clicks and pops
            if enable_click_removal:
                enhanced_audio = self.remove_clicks_and_pops(enhanced_audio)
                processing_info["processing_steps"].append({"step": "click_removal"})
            
            # 3. Apply gentle filtering
            if enable_filtering:
                enhanced_audio = self.apply_gentle_filtering(enhanced_audio)
                processing_info["processing_steps"].append({"step": "gentle_filtering"})
            
            # 4. Reduce internal silences
            if enable_silence_reduction:
                before_length = len(enhanced_audio)
                enhanced_audio = self.reduce_internal_silence(enhanced_audio)
                after_length = len(enhanced_audio)
                
                processing_info["processing_steps"].append({
                    "step": "silence_reduction",
                    "samples_removed": before_length - after_length,
                    "duration_removed_seconds": (before_length - after_length) / self.sample_rate
                })
            
            # 5. Normalize audio (do this last)
            if enable_normalization:
                enhanced_audio = self.normalize_audio(enhanced_audio)
                processing_info["processing_steps"].append({"step": "normalization"})
            
            processing_info.update({
                "final_length": len(enhanced_audio),
                "total_duration_seconds": len(enhanced_audio) / self.sample_rate,
                "compression_ratio": len(audio) / len(enhanced_audio) if len(enhanced_audio) > 0 else 1.0
            })
            
        except Exception as e:
            logger.error(f"Audio enhancement failed: {e}")
            # Return original audio if enhancement fails
            enhanced_audio = audio
            processing_info["error"] = str(e)
        
        return enhanced_audio, processing_info
    
    def convert_format(
        self,
        audio: np.ndarray,
        target_format: str,
        quality: str = "high"
    ) -> bytes:
        """
        Convert audio to specified format.
        
        Args:
            audio: Input audio array
            target_format: Target format (wav, mp3, opus, etc.)
            quality: Quality setting (low, medium, high)
            
        Returns:
            Encoded audio bytes
        """
        import io
        
        buffer = io.BytesIO()
        
        # Quality settings
        quality_map = {
            "low": {"wav": None, "mp3": 128, "opus": 64},
            "medium": {"wav": None, "mp3": 192, "opus": 96},
            "high": {"wav": None, "mp3": 320, "opus": 128}
        }
        
        try:
            if target_format.lower() == "wav":
                sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
            elif target_format.lower() == "mp3":
                # For MP3, we need additional libraries. For now, use WAV
                logger.warning("MP3 encoding not available, using WAV")
                sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
            elif target_format.lower() == "opus":
                # For Opus, we need additional libraries. For now, use WAV
                logger.warning("Opus encoding not available, using WAV")
                sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
            else:
                # Default to WAV
                sf.write(buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
            
            buffer.seek(0)
            return buffer.read()
            
        except Exception as e:
            logger.error(f"Format conversion failed: {e}")
            # Fallback to WAV
            buffer = io.BytesIO()
            sf.write(buffer, audio, self.sample_rate, format="WAV")
            buffer.seek(0)
            return buffer.read()


# Convenience functions for common operations
def quick_enhance_audio(
    audio: np.ndarray,
    sample_rate: int = 44100,
    preset: str = "balanced"
) -> Tuple[np.ndarray, Dict]:
    """
    Quick audio enhancement with presets.
    
    Args:
        audio: Input audio array
        sample_rate: Audio sample rate
        preset: Enhancement preset (minimal, balanced, aggressive)
        
    Returns:
        Tuple of (enhanced_audio, processing_info)
    """
    processor = AudioProcessor(sample_rate)
    
    presets = {
        "minimal": {
            "enable_trimming": True,
            "enable_normalization": True,
            "enable_filtering": False,
            "enable_silence_reduction": False,
            "enable_click_removal": False
        },
        "balanced": {
            "enable_trimming": True,
            "enable_normalization": True,
            "enable_filtering": True,
            "enable_silence_reduction": True,
            "enable_click_removal": False
        },
        "aggressive": {
            "enable_trimming": True,
            "enable_normalization": True,
            "enable_filtering": True,
            "enable_silence_reduction": True,
            "enable_click_removal": True
        }
    }
    
    settings = presets.get(preset, presets["balanced"])
    return processor.enhance_audio(audio, **settings)


def batch_process_audio_files(
    input_files: list,
    output_dir: str,
    enhancement_preset: str = "balanced"
):
    """
    Batch process audio files with enhancement.
    
    Args:
        input_files: List of input file paths
        output_dir: Output directory
        enhancement_preset: Enhancement preset to use
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    for input_file in input_files:
        try:
            # Load audio
            audio, sample_rate = sf.read(input_file)
            
            # Enhance
            enhanced_audio, processing_info = quick_enhance_audio(
                audio, sample_rate, enhancement_preset
            )
            
            # Save
            output_file = os.path.join(
                output_dir,
                f"enhanced_{os.path.basename(input_file)}"
            )
            sf.write(output_file, enhanced_audio, sample_rate)
            
            logger.info(f"Processed {input_file} -> {output_file}")
            logger.debug(f"Processing info: {processing_info}")
            
        except Exception as e:
            logger.error(f"Failed to process {input_file}: {e}")