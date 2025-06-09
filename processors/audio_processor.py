"""
Audio processing utilities for optimal speech recognition
"""

import os
import numpy as np
import soundfile as sf
import logging
from typing import Tuple, Optional, List
import subprocess

logger = logging.getLogger(__name__)

class AudioProcessor:
    """Audio processing for speech recognition optimization"""
    
    def __init__(self, 
                 target_sample_rate: int = 16000,
                 target_channels: int = 1,
                 normalization: bool = True,
                 noise_reduction: bool = True):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.normalization = normalization
        self.noise_reduction = noise_reduction
        
    def process_audio_file(self, input_path: str, output_path: Optional[str] = None) -> str:
        """Process audio file for optimal speech recognition"""
        if output_path is None:
            name, ext = os.path.splitext(input_path)
            output_path = f"{name}_processed.wav"
        
        try:
            logger.info(f"处理音频文件: {os.path.basename(input_path)}")
            
            # Load audio
            audio_data, sample_rate = self._load_audio(input_path)
            
            # Process audio
            processed_audio = self._process_audio_data(audio_data, sample_rate)
            
            # Save processed audio
            sf.write(output_path, processed_audio, self.target_sample_rate)
            
            logger.info(f"音频处理完成: {os.path.basename(output_path)}")
            return output_path
            
        except Exception as e:
            logger.error(f"音频处理失败: {e}")
            # Return original file if processing fails
            return input_path
    
    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file with fallback methods"""
        try:
            # Try soundfile first
            audio_data, sample_rate = sf.read(audio_path)
            logger.debug(f"使用soundfile加载音频: {sample_rate}Hz, {audio_data.shape}")
            return audio_data, sample_rate
            
        except Exception as e:
            logger.warning(f"soundfile加载失败，尝试其他方法: {e}")
            
            # Fallback to librosa
            try:
                import librosa
                audio_data, sample_rate = librosa.load(audio_path, sr=None)
                logger.debug(f"使用librosa加载音频: {sample_rate}Hz, {audio_data.shape}")
                return audio_data, sample_rate
                
            except Exception as e:
                logger.error(f"librosa加载失败: {e}")
                raise
    
    def _process_audio_data(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Process audio data for optimal recognition"""
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = self._convert_to_mono(audio_data)
        
        # Resample if needed
        if sample_rate != self.target_sample_rate:
            audio_data = self._resample_audio(audio_data, sample_rate, self.target_sample_rate)
        
        # Normalize audio
        if self.normalization:
            audio_data = self._normalize_audio(audio_data)
        
        # Apply noise reduction
        if self.noise_reduction:
            audio_data = self._reduce_noise(audio_data)
        
        # Apply loudness normalization
        audio_data = self._normalize_loudness(audio_data, self.target_sample_rate)
        
        return audio_data
    
    def _convert_to_mono(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert stereo/multi-channel audio to mono"""
        if len(audio_data.shape) == 2:
            # Average all channels
            return np.mean(audio_data, axis=1)
        return audio_data
    
    def _resample_audio(self, audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        try:
            import librosa
            return librosa.resample(audio_data, orig_sr=orig_sr, target_sr=target_sr)
        except ImportError:
            logger.warning("librosa不可用，跳过重采样")
            return audio_data
    
    def _normalize_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Normalize audio amplitude"""
        # RMS normalization
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0:
            target_rms = 0.1  # Target RMS level
            audio_data = audio_data * (target_rms / rms)
        
        # Peak normalization
        max_val = np.max(np.abs(audio_data))
        if max_val > 1.0:
            audio_data = audio_data / max_val
        
        return audio_data
    
    def _reduce_noise(self, audio_data: np.ndarray) -> np.ndarray:
        """Apply simple noise reduction"""
        try:
            # Simple spectral gating for noise reduction
            import scipy.signal
            
            # Apply high-pass filter to remove low-frequency noise
            sos = scipy.signal.butter(4, 80, 'hp', fs=self.target_sample_rate, output='sos')
            audio_data = scipy.signal.sosfilt(sos, audio_data)
            
            # Apply noise gate
            threshold = np.percentile(np.abs(audio_data), 20)  # 20th percentile as noise floor
            mask = np.abs(audio_data) > threshold
            audio_data = audio_data * mask
            
            return audio_data
            
        except ImportError:
            logger.debug("scipy不可用，跳过噪声降噪")
            return audio_data
        except Exception as e:
            logger.warning(f"噪声降噪失败: {e}")
            return audio_data
    
    def _normalize_loudness(self, audio_data: np.ndarray, sample_rate: int) -> np.ndarray:
        """Normalize loudness using pyloudnorm"""
        try:
            import pyloudnorm as pyln
            
            # Measure loudness
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(audio_data)
            
            # Normalize to -23 LUFS (broadcast standard)
            if not np.isnan(loudness) and not np.isinf(loudness):
                normalized_audio = pyln.normalize.loudness(audio_data, loudness, -23.0)
                return normalized_audio
            
        except ImportError:
            logger.debug("pyloudnorm不可用，跳过响度标准化")
        except Exception as e:
            logger.warning(f"响度标准化失败: {e}")
        
        return audio_data
    
    def extract_audio_from_video(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract audio from video file using FFmpeg"""
        if output_path is None:
            name, _ = os.path.splitext(video_path)
            output_path = f"{name}_audio.wav"
        
        try:
            logger.info(f"从视频提取音频: {os.path.basename(video_path)}")
            
            # FFmpeg command for audio extraction
            command = [
                "ffmpeg", "-y",  # Overwrite output
                "-i", video_path,
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", str(self.target_sample_rate),  # Sample rate
                "-ac", str(self.target_channels),  # Channels
                "-af", "volume=1.0,highpass=f=80,lowpass=f=8000",  # Audio filters
                output_path
            ]
            
            result = subprocess.run(
                command, 
                capture_output=True, 
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode != 0:
                logger.error(f"FFmpeg执行失败: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command)
            
            logger.info(f"音频提取完成: {os.path.basename(output_path)}")
            return output_path
            
        except subprocess.TimeoutExpired:
            logger.error("音频提取超时")
            raise
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            raise
    
    def split_audio_by_silence(self, audio_path: str, 
                              min_silence_len: float = 1.0,
                              silence_thresh: float = -40) -> List[str]:
        """Split audio by silence detection"""
        try:
            from pydub import AudioSegment
            from pydub.silence import split_on_silence
            
            # Load audio
            audio = AudioSegment.from_wav(audio_path)
            
            # Split on silence
            chunks = split_on_silence(
                audio,
                min_silence_len=int(min_silence_len * 1000),  # Convert to ms
                silence_thresh=silence_thresh,
                keep_silence=500  # Keep 500ms of silence
            )
            
            # Save chunks
            chunk_paths = []
            base_name = os.path.splitext(audio_path)[0]
            
            for i, chunk in enumerate(chunks):
                if len(chunk) > 1000:  # Only save chunks longer than 1 second
                    chunk_path = f"{base_name}_chunk_{i:03d}.wav"
                    chunk.export(chunk_path, format="wav")
                    chunk_paths.append(chunk_path)
            
            logger.info(f"音频分割完成: {len(chunk_paths)} 个片段")
            return chunk_paths
            
        except ImportError:
            logger.warning("pydub不可用，无法进行静音分割")
            return [audio_path]
        except Exception as e:
            logger.error(f"音频分割失败: {e}")
            return [audio_path]
    
    def get_audio_info(self, audio_path: str) -> dict:
        """Get audio file information"""
        try:
            info = sf.info(audio_path)
            return {
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
                "frames": info.frames,
                "format": info.format,
                "subtype": info.subtype
            }
        except Exception as e:
            logger.error(f"获取音频信息失败: {e}")
            return {}
    
    def validate_audio_quality(self, audio_path: str) -> dict:
        """Validate audio quality for speech recognition"""
        info = self.get_audio_info(audio_path)
        
        quality_report = {
            "valid": True,
            "warnings": [],
            "recommendations": []
        }
        
        # Check duration
        if info.get("duration", 0) > 3600:  # > 1 hour
            quality_report["warnings"].append("音频过长，建议分割处理")
            quality_report["recommendations"].append("使用分块处理模式")
        
        # Check sample rate
        if info.get("sample_rate", 0) < 16000:
            quality_report["warnings"].append("采样率较低，可能影响识别质量")
            quality_report["recommendations"].append("建议使用16kHz或更高采样率")
        
        # Check channels
        if info.get("channels", 1) > 1:
            quality_report["warnings"].append("多声道音频，建议转换为单声道")
            quality_report["recommendations"].append("启用音频预处理")
        
        return quality_report
