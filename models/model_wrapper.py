"""
Base model wrapper class for unified interface
"""

import os
import torch
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from utils.memory_manager import get_memory_manager

logger = logging.getLogger(__name__)

class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers"""
    
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        self.model_id = model_id
        self.device = device if torch.cuda.is_available() else "cpu"
        self.kwargs = kwargs
        self.model = None
        self.is_loaded = False
        self.memory_manager = get_memory_manager()
        
        # Create temp directory for audio processing
        self.temp_dir = kwargs.get("temp_dir", "temp_audio")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"初始化模型包装器: {model_id}, 设备: {self.device}")
    
    @abstractmethod
    def load_model(self):
        """Load the model - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio file - must be implemented by subclasses"""
        pass
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model is not None:
            if hasattr(self.model, 'cpu'):
                self.model.cpu()
            del self.model
            self.model = None
            self.is_loaded = False
            
            # Clean up GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info(f"模型已卸载: {self.model_id}")
    
    def ensure_model_loaded(self):
        """Ensure model is loaded"""
        if not self.is_loaded:
            with self.memory_manager.memory_context(f"loading_{self.model_id}"):
                self.load_model()
                self.is_loaded = True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "is_loaded": self.is_loaded,
            "type": self.__class__.__name__
        }
    
    def validate_audio_file(self, audio_path: str) -> bool:
        """Validate audio file exists and is readable"""
        if not os.path.exists(audio_path):
            logger.error(f"音频文件不存在: {audio_path}")
            return False
        
        if os.path.getsize(audio_path) == 0:
            logger.error(f"音频文件为空: {audio_path}")
            return False
        
        return True
    
    def get_audio_duration(self, audio_path: str) -> float:
        """Get audio file duration in seconds"""
        try:
            import librosa
            duration = librosa.get_duration(path=audio_path)
            return duration
        except Exception:
            # Fallback to ffprobe
            try:
                import subprocess
                command = [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration", 
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    audio_path
                ]
                result = subprocess.run(command, capture_output=True, text=True)
                return float(result.stdout.strip()) if result.stdout else 0.0
            except Exception as e:
                logger.error(f"无法获取音频时长: {e}")
                return 0.0
    
    def chunk_audio_if_needed(self, audio_path: str, max_duration: float = 1800) -> List[str]:
        """Split audio into chunks if too long"""
        duration = self.get_audio_duration(audio_path)
        
        if duration <= max_duration:
            return [audio_path]
        
        # Calculate number of chunks needed
        num_chunks = int(duration / max_duration) + 1
        chunk_duration = duration / num_chunks
        
        chunks = []
        try:
            import subprocess
            
            for i in range(num_chunks):
                start_time = i * chunk_duration
                chunk_path = os.path.join(
                    self.temp_dir, 
                    f"chunk_{i}_{os.path.basename(audio_path)}"
                )
                
                command = [
                    "ffmpeg", "-y", "-i", audio_path,
                    "-ss", str(start_time),
                    "-t", str(chunk_duration),
                    "-c", "copy",
                    chunk_path
                ]
                
                result = subprocess.run(command, capture_output=True)
                if result.returncode == 0:
                    chunks.append(chunk_path)
                else:
                    logger.error(f"音频分块失败: chunk {i}")
            
            logger.info(f"音频已分为 {len(chunks)} 块，每块约 {chunk_duration:.1f} 秒")
            return chunks
            
        except Exception as e:
            logger.error(f"音频分块失败: {e}")
            return [audio_path]  # Return original if chunking fails
    
    def cleanup_temp_files(self):
        """Clean up temporary files"""
        try:
            import glob
            temp_files = glob.glob(os.path.join(self.temp_dir, "*"))
            for file_path in temp_files:
                try:
                    os.remove(file_path)
                except Exception:
                    pass
            logger.debug("临时文件清理完成")
        except Exception as e:
            logger.error(f"临时文件清理失败: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        self.ensure_model_loaded()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup_temp_files()
        if exc_type is not None:
            logger.error(f"模型操作异常: {exc_val}")

class ModelFactory:
    """Factory class for creating model wrappers"""
    
    @staticmethod
    def create_model(model_id: str, device: str = "cuda", **kwargs) -> BaseModelWrapper:
        """Create appropriate model wrapper based on model_id"""
        
        # Import here to avoid circular imports
        from models.whisper_wrapper import WhisperModelWrapper
        from models.huggingface_wrapper import HuggingFaceModelWrapper
        
        if model_id.startswith("whisper-"):
            return WhisperModelWrapper(model_id, device, **kwargs)
        elif "/" in model_id:  # Hugging Face model format
            return HuggingFaceModelWrapper(model_id, device, **kwargs)
        else:
            # Default to Whisper for unknown formats
            logger.warning(f"未知模型格式: {model_id}，默认使用Whisper包装器")
            return WhisperModelWrapper(model_id, device, **kwargs)
    
    @staticmethod
    def get_available_models() -> List[str]:
        """Get list of available models"""
        return [
            "whisper-small",
            "whisper-medium", 
            "whisper-large-v3",
            "openai/whisper-medium",
            "openai/whisper-large-v3",
            "BELLE-2/Belle-whisper-large-v3-zh"
        ]
    
    @staticmethod
    def get_model_recommendations(language: str = "auto") -> List[str]:
        """Get recommended models for specific language"""
        if language == "zh" or language == "chinese":
            return [
                "BELLE-2/Belle-whisper-large-v3-zh",
                "openai/whisper-large-v3",
                "whisper-large-v3"
            ]
        elif language == "en" or language == "english":
            return [
                "openai/whisper-large-v3",
                "whisper-large-v3",
                "whisper-medium"
            ]
        else:  # auto or multilingual
            return [
                "openai/whisper-large-v3",
                "whisper-large-v3",
                "BELLE-2/Belle-whisper-large-v3-zh"
            ]
