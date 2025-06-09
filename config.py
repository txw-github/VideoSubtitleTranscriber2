"""
Configuration module for video subtitle recognition system
Optimized for NVIDIA RTX 3060 Ti 6GB
"""

import os
import platform
from typing import Dict, Any, List

class Config:
    """Configuration class with GPU memory optimization for RTX 3060 Ti"""
    
    # System Configuration
    SYSTEM_PLATFORM = platform.system()
    GPU_MEMORY_LIMIT_GB = 6  # RTX 3060 Ti VRAM
    CUDA_VERSION = "12.1"
    
    # Model Configuration
    DEFAULT_MODELS = {
        "whisper": {
            "small": "whisper-small",
            "medium": "whisper-medium", 
            "large": "whisper-large-v3"
        },
        "huggingface": {
            "openai_whisper_medium": "openai/whisper-medium",
            "openai_whisper_large": "openai/whisper-large-v3",
            "belle_whisper": "BELLE-2/Belle-whisper-large-v3-zh"
        }
    }
    
    # Memory Management
    MEMORY_CONFIG = {
        "max_split_size_mb": 512,
        "expandable_segments": True,
        "cuda_lazy_loading": True,
        "batch_size_small": 1,
        "batch_size_medium": 1,
        "batch_size_large": 1,
        "chunk_length_s": 30,
        "max_audio_length_s": 1800  # 30 minutes max per chunk
    }
    
    # TensorRT Configuration
    TENSORRT_CONFIG = {
        "precision": "fp16",
        "max_workspace_size": 2 << 30,  # 2GB
        "max_batch_size": 1,
        "optimization_level": 3,
        "enable_fp16": True,
        "enable_int8": False,  # Disabled for better accuracy
        "calibration_cache": "tensorrt_calibration.cache"
    }
    
    # Audio Processing
    AUDIO_CONFIG = {
        "sample_rate": 16000,
        "channels": 1,
        "bit_depth": 16,
        "normalization": True,
        "noise_reduction": True,
        "loudness_normalization": -23.0,  # LUFS
        "temp_dir": "temp_audio"
    }
    
    # Video Processing
    VIDEO_CONFIG = {
        "supported_formats": [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".webm"],
        "audio_codec": "pcm_s16le",
        "audio_bitrate": "128k",
        "extract_audio_only": True
    }
    
    # Output Configuration
    OUTPUT_CONFIG = {
        "formats": ["srt", "vtt", "txt"],
        "default_format": "srt",
        "encoding": "utf-8",
        "max_chars_per_line": 80,
        "max_lines_per_subtitle": 2,
        "min_duration": 0.5,  # Minimum subtitle duration in seconds
        "output_dir": "output"
    }
    
    # Language Configuration
    LANGUAGE_CONFIG = {
        "supported_languages": ["zh", "en", "auto"],
        "default_language": "auto",
        "chinese_models": ["BELLE-2/Belle-whisper-large-v3-zh"],
        "multilingual_models": ["openai/whisper-large-v3", "whisper-large-v3"]
    }
    
    # Paths Configuration
    @staticmethod
    def get_paths() -> Dict[str, str]:
        """Get system-specific paths"""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        paths = {
            "base_dir": base_dir,
            "models_dir": os.path.join(base_dir, "models_cache"),
            "temp_dir": os.path.join(base_dir, "temp"),
            "output_dir": os.path.join(base_dir, "output"),
            "logs_dir": os.path.join(base_dir, "logs"),
            "cache_dir": os.path.join(base_dir, "cache")
        }
        
        # Create directories if they don't exist
        for path in paths.values():
            os.makedirs(path, exist_ok=True)
            
        # Windows-specific paths
        if Config.SYSTEM_PLATFORM == "Windows":
            # FFmpeg path (user should modify according to their installation)
            ffmpeg_paths = [
                r"D:\code\ffmpeg\bin",
                r"C:\ffmpeg\bin",
                r"C:\Program Files\ffmpeg\bin",
                os.path.join(os.environ.get("USERPROFILE", ""), "ffmpeg", "bin")
            ]
            
            for ffmpeg_path in ffmpeg_paths:
                if os.path.exists(ffmpeg_path):
                    paths["ffmpeg"] = ffmpeg_path
                    if ffmpeg_path not in os.environ["PATH"]:
                        os.environ["PATH"] += os.pathsep + ffmpeg_path
                    break
            
            # CUDA path
            cuda_paths = [
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
                r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
            ]
            
            for cuda_path in cuda_paths:
                if os.path.exists(cuda_path):
                    paths["cuda"] = cuda_path
                    if cuda_path not in os.environ["PATH"]:
                        os.environ["PATH"] = cuda_path + ";" + os.environ["PATH"]
                    break
        
        return paths
    
    @staticmethod
    def get_model_config(model_name: str) -> Dict[str, Any]:
        """Get configuration for specific model"""
        model_configs = {
            "whisper-small": {
                "compute_type": "int8_float16",
                "batch_size": 1,
                "beam_size": 5,
                "temperature": 0.2,
                "vram_usage_mb": 1000
            },
            "whisper-medium": {
                "compute_type": "int8_float16", 
                "batch_size": 1,
                "beam_size": 5,
                "temperature": 0.2,
                "vram_usage_mb": 2000
            },
            "whisper-large-v3": {
                "compute_type": "int8_float16",
                "batch_size": 1,
                "beam_size": 5,
                "temperature": 0.2,
                "vram_usage_mb": 3000
            },
            "openai/whisper-medium": {
                "torch_dtype": "float16",
                "batch_size": 1,
                "max_new_tokens": 4096,
                "chunk_length_s": 30,
                "vram_usage_mb": 2500
            },
            "openai/whisper-large-v3": {
                "torch_dtype": "float16",
                "batch_size": 1,
                "max_new_tokens": 4096,
                "chunk_length_s": 30,
                "vram_usage_mb": 4000
            },
            "BELLE-2/Belle-whisper-large-v3-zh": {
                "torch_dtype": "float16",
                "batch_size": 1,
                "max_new_tokens": 4096,
                "chunk_length_s": 30,
                "language": "zh",
                "vram_usage_mb": 4000
            }
        }
        
        return model_configs.get(model_name, {
            "compute_type": "int8_float16",
            "batch_size": 1,
            "beam_size": 5,
            "temperature": 0.2,
            "vram_usage_mb": 2000
        })
    
    @staticmethod
    def setup_environment():
        """Setup environment variables for optimal performance"""
        env_vars = {
            "CUDA_MODULE_LOADING": "LAZY",
            "CUDA_LAZY_LOADING": "1", 
            "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512,expandable_segments:True",
            "TOKENIZERS_PARALLELISM": "false",
            "OMP_NUM_THREADS": "4"
        }
        
        for key, value in env_vars.items():
            os.environ[key] = value
