"""
Main application class for video subtitle recognition system
Optimized for NVIDIA RTX 3060 Ti 6GB with TensorRT acceleration
"""

import os
import sys
import time
import gc
import traceback
from typing import List, Dict, Any, Optional, Tuple
import logging

# Import system modules
from config import Config
from utils.logger import setup_logger, get_logger
from utils.system_info import SystemInfo
from utils.memory_manager import setup_memory_management, get_memory_manager
from models.model_wrapper import ModelFactory
from processors.video_processor import VideoProcessor
from processors.audio_processor import AudioProcessor
from utils.subtitle_formatter import SubtitleFormatter

class SubtitleGenerationSystem:
    """Main system class for video subtitle generation"""
    
    def __init__(self, model_name: str = "whisper-medium", language: str = "auto", **kwargs):
        # Setup environment
        Config.setup_environment()
        
        # Initialize logging
        self.logger_manager = setup_logger(
            log_dir=kwargs.get("log_dir", "logs"),
            log_level=kwargs.get("log_level", logging.INFO)
        )
        self.logger = get_logger()
        
        # Log system startup
        self.logger_manager.log_system_info()
        
        # Initialize memory management
        self.memory_manager = setup_memory_management(
            gpu_memory_limit_gb=kwargs.get("gpu_memory_limit", 6.0)
        )
        
        # System validation
        self._validate_system()
        
        # Initialize components
        self.model_name = model_name
        self.language = language
        self.device = "cuda" if SystemInfo.detect_gpu()["cuda_available"] else "cpu"
        
        # Initialize processors
        self.video_processor = VideoProcessor()
        self.audio_processor = AudioProcessor(
            target_sample_rate=Config.AUDIO_CONFIG["sample_rate"],
            target_channels=Config.AUDIO_CONFIG["channels"],
            normalization=Config.AUDIO_CONFIG["normalization"],
            noise_reduction=Config.AUDIO_CONFIG["noise_reduction"]
        )
        
        # Initialize subtitle formatter
        self.subtitle_formatter = SubtitleFormatter(
            max_chars_per_line=Config.OUTPUT_CONFIG["max_chars_per_line"],
            max_lines_per_subtitle=Config.OUTPUT_CONFIG["max_lines_per_subtitle"],
            min_duration=Config.OUTPUT_CONFIG["min_duration"]
        )
        
        # Model wrapper (lazy loaded)
        self.model_wrapper = None
        self.model_kwargs = kwargs
        
        # Processing statistics
        self.stats = {
            "files_processed": 0,
            "total_duration": 0.0,
            "total_processing_time": 0.0,
            "errors": []
        }
        
        self.logger.info(f"字幕生成系统初始化完成 - 模型: {model_name}, 语言: {language}, 设备: {self.device}")
    
    def _validate_system(self):
        """Validate system requirements"""
        self.logger.info("验证系统要求...")
        
        # Check GPU
        gpu_info = SystemInfo.detect_gpu()
        if not gpu_info["cuda_available"]:
            self.logger.warning("CUDA不可用，将使用CPU模式（处理速度较慢）")
        else:
            self.logger.info(f"检测到GPU: {gpu_info['gpu_devices'][0]['name']} "
                           f"({gpu_info['gpu_devices'][0]['memory_gb']:.1f}GB)")
        
        # Check dependencies
        deps = SystemInfo.check_dependencies()
        missing_deps = [dep for dep, available in deps.items() if not available and not dep.endswith("_optional")]
        
        if missing_deps:
            self.logger.error(f"缺少必需依赖: {missing_deps}")
            raise RuntimeError(f"请安装缺少的依赖: {missing_deps}")
        
        # Check optional dependencies
        optional_missing = [dep for dep, available in deps.items() if not available and dep.endswith("_optional")]
        if optional_missing:
            self.logger.warning(f"缺少可选依赖（可能影响性能）: {optional_missing}")
        
        # Validate RTX 3060 Ti compatibility
        if SystemInfo.validate_rtx_3060_ti():
            self.logger.info("系统与RTX 3060 Ti兼容")
        else:
            self.logger.warning("系统可能不完全兼容RTX 3060 Ti，性能可能受影响")
        
        self.logger.info("系统验证完成")
    
    def _load_model(self):
        """Load the specified model"""
        if self.model_wrapper is not None:
            return
        
        try:
            self.logger.info(f"加载模型: {self.model_name}")
            self.logger_manager.log_model_info(self.model_name)
            
            # Get model configuration
            model_config = Config.get_model_config(self.model_name)
            model_config.update(self.model_kwargs)
            model_config["cache_dir"] = Config.get_paths()["models_dir"]
            
            # Create model wrapper
            self.model_wrapper = ModelFactory.create_model(
                self.model_name,
                device=self.device,
                **model_config
            )
            
            # Load model with memory management
            with self.memory_manager.memory_context("model_loading"):
                self.model_wrapper.ensure_model_loaded()
            
            self.logger.info(f"模型加载完成: {self.model_name}")
            self.memory_manager.log_memory_usage()
            
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
            traceback.print_exc()
            raise
    
    def process_video_file(self, video_path: str, output_dir: Optional[str] = None, 
                          formats: Optional[List[str]] = None) -> Dict[str, Any]:
        """Process single video file"""
        start_time = time.time()
        
        if output_dir is None:
            output_dir = Config.get_paths()["output_dir"]
        
        if formats is None:
            formats = ["srt"]
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            self.logger_manager.log_processing_start(video_path)
            
            # Validate video file
            if not self.video_processor.validate_video_file(video_path):
                raise ValueError(f"无效的视频文件: {video_path}")
            
            # Get video information
            video_info = self.video_processor.get_video_info(video_path)
            self.logger.info(f"视频信息: 时长 {video_info.get('duration', 0):.1f}秒, "
                           f"音频流 {len(video_info.get('audio_streams', []))} 个")
            
            # Extract audio
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            audio_path = os.path.join(output_dir, f"{base_name}_extracted_audio.wav")
            
            self.logger.info("提取音频...")
            extracted_audio = self.video_processor.extract_best_audio(video_path, audio_path)
            
            # Validate audio quality
            audio_quality = self.audio_processor.validate_audio_quality(extracted_audio)
            if audio_quality["warnings"]:
                for warning in audio_quality["warnings"]:
                    self.logger.warning(f"音频质量警告: {warning}")
            
            # Load model if not already loaded
            self._load_model()
            
            # Transcribe audio
            self.logger.info("开始语音识别...")
            transcription_start = time.time()
            
            with self.model_wrapper as model:
                transcription_result = model.transcribe(
                    extracted_audio,
                    language=self.language if self.language != "auto" else None,
                    beam_size=self.model_kwargs.get("beam_size", 5),
                    temperature=self.model_kwargs.get("temperature", 0.2),
                    max_duration=self.model_kwargs.get("max_duration", 1800)
                )
            
            transcription_time = time.time() - transcription_start
            
            if "error" in transcription_result:
                raise RuntimeError(f"语音识别失败: {transcription_result['error']}")
            
            segments = transcription_result.get("segments", [])
            detected_language = transcription_result.get("language", "unknown")
            
            self.logger.info(f"语音识别完成: {len(segments)} 个片段, "
                           f"语言: {detected_language}, 耗时: {transcription_time:.1f}秒")
            
            # Format subtitles
            self.logger.info("生成字幕...")
            subtitles = self.subtitle_formatter.format_segments_to_subtitles(segments)
            
            # Save subtitles in requested formats
            output_files = {}
            for format_type in formats:
                output_file = os.path.join(output_dir, f"{base_name}.{format_type}")
                saved_file = self.subtitle_formatter.save_subtitles(
                    subtitles, output_file, format_type
                )
                output_files[format_type] = saved_file
            
            # Clean up temporary files
            try:
                if os.path.exists(extracted_audio):
                    os.remove(extracted_audio)
            except Exception:
                pass
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats["files_processed"] += 1
            self.stats["total_duration"] += video_info.get("duration", 0)
            self.stats["total_processing_time"] += processing_time
            
            self.logger_manager.log_processing_complete(video_path, processing_time)
            self.memory_manager.log_memory_usage()
            
            result = {
                "success": True,
                "video_path": video_path,
                "video_info": video_info,
                "transcription": {
                    "segments_count": len(segments),
                    "language": detected_language,
                    "model": self.model_name
                },
                "subtitles": {
                    "count": len(subtitles),
                    "formats": list(output_files.keys())
                },
                "output_files": output_files,
                "processing_time": processing_time,
                "transcription_time": transcription_time
            }
            
            return result
            
        except Exception as e:
            error_msg = f"处理视频文件失败 {video_path}: {str(e)}"
            self.logger.error(error_msg)
            self.logger_manager.log_error(e, f"处理文件: {video_path}")
            
            self.stats["errors"].append({
                "file": video_path,
                "error": str(e),
                "timestamp": time.time()
            })
            
            return {
                "success": False,
                "video_path": video_path,
                "error": str(e),
                "processing_time": time.time() - start_time
            }
    
    def process_video_batch(self, video_paths: List[str], output_dir: Optional[str] = None,
                           formats: Optional[List[str]] = None, 
                           continue_on_error: bool = True) -> List[Dict[str, Any]]:
        """Process multiple video files in batch"""
        results = []
        
        self.logger.info(f"开始批量处理 {len(video_paths)} 个视频文件")
        
        for i, video_path in enumerate(video_paths, 1):
            self.logger.info(f"处理进度: {i}/{len(video_paths)} - {os.path.basename(video_path)}")
            
            try:
                result = self.process_video_file(video_path, output_dir, formats)
                results.append(result)
                
                if not result["success"] and not continue_on_error:
                    self.logger.error("遇到错误，停止批量处理")
                    break
                
                # Memory cleanup between files
                self.memory_manager.cleanup_memory()
                gc.collect()
                
            except Exception as e:
                error_result = {
                    "success": False,
                    "video_path": video_path,
                    "error": str(e),
                    "processing_time": 0
                }
                results.append(error_result)
                
                if not continue_on_error:
                    self.logger.error("遇到严重错误，停止批量处理")
                    break
        
        # Print batch summary
        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful
        
        self.logger.info(f"批量处理完成: {successful} 成功, {failed} 失败")
        
        return results
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status"""
        gpu_info = SystemInfo.detect_gpu()
        memory_stats = self.memory_manager.get_memory_stats()
        
        status = {
            "system": {
                "platform": SystemInfo.get_system_info()["platform"],
                "gpu_available": gpu_info["cuda_available"],
                "gpu_devices": gpu_info.get("gpu_devices", []),
                "tensorrt_available": gpu_info.get("tensorrt_available", False)
            },
            "memory": memory_stats,
            "model": {
                "loaded": self.model_wrapper is not None,
                "name": self.model_name,
                "device": self.device,
                "language": self.language
            },
            "statistics": self.stats.copy()
        }
        
        return status
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if self.stats["files_processed"] == 0:
            return {"message": "尚未处理任何文件"}
        
        avg_processing_time = self.stats["total_processing_time"] / self.stats["files_processed"]
        
        if self.stats["total_duration"] > 0:
            processing_speed = self.stats["total_duration"] / self.stats["total_processing_time"]
        else:
            processing_speed = 0
        
        report = {
            "files_processed": self.stats["files_processed"],
            "total_video_duration": f"{self.stats['total_duration']:.1f} 秒",
            "total_processing_time": f"{self.stats['total_processing_time']:.1f} 秒",
            "average_processing_time": f"{avg_processing_time:.1f} 秒/文件",
            "processing_speed": f"{processing_speed:.2f}x 实时速度",
            "error_count": len(self.stats["errors"]),
            "success_rate": f"{((self.stats['files_processed'] - len(self.stats['errors'])) / self.stats['files_processed'] * 100):.1f}%"
        }
        
        if self.stats["errors"]:
            report["recent_errors"] = self.stats["errors"][-5:]  # Last 5 errors
        
        return report
    
    def cleanup(self):
        """Clean up system resources"""
        self.logger.info("清理系统资源...")
        
        try:
            # Unload model
            if self.model_wrapper:
                self.model_wrapper.unload_model()
                self.model_wrapper = None
            
            # Clean up processors
            if hasattr(self.audio_processor, 'cleanup'):
                self.audio_processor.cleanup()
            
            # Memory cleanup
            self.memory_manager.cleanup_memory()
            gc.collect()
            
            self.logger.info("系统资源清理完成")
            
        except Exception as e:
            self.logger.error(f"资源清理失败: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        if exc_type is not None:
            self.logger.error(f"系统异常退出: {exc_val}")

# Factory function for easy system creation
def create_subtitle_system(model_name: str = "whisper-medium", **kwargs) -> SubtitleGenerationSystem:
    """Create subtitle generation system with optimal settings"""
    
    # Get optimal settings based on GPU
    gpu_info = SystemInfo.detect_gpu()
    if gpu_info["cuda_available"] and gpu_info["gpu_devices"]:
        gpu_memory = gpu_info["gpu_devices"][0]["memory_gb"]
        optimal_settings = SystemInfo.get_optimal_settings(gpu_memory)
        
        # Update kwargs with optimal settings
        for key, value in optimal_settings.items():
            if key not in kwargs:
                kwargs[key] = value
    
    return SubtitleGenerationSystem(model_name, **kwargs)

# CLI support functions
def list_available_models() -> List[str]:
    """Get list of available models"""
    return ModelFactory.get_available_models()

def get_model_recommendations(language: str = "auto") -> List[str]:
    """Get recommended models for language"""
    return ModelFactory.get_model_recommendations(language)

def validate_model_compatibility(model_name: str) -> Dict[str, Any]:
    """Validate if model is compatible with current system"""
    gpu_info = SystemInfo.detect_gpu()
    model_config = Config.get_model_config(model_name)
    
    compatibility = {
        "compatible": True,
        "warnings": [],
        "requirements": {}
    }
    
    # Check VRAM requirements
    required_vram = model_config.get("vram_usage_mb", 2000) / 1024  # Convert to GB
    if gpu_info["cuda_available"] and gpu_info["gpu_devices"]:
        available_vram = gpu_info["gpu_devices"][0]["memory_gb"]
        compatibility["requirements"]["vram_required"] = f"{required_vram:.1f}GB"
        compatibility["requirements"]["vram_available"] = f"{available_vram:.1f}GB"
        
        if required_vram > available_vram * 0.9:  # Use 90% as safe threshold
            compatibility["warnings"].append(f"模型可能需要 {required_vram:.1f}GB VRAM，但只有 {available_vram:.1f}GB 可用")
            if required_vram > available_vram:
                compatibility["compatible"] = False
    else:
        compatibility["warnings"].append("CUDA不可用，将使用CPU模式（速度较慢）")
    
    return compatibility

if __name__ == "__main__":
    # Simple test run
    print("视频字幕识别系统 - 主模块")
    print("请使用 python run_subtitle_generation.py 来运行完整系统")
    
    # Print system information
    SystemInfo.print_system_report()
