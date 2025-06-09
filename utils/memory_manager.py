"""
Memory management utilities for RTX 3060 Ti optimization
"""

import gc
import torch
import psutil
import logging
from typing import Optional, Dict, Any
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class MemoryManager:
    """Memory management for optimal RTX 3060 Ti performance"""
    
    def __init__(self, gpu_memory_limit_gb: float = 6.0):
        self.gpu_memory_limit_gb = gpu_memory_limit_gb
        self.gpu_memory_limit_bytes = int(gpu_memory_limit_gb * 1024**3)
        self.monitoring_enabled = True
        
    def setup_gpu_memory(self):
        """Setup GPU memory configuration for RTX 3060 Ti"""
        if not torch.cuda.is_available():
            logger.warning("CUDA不可用，跳过GPU内存设置")
            return
        
        try:
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Set memory fraction (use 90% of available memory)
            torch.cuda.set_per_process_memory_fraction(0.9)
            
            # Enable memory mapping
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Enable memory efficient attention
            torch.backends.cuda.enable_flash_sdp(True)
            
            logger.info(f"GPU内存配置完成，限制: {self.gpu_memory_limit_gb}GB")
            
        except Exception as e:
            logger.error(f"GPU内存设置失败: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        stats = {}
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_allocated = torch.cuda.memory_allocated()
            gpu_reserved = torch.cuda.memory_reserved()
            gpu_max_allocated = torch.cuda.max_memory_allocated()
            
            stats["gpu"] = {
                "allocated_gb": gpu_allocated / 1024**3,
                "reserved_gb": gpu_reserved / 1024**3,
                "max_allocated_gb": gpu_max_allocated / 1024**3,
                "utilization_percent": (gpu_allocated / self.gpu_memory_limit_bytes) * 100
            }
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        stats["cpu"] = {
            "used_gb": cpu_memory.used / 1024**3,
            "available_gb": cpu_memory.available / 1024**3,
            "total_gb": cpu_memory.total / 1024**3,
            "percent": cpu_memory.percent
        }
        
        return stats
    
    def check_memory_availability(self, required_memory_gb: float) -> bool:
        """Check if enough memory is available for operation"""
        if torch.cuda.is_available():
            available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
            available_gb = available_memory / 1024**3
            
            if available_gb < required_memory_gb:
                logger.warning(f"GPU内存不足: 需要 {required_memory_gb:.1f}GB, 可用 {available_gb:.1f}GB")
                return False
        
        return True
    
    def cleanup_memory(self):
        """Perform memory cleanup"""
        # Python garbage collection
        gc.collect()
        
        # GPU memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.debug("内存清理完成")
    
    def monitor_memory(self) -> bool:
        """Monitor memory usage and return True if within limits"""
        if not self.monitoring_enabled:
            return True
        
        stats = self.get_memory_stats()
        
        # Check GPU memory
        if "gpu" in stats:
            gpu_util = stats["gpu"]["utilization_percent"]
            if gpu_util > 90:
                logger.warning(f"GPU内存使用率过高: {gpu_util:.1f}%")
                self.cleanup_memory()
                return False
        
        # Check CPU memory
        cpu_util = stats["cpu"]["percent"]
        if cpu_util > 85:
            logger.warning(f"CPU内存使用率过高: {cpu_util:.1f}%")
            self.cleanup_memory()
            return False
        
        return True
    
    @contextmanager
    def memory_context(self, operation_name: str = "unknown"):
        """Context manager for memory-safe operations"""
        initial_stats = self.get_memory_stats()
        
        try:
            logger.debug(f"开始操作: {operation_name}")
            yield
            
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU内存不足 - {operation_name}: {e}")
            self.cleanup_memory()
            raise
            
        except Exception as e:
            logger.error(f"操作失败 - {operation_name}: {e}")
            raise
            
        finally:
            final_stats = self.get_memory_stats()
            self.log_memory_change(initial_stats, final_stats, operation_name)
            self.cleanup_memory()
    
    def log_memory_change(self, initial_stats: Dict, final_stats: Dict, operation: str):
        """Log memory usage changes"""
        if "gpu" in initial_stats and "gpu" in final_stats:
            initial_gpu = initial_stats["gpu"]["allocated_gb"]
            final_gpu = final_stats["gpu"]["allocated_gb"]
            gpu_change = final_gpu - initial_gpu
            
            logger.debug(f"内存变化 - {operation}: GPU {gpu_change:+.2f}GB")
    
    def get_optimal_batch_size(self, model_memory_gb: float, audio_length_s: float) -> int:
        """Calculate optimal batch size based on available memory"""
        available_memory = self.gpu_memory_limit_gb - 1.0  # Reserve 1GB for overhead
        
        # Estimate memory per audio second (empirical values)
        memory_per_second = 0.01  # 10MB per second of audio
        audio_memory = audio_length_s * memory_per_second
        
        total_memory_per_item = model_memory_gb + audio_memory
        
        if total_memory_per_item > available_memory:
            return 0  # Cannot process
        
        batch_size = max(1, int(available_memory / total_memory_per_item))
        return min(batch_size, 4)  # Cap at 4 for stability
    
    def optimize_for_model(self, model_name: str) -> Dict[str, Any]:
        """Get memory optimization settings for specific model"""
        model_configs = {
            "whisper-small": {"memory_gb": 1.0, "batch_size": 2},
            "whisper-medium": {"memory_gb": 2.0, "batch_size": 1},
            "whisper-large-v3": {"memory_gb": 3.0, "batch_size": 1},
            "openai/whisper-medium": {"memory_gb": 2.5, "batch_size": 1},
            "openai/whisper-large-v3": {"memory_gb": 4.0, "batch_size": 1},
            "BELLE-2/Belle-whisper-large-v3-zh": {"memory_gb": 4.0, "batch_size": 1}
        }
        
        config = model_configs.get(model_name, {"memory_gb": 2.0, "batch_size": 1})
        
        # Adjust based on available memory
        available_memory = self.gpu_memory_limit_gb - 1.0
        if config["memory_gb"] > available_memory:
            logger.warning(f"模型 {model_name} 需要 {config['memory_gb']}GB，但只有 {available_memory}GB 可用")
            config["batch_size"] = 1
            config["chunk_length_s"] = 20  # Shorter chunks
        
        return config
    
    def enable_memory_efficient_mode(self):
        """Enable memory efficient mode for large models"""
        if torch.cuda.is_available():
            # Enable memory efficient attention
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                logger.info("启用内存高效注意力机制")
            except:
                pass
            
            # Set conservative memory settings
            torch.cuda.set_per_process_memory_fraction(0.8)
            
            # Enable gradient checkpointing for models that support it
            import os
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
            
        logger.info("启用内存高效模式")
    
    def disable_monitoring(self):
        """Disable memory monitoring for performance"""
        self.monitoring_enabled = False
        logger.debug("禁用内存监控")
    
    def enable_monitoring(self):
        """Enable memory monitoring"""
        self.monitoring_enabled = True
        logger.debug("启用内存监控")

# Global memory manager instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

def setup_memory_management(gpu_memory_limit_gb: float = 6.0) -> MemoryManager:
    """Setup global memory management"""
    global _memory_manager
    _memory_manager = MemoryManager(gpu_memory_limit_gb)
    _memory_manager.setup_gpu_memory()
    return _memory_manager
