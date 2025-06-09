"""
Logging utility for video subtitle recognition system
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional

class SubtitleLogger:
    """Custom logger for subtitle generation system"""
    
    def __init__(self, log_dir: str = "logs", log_level: int = logging.INFO):
        self.log_dir = log_dir
        self.log_level = log_level
        os.makedirs(log_dir, exist_ok=True)
        
        # Create logger
        self.logger = logging.getLogger("subtitle_system")
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] %(message)s"
        )
        simple_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(message)s"
        )
        
        # File handler for detailed logs
        log_file = os.path.join(log_dir, f"subtitle_system_{datetime.now().strftime('%Y%m%d')}.log")
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        # Console handler for user-friendly output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        
        # Add handlers
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Error file handler
        error_file = os.path.join(log_dir, f"errors_{datetime.now().strftime('%Y%m%d')}.log")
        error_handler = logging.FileHandler(error_file, encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(error_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get the configured logger instance"""
        return self.logger
    
    def log_system_info(self):
        """Log system information"""
        import torch
        import platform
        
        self.logger.info("=" * 50)
        self.logger.info("视频字幕识别系统启动")
        self.logger.info("=" * 50)
        self.logger.info(f"系统: {platform.system()} {platform.release()}")
        self.logger.info(f"Python版本: {platform.python_version()}")
        
        if torch.cuda.is_available():
            self.logger.info(f"CUDA版本: {torch.version.cuda}")
            self.logger.info(f"PyTorch版本: {torch.__version__}")
            self.logger.info(f"GPU设备: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.logger.warning("CUDA不可用，将使用CPU模式")
    
    def log_model_info(self, model_name: str, model_size: Optional[str] = None):
        """Log model loading information"""
        self.logger.info(f"正在加载模型: {model_name}")
        if model_size:
            self.logger.info(f"模型大小: {model_size}")
    
    def log_processing_start(self, file_path: str):
        """Log start of file processing"""
        self.logger.info(f"开始处理文件: {os.path.basename(file_path)}")
    
    def log_processing_complete(self, file_path: str, duration: float):
        """Log completion of file processing"""
        self.logger.info(f"文件处理完成: {os.path.basename(file_path)} (耗时: {duration:.2f}秒)")
    
    def log_memory_usage(self):
        """Log current memory usage"""
        import torch
        import psutil
        
        # GPU memory
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
            self.logger.info(f"GPU内存使用: {gpu_memory:.2f}GB / 峰值: {gpu_memory_max:.2f}GB")
        
        # CPU memory
        cpu_memory = psutil.virtual_memory()
        self.logger.info(f"CPU内存使用: {cpu_memory.percent}% ({cpu_memory.used / 1024**3:.2f}GB / {cpu_memory.total / 1024**3:.2f}GB)")
    
    def log_error(self, error: Exception, context: str = ""):
        """Log error with context"""
        import traceback
        
        error_msg = f"错误发生"
        if context:
            error_msg += f" - {context}"
        error_msg += f": {str(error)}"
        
        self.logger.error(error_msg)
        self.logger.error(f"错误详情:\n{traceback.format_exc()}")
    
    def log_performance(self, operation: str, duration: float, file_size_mb: Optional[float] = None):
        """Log performance metrics"""
        if file_size_mb:
            speed = file_size_mb / duration if duration > 0 else 0
            self.logger.info(f"性能统计 - {operation}: {duration:.2f}秒, 处理速度: {speed:.2f}MB/s")
        else:
            self.logger.info(f"性能统计 - {operation}: {duration:.2f}秒")

# Global logger instance
_logger_instance = None

def get_logger() -> logging.Logger:
    """Get global logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = SubtitleLogger()
    return _logger_instance.get_logger()

def setup_logger(log_dir: str = "logs", log_level: int = logging.INFO) -> SubtitleLogger:
    """Setup and return logger instance"""
    global _logger_instance
    _logger_instance = SubtitleLogger(log_dir, log_level)
    return _logger_instance
