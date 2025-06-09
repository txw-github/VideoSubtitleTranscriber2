"""
System information and GPU detection utilities
"""

import os
import platform
import subprocess
import psutil
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class SystemInfo:
    """System information collector and GPU detector"""
    
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.architecture(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(),
            "cpu_freq": psutil.cpu_freq(),
            "memory": psutil.virtual_memory(),
            "disk": psutil.disk_usage('/') if platform.system() != "Windows" else psutil.disk_usage('C:'),
        }
        return info
    
    @staticmethod
    def detect_gpu() -> Dict[str, Any]:
        """Detect GPU information and CUDA availability"""
        gpu_info = {
            "cuda_available": False,
            "gpu_count": 0,
            "gpu_devices": [],
            "cuda_version": None,
            "driver_version": None,
            "tensorrt_available": False
        }
        
        try:
            import torch
            gpu_info["cuda_available"] = torch.cuda.is_available()
            gpu_info["gpu_count"] = torch.cuda.device_count()
            gpu_info["cuda_version"] = torch.version.cuda
            
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    device_props = torch.cuda.get_device_properties(i)
                    gpu_info["gpu_devices"].append({
                        "index": i,
                        "name": device_props.name,
                        "total_memory": device_props.total_memory,
                        "memory_gb": device_props.total_memory / (1024**3),
                        "major": device_props.major,
                        "minor": device_props.minor,
                        "multi_processor_count": device_props.multi_processor_count
                    })
        except ImportError:
            logger.warning("PyTorch not installed, cannot detect CUDA")
        
        # Check TensorRT availability
        try:
            import tensorrt as trt
            gpu_info["tensorrt_available"] = True
            gpu_info["tensorrt_version"] = trt.__version__
        except ImportError:
            logger.info("TensorRT not available")
        
        # Get NVIDIA driver version
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            if result.returncode == 0:
                gpu_info["driver_version"] = result.stdout.strip().split('\n')[0]
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, cannot get driver version")
        
        return gpu_info
    
    @staticmethod
    def check_dependencies() -> Dict[str, bool]:
        """Check if required dependencies are installed"""
        dependencies = {}
        
        # Check Python packages
        packages = [
            "torch", "torchaudio", "transformers", "faster_whisper",
            "soundfile", "librosa", "numpy", "jieba", "psutil",
            "ffmpeg-python", "pyloudnorm"
        ]
        
        for package in packages:
            try:
                __import__(package.replace("-", "_"))
                dependencies[package] = True
            except ImportError:
                dependencies[package] = False
        
        # Check optional packages
        optional_packages = ["tensorrt", "pycuda"]
        for package in optional_packages:
            try:
                __import__(package)
                dependencies[f"{package}_optional"] = True
            except ImportError:
                dependencies[f"{package}_optional"] = False
        
        # Check system tools
        system_tools = ["ffmpeg", "ffprobe"]
        for tool in system_tools:
            try:
                result = subprocess.run([tool, "-version"], capture_output=True)
                dependencies[tool] = result.returncode == 0
            except FileNotFoundError:
                dependencies[tool] = False
        
        return dependencies
    
    @staticmethod
    def get_optimal_settings(gpu_memory_gb: float) -> Dict[str, Any]:
        """Get optimal settings based on available GPU memory"""
        if gpu_memory_gb >= 6:
            # RTX 3060 Ti or better
            return {
                "batch_size": 1,
                "max_audio_length": 1800,  # 30 minutes
                "chunk_length": 30,
                "recommended_models": ["whisper-large-v3", "openai/whisper-large-v3"],
                "tensorrt_enabled": True,
                "fp16_enabled": True
            }
        elif gpu_memory_gb >= 4:
            # RTX 3060 or similar
            return {
                "batch_size": 1,
                "max_audio_length": 1200,  # 20 minutes
                "chunk_length": 30,
                "recommended_models": ["whisper-medium", "openai/whisper-medium"],
                "tensorrt_enabled": True,
                "fp16_enabled": True
            }
        elif gpu_memory_gb >= 2:
            # Lower-end GPU
            return {
                "batch_size": 1,
                "max_audio_length": 600,  # 10 minutes
                "chunk_length": 20,
                "recommended_models": ["whisper-small"],
                "tensorrt_enabled": False,
                "fp16_enabled": True
            }
        else:
            # CPU fallback
            return {
                "batch_size": 1,
                "max_audio_length": 300,  # 5 minutes
                "chunk_length": 15,
                "recommended_models": ["whisper-small"],
                "tensorrt_enabled": False,
                "fp16_enabled": False,
                "device": "cpu"
            }
    
    @staticmethod
    def validate_rtx_3060_ti() -> bool:
        """Validate if the system has RTX 3060 Ti or compatible GPU"""
        gpu_info = SystemInfo.detect_gpu()
        
        if not gpu_info["cuda_available"]:
            return False
        
        for device in gpu_info["gpu_devices"]:
            # Check for RTX 3060 Ti or better
            gpu_name = device["name"].lower()
            memory_gb = device["memory_gb"]
            
            # RTX 3060 Ti has 8GB but effective ~6GB for inference
            if ("rtx 3060 ti" in gpu_name or 
                ("rtx" in gpu_name and memory_gb >= 6) or
                ("geforce" in gpu_name and memory_gb >= 6)):
                return True
        
        return False
    
    @staticmethod
    def print_system_report():
        """Print comprehensive system report"""
        print("=" * 60)
        print("视频字幕识别系统 - 系统信息报告")
        print("=" * 60)
        
        # System info
        sys_info = SystemInfo.get_system_info()
        print(f"操作系统: {sys_info['platform']} {sys_info['platform_release']}")
        print(f"架构: {sys_info['architecture'][0]}")
        print(f"处理器: {sys_info['processor']}")
        print(f"CPU核心数: {sys_info['cpu_count']}")
        print(f"内存: {sys_info['memory'].total / (1024**3):.1f}GB")
        
        # GPU info
        gpu_info = SystemInfo.detect_gpu()
        print(f"\nGPU信息:")
        print(f"CUDA可用: {gpu_info['cuda_available']}")
        if gpu_info['cuda_available']:
            print(f"CUDA版本: {gpu_info['cuda_version']}")
            print(f"驱动版本: {gpu_info['driver_version']}")
            print(f"GPU数量: {gpu_info['gpu_count']}")
            
            for device in gpu_info['gpu_devices']:
                print(f"  设备 {device['index']}: {device['name']}")
                print(f"    显存: {device['memory_gb']:.1f}GB")
                print(f"    计算能力: {device['major']}.{device['minor']}")
        
        print(f"TensorRT可用: {gpu_info['tensorrt_available']}")
        
        # Dependencies
        deps = SystemInfo.check_dependencies()
        print(f"\n依赖检查:")
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            print(f"  {status} {dep}")
        
        # RTX 3060 Ti validation
        rtx_compatible = SystemInfo.validate_rtx_3060_ti()
        print(f"\nRTX 3060 Ti兼容性: {'✓' if rtx_compatible else '✗'}")
        
        # Optimal settings
        if gpu_info['gpu_devices']:
            memory_gb = max(device['memory_gb'] for device in gpu_info['gpu_devices'])
            settings = SystemInfo.get_optimal_settings(memory_gb)
            print(f"\n推荐设置:")
            for key, value in settings.items():
                print(f"  {key}: {value}")
        
        print("=" * 60)
