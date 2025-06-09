#!/usr/bin/env python3
"""
Video Subtitle Generation System
Main execution script for RTX 3060 Ti optimized video subtitle recognition

Usage:
    python run_subtitle_generation.py [arguments]
    
For help:
    python run_subtitle_generation.py --help
"""

import sys
import os
import signal
import atexit
from typing import Optional

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

try:
    from cli import main as cli_main
    from main import create_subtitle_system
    from utils.system_info import SystemInfo
    from config import Config
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保所有依赖已正确安装")
    sys.exit(1)

# Global system instance for cleanup
_system_instance: Optional[object] = None

def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown"""
    print(f"\n收到信号 {signum}，正在清理资源...")
    cleanup_system()
    sys.exit(0)

def cleanup_system():
    """Clean up system resources"""
    global _system_instance
    if _system_instance and hasattr(_system_instance, 'cleanup'):
        try:
            _system_instance.cleanup()
        except Exception as e:
            print(f"清理资源时出错: {e}")

def check_system_requirements():
    """Check if system meets minimum requirements"""
    print("检查系统要求...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("错误: 需要Python 3.8或更高版本")
        return False
    
    # Check dependencies
    deps = SystemInfo.check_dependencies()
    essential_deps = ['torch', 'transformers', 'faster_whisper', 'soundfile']
    missing_deps = [dep for dep in essential_deps if not deps.get(dep, False)]
    
    if missing_deps:
        print(f"错误: 缺少必需依赖: {', '.join(missing_deps)}")
        print("\n请安装缺少的依赖:")
        for dep in missing_deps:
            if dep == 'torch':
                print("  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
            elif dep == 'faster_whisper':
                print("  pip install faster-whisper")
            elif dep == 'transformers':
                print("  pip install transformers")
            elif dep == 'soundfile':
                print("  pip install soundfile")
        return False
    
    # Check FFmpeg
    if not deps.get('ffmpeg', False):
        print("警告: FFmpeg未找到")
        print("请确保FFmpeg已安装并在系统PATH中")
        print("下载地址: https://ffmpeg.org/download.html")
        
        response = input("是否继续? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            return False
    
    # Check GPU
    gpu_info = SystemInfo.detect_gpu()
    if not gpu_info["cuda_available"]:
        print("警告: CUDA不可用，将使用CPU模式（处理速度较慢）")
        response = input("是否继续? (y/N): ")
        if response.lower() not in ['y', 'yes']:
            return False
    else:
        # Check for RTX 3060 Ti compatibility
        if SystemInfo.validate_rtx_3060_ti():
            print("✓ 检测到RTX 3060 Ti兼容GPU")
        else:
            print("注意: 未检测到RTX 3060 Ti，性能可能有所不同")
    
    print("✓ 系统要求检查通过\n")
    return True

def show_welcome_message():
    """Show welcome message and system info"""
    print("=" * 60)
    print("视频字幕识别系统")
    print("针对NVIDIA RTX 3060 Ti优化的多模型语音识别系统")
    print("=" * 60)
    
    # Basic system info
    system_info = SystemInfo.get_system_info()
    gpu_info = SystemInfo.detect_gpu()
    
    print(f"系统: {system_info['platform']} {system_info['platform_release']}")
    print(f"Python: {system_info['python_version']}")
    
    if gpu_info["cuda_available"]:
        for device in gpu_info["gpu_devices"]:
            print(f"GPU: {device['name']} ({device['memory_gb']:.1f}GB)")
        print(f"CUDA: {gpu_info['cuda_version']}")
        if gpu_info.get("tensorrt_available"):
            print("TensorRT: 可用")
    else:
        print("GPU: 不可用 (将使用CPU)")
    
    print()

def interactive_mode():
    """Run in interactive mode"""
    print("交互模式")
    print("支持的操作:")
    print("1. 处理单个视频文件")
    print("2. 批量处理视频文件") 
    print("3. 查看系统信息")
    print("4. 列出可用模型")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请选择操作 (1-5): ").strip()
            
            if choice == '1':
                handle_single_file()
            elif choice == '2':
                handle_batch_files()
            elif choice == '3':
                SystemInfo.print_system_report()
            elif choice == '4':
                show_available_models()
            elif choice == '5':
                print("退出程序")
                break
            else:
                print("无效选择，请输入1-5")
                
        except KeyboardInterrupt:
            print("\n用户中断，退出程序")
            break
        except Exception as e:
            print(f"操作失败: {e}")

def handle_single_file():
    """Handle single file processing in interactive mode"""
    file_path = input("请输入视频文件路径: ").strip().strip('"')
    
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return
    
    # Model selection
    models = ['whisper-small', 'whisper-medium', 'whisper-large-v3', 'openai/whisper-large-v3']
    print("可用模型:")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    
    try:
        model_choice = int(input("选择模型 (1-4, 默认2): ").strip() or "2")
        model = models[model_choice - 1]
    except (ValueError, IndexError):
        model = "whisper-medium"
        print(f"使用默认模型: {model}")
    
    # Language selection
    language = input("语言 (zh/en/auto, 默认auto): ").strip() or "auto"
    
    # Output directory
    output_dir = input("输出目录 (默认当前目录): ").strip() or "."
    
    print(f"\n开始处理...")
    print(f"文件: {file_path}")
    print(f"模型: {model}")
    print(f"语言: {language}")
    
    try:
        global _system_instance
        with create_subtitle_system(model, language=language) as system:
            _system_instance = system
            result = system.process_video_file(file_path, output_dir, formats=['srt'])
            
            if result["success"]:
                print(f"\n✓ 处理成功!")
                print(f"输出文件: {result['output_files']['srt']}")
                print(f"处理时间: {result['processing_time']:.1f}秒")
            else:
                print(f"\n✗ 处理失败: {result['error']}")
                
    except Exception as e:
        print(f"处理过程中出错: {e}")
    finally:
        _system_instance = None

def handle_batch_files():
    """Handle batch processing in interactive mode"""
    input_dir = input("请输入视频文件目录: ").strip().strip('"')
    
    if not os.path.exists(input_dir):
        print(f"目录不存在: {input_dir}")
        return
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    video_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(os.path.join(input_dir, file))
    
    if not video_files:
        print("目录中未找到视频文件")
        return
    
    print(f"找到 {len(video_files)} 个视频文件:")
    for file in video_files[:5]:  # Show first 5
        print(f"  - {os.path.basename(file)}")
    if len(video_files) > 5:
        print(f"  ... 还有 {len(video_files) - 5} 个文件")
    
    # Model selection
    model = input("模型 (whisper-medium): ").strip() or "whisper-medium"
    language = input("语言 (zh/en/auto, 默认auto): ").strip() or "auto"
    output_dir = input("输出目录: ").strip()
    
    if not output_dir:
        print("必须指定输出目录")
        return
    
    print(f"\n开始批量处理 {len(video_files)} 个文件...")
    
    try:
        global _system_instance
        with create_subtitle_system(model, language=language) as system:
            _system_instance = system
            results = system.process_video_batch(video_files, output_dir, formats=['srt'])
            
            successful = sum(1 for r in results if r["success"])
            failed = len(results) - successful
            
            print(f"\n批量处理完成:")
            print(f"  成功: {successful} 个文件")
            print(f"  失败: {failed} 个文件")
            
    except Exception as e:
        print(f"批量处理过程中出错: {e}")
    finally:
        _system_instance = None

def show_available_models():
    """Show available models"""
    from main import list_available_models, get_model_recommendations
    
    print("可用模型:")
    models = list_available_models()
    for model in models:
        model_config = Config.get_model_config(model)
        vram_usage = model_config.get("vram_usage_mb", 0) / 1024
        print(f"  - {model} (约需 {vram_usage:.1f}GB VRAM)")
    
    print("\n中文推荐模型:")
    zh_models = get_model_recommendations("zh")
    for model in zh_models:
        print(f"  - {model}")
    
    print("\n英文推荐模型:")
    en_models = get_model_recommendations("en")
    for model in en_models:
        print(f"  - {model}")

def main():
    """Main entry point"""
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    atexit.register(cleanup_system)
    
    # Show welcome message
    show_welcome_message()
    
    # Check system requirements
    if not check_system_requirements():
        return 1
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        # Use CLI mode
        return cli_main()
    else:
        # Use interactive mode
        try:
            interactive_mode()
            return 0
        except Exception as e:
            print(f"程序异常: {e}")
            return 1

if __name__ == "__main__":
    sys.exit(main())
