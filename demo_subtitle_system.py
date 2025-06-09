#!/usr/bin/env python3
"""
Demo Video Subtitle System
Simple demonstration of the working subtitle generation capabilities
"""

import os
import sys
import time
from pathlib import Path

def create_demo_video():
    """Create a simple demo video with audio for testing"""
    demo_file = "demo_video.mp4"
    
    if os.path.exists(demo_file):
        print(f"Demo video already exists: {demo_file}")
        return demo_file
    
    # Create a simple 10-second video with synthesized speech
    try:
        import subprocess
        
        # Generate a 10-second video with a sine wave tone and visual pattern
        cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', 'testsrc2=duration=10:size=640x480:rate=30',
            '-f', 'lavfi', '-i', 'sine=frequency=440:duration=10',
            '-c:v', 'libx264', '-c:a', 'aac',
            '-shortest', demo_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"Created demo video: {demo_file}")
            return demo_file
        else:
            print(f"Failed to create demo video: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Error creating demo video: {e}")
        return None

def demo_system_capabilities():
    """Demonstrate the subtitle system capabilities"""
    
    print("=" * 60)
    print("视频字幕识别系统 - 演示")
    print("Video Subtitle Recognition System - Demo")
    print("=" * 60)
    
    # Import and test the working system
    try:
        from working_subtitle_system import VideoSubtitleGenerator
        
        print("\n1. 初始化系统 / Initializing System...")
        generator = VideoSubtitleGenerator()
        
        print("\n2. 系统状态检查 / System Status Check:")
        status = generator.check_system()
        for component, available in status.items():
            status_text = "✓ 可用" if available else "✗ 不可用"
            print(f"   {component}: {status_text}")
        
        # Check if core components are available
        if not status.get('ffmpeg') or not status.get('speech_recognition'):
            print("\n⚠️ 警告: 缺少核心组件，无法完整演示")
            print("Warning: Missing core components, cannot run full demo")
            return
        
        print("\n3. 支持的功能 / Supported Features:")
        print("   ✓ 视频音频提取 / Video Audio Extraction")
        print("   ✓ 语音识别 / Speech Recognition")
        print("   ✓ 字幕格式化 / Subtitle Formatting")
        print("   ✓ 多种输出格式 / Multiple Output Formats (SRT, VTT)")
        print("   ✓ 批量处理 / Batch Processing")
        print("   ✓ 语音活动检测 / Voice Activity Detection")
        
        print("\n4. 模型信息 / Model Information:")
        print("   主要模型: Google Speech Recognition")
        print("   Primary Model: Google Speech Recognition")
        print("   备用模型: FFmpeg语音活动检测")
        print("   Fallback: FFmpeg Voice Activity Detection")
        
        print("\n5. 处理流程 / Processing Pipeline:")
        print("   输入视频 → 音频提取 → 语音识别 → 字幕格式化 → 输出文件")
        print("   Video Input → Audio Extract → Speech Recognition → Subtitle Format → Output Files")
        
        # Create and test with demo video
        print("\n6. 创建演示视频 / Creating Demo Video...")
        demo_video = create_demo_video()
        
        if demo_video and os.path.exists(demo_video):
            print(f"\n7. 演示处理 / Demo Processing...")
            print(f"   处理文件: {demo_video}")
            
            # Get video info
            video_info = generator.get_video_info(demo_video)
            if video_info:
                print(f"   视频时长: {video_info.get('duration', 0):.1f} 秒")
                print(f"   文件大小: {video_info.get('size_mb', 0):.1f} MB")
                print(f"   音频流: {len(video_info.get('audio_streams', []))} 个")
            
            # Note: We won't actually process since it requires internet for Google Speech API
            print("\n   注意: 完整处理需要网络连接用于语音识别")
            print("   Note: Full processing requires internet connection for speech recognition")
        
        print("\n8. 使用示例 / Usage Examples:")
        print("   单文件处理:")
        print("   python working_subtitle_system.py video.mp4")
        print("   ")
        print("   批量处理:")
        print("   python working_subtitle_system.py videos/ --batch -o output/")
        print("   ")
        print("   指定语言:")
        print("   python working_subtitle_system.py video.mp4 -l zh-CN")
        
        print("\n9. 输出格式 / Output Formats:")
        print("   ✓ SRT (SubRip) - 通用字幕格式")
        print("   ✓ VTT (WebVTT) - Web视频字幕格式")
        
        print("\n" + "=" * 60)
        print("演示完成! / Demo Complete!")
        print("系统已准备就绪，可以处理视频文件")
        print("System is ready to process video files")
        print("=" * 60)
        
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保所有依赖已正确安装")
    except Exception as e:
        print(f"演示过程中出错: {e}")

def show_installation_guide():
    """Show installation guide for missing dependencies"""
    print("\n" + "=" * 60)
    print("安装指南 / Installation Guide")
    print("=" * 60)
    
    print("\n必需依赖 / Required Dependencies:")
    print("1. FFmpeg")
    print("   Windows: https://ffmpeg.org/download.html")
    print("   Linux: sudo apt install ffmpeg")
    print("   macOS: brew install ffmpeg")
    
    print("\n2. Python包 / Python Packages:")
    print("   pip install speechrecognition")
    print("   pip install pydub")
    print("   pip install requests")
    print("   pip install numpy")
    print("   pip install psutil")
    print("   pip install jieba")
    
    print("\n可选依赖 (性能增强) / Optional Dependencies (Performance Enhancement):")
    print("   pip install torch torchaudio")
    print("   pip install transformers")
    print("   pip install openai-whisper")
    print("   pip install faster-whisper")
    
    print("\nGPU加速 (NVIDIA) / GPU Acceleration (NVIDIA):")
    print("   安装CUDA Toolkit 12.1+")
    print("   Install CUDA Toolkit 12.1+")
    print("   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--install-guide":
        show_installation_guide()
    else:
        demo_system_capabilities()