#!/usr/bin/env python3
"""
Video Subtitle Generation System - Main Entry Point
Comprehensive multi-model video-to-subtitle conversion system
"""

import sys
import os
import argparse
from pathlib import Path

def main():
    """Main entry point with comprehensive help and routing"""
    
    parser = argparse.ArgumentParser(
        description='Video Subtitle Generation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Commands:
  
  System Information:
    --check         Check system capabilities and dependencies
    --demo          Run interactive demonstration
    --install-guide Show installation guide for dependencies
    
  Processing:
    video.mp4       Process single video file
    videos/         Process directory (use --batch)
    
  Examples:
    %(prog)s --check
    %(prog)s --demo
    %(prog)s video.mp4
    %(prog)s video.mp4 -l zh-CN -f srt vtt
    %(prog)s videos/ --batch -o output/
    %(prog)s --install-guide

Supported Features:
  ✓ Video audio extraction with FFmpeg
  ✓ Speech recognition (Google Speech API)
  ✓ Voice activity detection
  ✓ Multiple subtitle formats (SRT, VTT)
  ✓ Batch processing
  ✓ Chinese and English language support
  ✓ GPU acceleration ready (with optional dependencies)
        """
    )
    
    # Main arguments
    parser.add_argument('input', nargs='?', 
                       help='Input video file or directory')
    parser.add_argument('-o', '--output', default='output',
                       help='Output directory (default: output)')
    parser.add_argument('-l', '--language', default='zh-CN',
                       choices=['zh-CN', 'en-US', 'auto'],
                       help='Speech recognition language (default: zh-CN)')
    parser.add_argument('-f', '--formats', nargs='+', default=['srt'],
                       choices=['srt', 'vtt'],
                       help='Output subtitle formats (default: srt)')
    
    # Processing options
    parser.add_argument('--batch', action='store_true',
                       help='Process all videos in directory')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    # System commands
    parser.add_argument('--check', action='store_true',
                       help='Check system capabilities')
    parser.add_argument('--demo', action='store_true',
                       help='Run interactive demonstration')
    parser.add_argument('--install-guide', action='store_true',
                       help='Show installation guide')
    
    args = parser.parse_args()
    
    # Handle system commands
    if args.check:
        return run_system_check()
    elif args.demo:
        return run_demo()
    elif args.install_guide:
        return show_install_guide()
    
    # Validate input for processing commands
    if not args.input:
        print("Error: Input file or directory required")
        print("Use --check for system status, --demo for demonstration, or --help for usage")
        return 1
    
    # Route to appropriate processing system
    return run_processing(args)

def run_system_check():
    """Run comprehensive system check"""
    try:
        from working_subtitle_system import VideoSubtitleGenerator
        
        print("=" * 60)
        print("视频字幕识别系统 - 系统检查")
        print("Video Subtitle System - System Check")
        print("=" * 60)
        
        generator = VideoSubtitleGenerator()
        status = generator.check_system()
        
        print("\n核心组件状态 / Core Components:")
        core_components = ['ffmpeg', 'ffprobe', 'speech_recognition', 'internet']
        for component in core_components:
            if component in status:
                status_text = "✓ 可用" if status[component] else "✗ 不可用"
                print(f"  {component}: {status_text}")
        
        # Check if system is ready
        ready = all(status.get(comp, False) for comp in ['ffmpeg', 'speech_recognition'])
        
        print(f"\n系统状态 / System Status:")
        if ready:
            print("✓ 系统已准备就绪，可以处理视频文件")
            print("✓ System ready for video processing")
        else:
            print("✗ 系统未准备就绪，请检查依赖")
            print("✗ System not ready, please check dependencies")
            print("\n建议运行: python run_subtitle_generation.py --install-guide")
            print("Recommended: python run_subtitle_generation.py --install-guide")
        
        # Show usage examples
        if ready:
            print(f"\n使用示例 / Usage Examples:")
            print(f"  python run_subtitle_generation.py video.mp4")
            print(f"  python run_subtitle_generation.py videos/ --batch")
            print(f"  python run_subtitle_generation.py --demo")
        
        return 0 if ready else 1
        
    except Exception as e:
        print(f"系统检查失败 / System check failed: {e}")
        return 1

def run_demo():
    """Run interactive demonstration"""
    try:
        from demo_subtitle_system import demo_system_capabilities
        demo_system_capabilities()
        return 0
    except Exception as e:
        print(f"演示运行失败 / Demo failed: {e}")
        return 1

def show_install_guide():
    """Show installation guide"""
    try:
        from demo_subtitle_system import show_installation_guide
        show_installation_guide()
        return 0
    except Exception as e:
        print(f"无法显示安装指南 / Cannot show install guide: {e}")
        return 1

def run_processing(args):
    """Run video processing with the working system"""
    try:
        from working_subtitle_system import VideoSubtitleGenerator
        import logging
        
        # Setup logging level
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)
        
        # Initialize system
        generator = VideoSubtitleGenerator()
        
        # Check system readiness
        status = generator.check_system()
        if not status.get('ffmpeg') or not status.get('speech_recognition'):
            print("错误: 系统未准备就绪")
            print("Error: System not ready")
            print("运行 --check 检查系统状态")
            print("Run --check to verify system status")
            return 1
        
        # Process files
        if args.batch:
            return process_batch(generator, args)
        else:
            return process_single(generator, args)
            
    except Exception as e:
        print(f"处理失败 / Processing failed: {e}")
        return 1

def process_single(generator, args):
    """Process single video file"""
    if not os.path.exists(args.input):
        print(f"错误: 文件不存在 / Error: File not found: {args.input}")
        return 1
    
    print(f"处理视频 / Processing video: {os.path.basename(args.input)}")
    
    try:
        result = generator.process_video(
            args.input,
            output_dir=args.output,
            formats=args.formats,
            language=args.language
        )
        
        if result['success']:
            print(f"✓ 处理成功 / Processing successful!")
            print(f"  输出目录 / Output directory: {result['output_directory']}")
            
            if 'output_files' in result:
                print(f"  生成文件 / Generated files:")
                for file_path in result['output_files']:
                    print(f"    {os.path.basename(file_path)}")
            
            if 'processing_time' in result:
                print(f"  处理时间 / Processing time: {result['processing_time']:.1f}s")
            
            return 0
        else:
            print(f"✗ 处理失败 / Processing failed: {result.get('error', 'Unknown error')}")
            return 1
            
    except Exception as e:
        print(f"✗ 处理过程中出错 / Error during processing: {e}")
        return 1

def process_batch(generator, args):
    """Process multiple video files"""
    if not os.path.isdir(args.input):
        print(f"错误: 不是目录 / Error: Not a directory: {args.input}")
        return 1
    
    # Find video files
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
    video_files = []
    
    for file in os.listdir(args.input):
        file_path = os.path.join(args.input, file)
        if os.path.isfile(file_path):
            _, ext = os.path.splitext(file.lower())
            if ext in video_extensions:
                video_files.append(file_path)
    
    if not video_files:
        print(f"错误: 目录中未找到视频文件 / Error: No video files found in directory")
        return 1
    
    print(f"找到 {len(video_files)} 个视频文件 / Found {len(video_files)} video files")
    
    try:
        results = generator.process_batch(
            video_files,
            output_dir=args.output,
            formats=args.formats,
            language=args.language
        )
        
        # Count results
        successful = sum(1 for r in results if r['success'])
        failed = len(results) - successful
        
        print(f"\n批量处理完成 / Batch processing complete:")
        print(f"  成功: {successful} / Successful: {successful}")
        print(f"  失败: {failed} / Failed: {failed}")
        
        if args.verbose and failed > 0:
            print(f"\n失败文件 / Failed files:")
            for result in results:
                if not result['success']:
                    filename = os.path.basename(result.get('video_path', 'unknown'))
                    error = result.get('error', 'unknown error')
                    print(f"  {filename}: {error}")
        
        return 0 if failed == 0 else 1
        
    except Exception as e:
        print(f"✗ 批量处理过程中出错 / Error during batch processing: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())