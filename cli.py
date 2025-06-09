"""
Command Line Interface for video subtitle recognition system
"""

import argparse
import os
import sys
import json
from typing import List, Optional, Dict, Any

from main import (
    create_subtitle_system, 
    list_available_models, 
    get_model_recommendations,
    validate_model_compatibility
)
from utils.system_info import SystemInfo
from config import Config

class SubtitleCLI:
    """Command Line Interface for subtitle generation"""
    
    def __init__(self):
        self.parser = self._create_parser()
    
    def _create_parser(self) -> argparse.ArgumentParser:
        """Create argument parser"""
        parser = argparse.ArgumentParser(
            description="视频字幕识别系统 - 支持多模型GPU加速",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
示例用法:
  # 处理单个视频文件
  python cli.py process video.mp4
  
  # 使用大型模型处理中文视频
  python cli.py process video.mp4 --model whisper-large-v3 --language zh
  
  # 批量处理视频文件
  python cli.py batch *.mp4 --output-dir ./subtitles
  
  # 生成多种格式字幕
  python cli.py process video.mp4 --formats srt vtt txt
  
  # 查看系统信息
  python cli.py info
  
  # 列出可用模型
  python cli.py models
            """
        )
        
        subparsers = parser.add_subparsers(dest='command', help='可用命令')
        
        # Process single file command
        process_parser = subparsers.add_parser('process', help='处理单个视频文件')
        self._add_process_args(process_parser)
        
        # Batch processing command
        batch_parser = subparsers.add_parser('batch', help='批量处理视频文件')
        self._add_batch_args(batch_parser)
        
        # System info command
        info_parser = subparsers.add_parser('info', help='显示系统信息')
        info_parser.add_argument('--detailed', action='store_true', help='显示详细信息')
        
        # Models command
        models_parser = subparsers.add_parser('models', help='列出可用模型')
        models_parser.add_argument('--language', default='auto', 
                                 help='显示特定语言的推荐模型')
        models_parser.add_argument('--check-compatibility', metavar='MODEL',
                                 help='检查模型兼容性')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='测试系统功能')
        test_parser.add_argument('--model', default='whisper-small', help='测试使用的模型')
        test_parser.add_argument('--duration', type=int, default=30, help='测试视频时长（秒）')
        
        return parser
    
    def _add_process_args(self, parser: argparse.ArgumentParser):
        """Add arguments for single file processing"""
        parser.add_argument('input_file', help='输入视频文件路径')
        parser.add_argument('--model', default='whisper-medium', 
                          choices=list_available_models(),
                          help='使用的语音识别模型')
        parser.add_argument('--language', default='auto',
                          choices=['auto', 'zh', 'en'],
                          help='音频语言 (auto=自动检测)')
        parser.add_argument('--output-dir', '-o', help='输出目录')
        parser.add_argument('--formats', nargs='+', default=['srt'],
                          choices=['srt', 'vtt', 'txt'],
                          help='输出字幕格式')
        parser.add_argument('--beam-size', type=int, default=5,
                          help='解码beam大小')
        parser.add_argument('--temperature', type=float, default=0.2,
                          help='解码温度参数')
        parser.add_argument('--max-duration', type=int, default=1800,
                          help='音频块最大时长（秒）')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='详细输出')
    
    def _add_batch_args(self, parser: argparse.ArgumentParser):
        """Add arguments for batch processing"""
        parser.add_argument('input_files', nargs='+', help='输入视频文件路径（支持通配符）')
        parser.add_argument('--model', default='whisper-medium',
                          choices=list_available_models(),
                          help='使用的语音识别模型')
        parser.add_argument('--language', default='auto',
                          choices=['auto', 'zh', 'en'],
                          help='音频语言')
        parser.add_argument('--output-dir', '-o', required=True,
                          help='输出目录')
        parser.add_argument('--formats', nargs='+', default=['srt'],
                          choices=['srt', 'vtt', 'txt'],
                          help='输出字幕格式')
        parser.add_argument('--continue-on-error', action='store_true',
                          help='遇到错误时继续处理其他文件')
        parser.add_argument('--beam-size', type=int, default=5,
                          help='解码beam大小')
        parser.add_argument('--temperature', type=float, default=0.2,
                          help='解码温度参数')
        parser.add_argument('--max-duration', type=int, default=1800,
                          help='音频块最大时长（秒）')
        parser.add_argument('--verbose', '-v', action='store_true',
                          help='详细输出')
        parser.add_argument('--parallel', type=int, default=1,
                          help='并行处理文件数（实验性）')
    
    def run(self, args: Optional[List[str]] = None) -> int:
        """Run CLI with given arguments"""
        parsed_args = self.parser.parse_args(args)
        
        if not parsed_args.command:
            self.parser.print_help()
            return 0
        
        try:
            if parsed_args.command == 'process':
                return self._handle_process(parsed_args)
            elif parsed_args.command == 'batch':
                return self._handle_batch(parsed_args)
            elif parsed_args.command == 'info':
                return self._handle_info(parsed_args)
            elif parsed_args.command == 'models':
                return self._handle_models(parsed_args)
            elif parsed_args.command == 'test':
                return self._handle_test(parsed_args)
            else:
                print(f"未知命令: {parsed_args.command}")
                return 1
                
        except KeyboardInterrupt:
            print("\n用户中断操作")
            return 1
        except Exception as e:
            print(f"错误: {e}")
            if hasattr(parsed_args, 'verbose') and parsed_args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    def _handle_process(self, args) -> int:
        """Handle single file processing"""
        if not os.path.exists(args.input_file):
            print(f"错误: 文件不存在 - {args.input_file}")
            return 1
        
        print(f"处理视频文件: {args.input_file}")
        print(f"使用模型: {args.model}")
        print(f"语言: {args.language}")
        print(f"输出格式: {', '.join(args.formats)}")
        
        # Check model compatibility
        compatibility = validate_model_compatibility(args.model)
        if not compatibility["compatible"]:
            print("错误: 模型与当前系统不兼容")
            for warning in compatibility["warnings"]:
                print(f"  - {warning}")
            return 1
        
        if compatibility["warnings"]:
            print("警告:")
            for warning in compatibility["warnings"]:
                print(f"  - {warning}")
            
            response = input("是否继续? (y/N): ")
            if response.lower() not in ['y', 'yes']:
                return 0
        
        # Create system
        kwargs = {
            "beam_size": args.beam_size,
            "temperature": args.temperature,
            "max_duration": args.max_duration,
            "log_level": 10 if args.verbose else 20  # DEBUG vs INFO
        }
        
        with create_subtitle_system(args.model, language=args.language, **kwargs) as system:
            result = system.process_video_file(
                args.input_file,
                output_dir=args.output_dir,
                formats=args.formats
            )
            
            if result["success"]:
                print(f"\n✓ 处理成功!")
                print(f"  处理时间: {result['processing_time']:.1f}秒")
                print(f"  转录时间: {result['transcription_time']:.1f}秒")
                print(f"  识别语言: {result['transcription']['language']}")
                print(f"  字幕片段: {result['subtitles']['count']} 个")
                print(f"  输出文件:")
                for format_type, file_path in result["output_files"].items():
                    print(f"    {format_type.upper()}: {file_path}")
                
                # Performance report
                if args.verbose:
                    report = system.get_performance_report()
                    print(f"\n性能报告:")
                    for key, value in report.items():
                        if key != "recent_errors":
                            print(f"  {key}: {value}")
                
                return 0
            else:
                print(f"\n✗ 处理失败: {result['error']}")
                return 1
    
    def _handle_batch(self, args) -> int:
        """Handle batch processing"""
        # Expand file patterns
        import glob
        input_files = []
        for pattern in args.input_files:
            expanded = glob.glob(pattern)
            if expanded:
                input_files.extend(expanded)
            else:
                print(f"警告: 未找到匹配文件 - {pattern}")
        
        if not input_files:
            print("错误: 未找到任何输入文件")
            return 1
        
        print(f"批量处理 {len(input_files)} 个文件:")
        for file in input_files:
            print(f"  - {file}")
        
        print(f"\n使用模型: {args.model}")
        print(f"输出目录: {args.output_dir}")
        print(f"输出格式: {', '.join(args.formats)}")
        
        # Check model compatibility
        compatibility = validate_model_compatibility(args.model)
        if not compatibility["compatible"]:
            print("错误: 模型与当前系统不兼容")
            return 1
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Create system
        kwargs = {
            "beam_size": args.beam_size,
            "temperature": args.temperature,
            "max_duration": args.max_duration,
            "log_level": 10 if args.verbose else 20
        }
        
        with create_subtitle_system(args.model, language=args.language, **kwargs) as system:
            results = system.process_video_batch(
                input_files,
                output_dir=args.output_dir,
                formats=args.formats,
                continue_on_error=args.continue_on_error
            )
            
            # Print summary
            successful = sum(1 for r in results if r["success"])
            failed = len(results) - successful
            
            print(f"\n批量处理完成:")
            print(f"  成功: {successful} 个文件")
            print(f"  失败: {failed} 个文件")
            
            if failed > 0:
                print(f"\n失败的文件:")
                for result in results:
                    if not result["success"]:
                        print(f"  - {result['video_path']}: {result['error']}")
            
            # Performance report
            if args.verbose:
                report = system.get_performance_report()
                print(f"\n性能报告:")
                for key, value in report.items():
                    if key != "recent_errors":
                        print(f"  {key}: {value}")
            
            return 0 if failed == 0 else 1
    
    def _handle_info(self, args) -> int:
        """Handle system info display"""
        if args.detailed:
            SystemInfo.print_system_report()
        else:
            # Basic system info
            system_info = SystemInfo.get_system_info()
            gpu_info = SystemInfo.detect_gpu()
            deps = SystemInfo.check_dependencies()
            
            print("系统信息:")
            print(f"  操作系统: {system_info['platform']} {system_info['platform_release']}")
            print(f"  Python版本: {system_info['python_version']}")
            print(f"  CPU核心: {system_info['cpu_count']}")
            print(f"  内存: {system_info['memory'].total / (1024**3):.1f}GB")
            
            print(f"\nGPU信息:")
            print(f"  CUDA可用: {gpu_info['cuda_available']}")
            if gpu_info['cuda_available']:
                for device in gpu_info['gpu_devices']:
                    print(f"  GPU: {device['name']} ({device['memory_gb']:.1f}GB)")
                print(f"  CUDA版本: {gpu_info['cuda_version']}")
                print(f"  TensorRT可用: {gpu_info.get('tensorrt_available', False)}")
            
            # Dependencies status
            essential_deps = ['torch', 'transformers', 'faster_whisper', 'soundfile', 'ffmpeg']
            missing_essential = [dep for dep in essential_deps if not deps.get(dep, False)]
            
            if missing_essential:
                print(f"\n⚠️  缺少必需依赖: {', '.join(missing_essential)}")
            else:
                print(f"\n✓ 所有必需依赖已安装")
        
        return 0
    
    def _handle_models(self, args) -> int:
        """Handle models listing and checking"""
        if args.check_compatibility:
            # Check specific model compatibility
            model_name = args.check_compatibility
            if model_name not in list_available_models():
                print(f"错误: 未知模型 - {model_name}")
                print(f"可用模型: {', '.join(list_available_models())}")
                return 1
            
            compatibility = validate_model_compatibility(model_name)
            print(f"模型兼容性检查: {model_name}")
            print(f"  兼容: {'是' if compatibility['compatible'] else '否'}")
            
            if compatibility["requirements"]:
                print(f"  系统要求:")
                for req, value in compatibility["requirements"].items():
                    print(f"    {req}: {value}")
            
            if compatibility["warnings"]:
                print(f"  警告:")
                for warning in compatibility["warnings"]:
                    print(f"    - {warning}")
            
            return 0 if compatibility["compatible"] else 1
        
        else:
            # List available models
            print("可用模型:")
            models = list_available_models()
            for model in models:
                model_config = Config.get_model_config(model)
                vram_usage = model_config.get("vram_usage_mb", 0) / 1024
                print(f"  - {model} (约需 {vram_usage:.1f}GB VRAM)")
            
            # Show recommendations for language
            if args.language != 'auto':
                recommendations = get_model_recommendations(args.language)
                print(f"\n{args.language} 语言推荐模型:")
                for model in recommendations:
                    print(f"  - {model}")
        
        return 0
    
    def _handle_test(self, args) -> int:
        """Handle system testing"""
        print(f"测试系统功能...")
        print(f"测试模型: {args.model}")
        
        try:
            # Create test system
            kwargs = {"log_level": 20}  # INFO level
            with create_subtitle_system(args.model, **kwargs) as system:
                status = system.get_system_status()
                
                print(f"\n系统状态:")
                print(f"  GPU可用: {status['system']['gpu_available']}")
                print(f"  TensorRT可用: {status['system']['tensorrt_available']}")
                print(f"  模型设备: {status['model']['device']}")
                
                if status['memory'].get('gpu'):
                    gpu_mem = status['memory']['gpu']
                    print(f"  GPU内存: {gpu_mem['allocated_gb']:.2f}GB / {gpu_mem['reserved_gb']:.2f}GB")
                
                print(f"\n✓ 系统测试通过")
                return 0
                
        except Exception as e:
            print(f"\n✗ 系统测试失败: {e}")
            return 1

def main():
    """Main CLI entry point"""
    cli = SubtitleCLI()
    return cli.run()

if __name__ == "__main__":
    sys.exit(main())
