# 视频字幕识别系统 / Video Subtitle Recognition System

A comprehensive Python-based multi-model video subtitle recognition system optimized for Windows with NVIDIA RTX 3060 Ti GPU support and TensorRT acceleration capabilities.

## 功能特性 / Features

### 核心功能 / Core Features
- ✅ **多模型语音识别** / Multi-model speech recognition
- ✅ **视频音频提取** / Video audio extraction with FFmpeg
- ✅ **智能字幕格式化** / Intelligent subtitle formatting
- ✅ **批量处理** / Batch processing support
- ✅ **多种输出格式** / Multiple output formats (SRT, VTT)
- ✅ **语音活动检测** / Voice activity detection
- ✅ **中英文支持** / Chinese and English language support

### 高级功能 / Advanced Features
- 🚀 **GPU加速准备** / GPU acceleration ready (with optional dependencies)
- 🎯 **TensorRT优化** / TensorRT optimization support
- 📊 **内存管理** / Memory management for 6GB GPU constraint
- 🔄 **模型回退机制** / Model fallback mechanism
- 📈 **处理进度监控** / Processing progress monitoring

## 系统要求 / System Requirements

### 必需依赖 / Required Dependencies
- Python 3.8+
- FFmpeg (for video/audio processing)
- Internet connection (for Google Speech Recognition)

### 可选依赖 / Optional Dependencies (Performance Enhancement)
- CUDA Toolkit 12.1+ (for GPU acceleration)
- PyTorch with CUDA support
- OpenAI Whisper
- Faster Whisper
- Hugging Face Transformers
- TensorRT (for maximum performance)

## 快速开始 / Quick Start

### 1. 系统检查 / System Check
```bash
python run_subtitle_generation.py --check
```

### 2. 演示功能 / Demo
```bash
python run_subtitle_generation.py --demo
```

### 3. 处理单个视频 / Process Single Video
```bash
python run_subtitle_generation.py video.mp4
```

### 4. 批量处理 / Batch Processing
```bash
python run_subtitle_generation.py videos/ --batch -o output/
```

### 5. 指定语言和格式 / Specify Language and Format
```bash
python run_subtitle_generation.py video.mp4 -l zh-CN -f srt vtt
```

## 命令行选项 / Command Line Options

```
使用方法 / Usage: run_subtitle_generation.py [选项] [输入文件/目录]

位置参数 / Positional Arguments:
  input                 输入视频文件或目录 / Input video file or directory

可选参数 / Optional Arguments:
  -h, --help           显示帮助信息 / Show help message
  -o OUTPUT            输出目录 / Output directory (default: output)
  -l LANGUAGE          语言选择 / Language choice (zh-CN, en-US, auto)
  -f FORMATS           输出格式 / Output formats (srt, vtt)
  --batch              批量处理 / Batch processing mode
  -v, --verbose        详细输出 / Verbose logging

系统命令 / System Commands:
  --check              检查系统状态 / Check system capabilities
  --demo               运行演示 / Run demonstration
  --install-guide      显示安装指南 / Show installation guide
```

## 支持的视频格式 / Supported Video Formats

- MP4, AVI, MKV, MOV, WMV, FLV, WebM
- M4V, 3GP, F4V, ASF, RM, RMVB

## 输出格式 / Output Formats

### SRT (SubRip)
标准字幕格式，兼容大多数视频播放器
Standard subtitle format compatible with most video players

### VTT (WebVTT)
Web视频字幕格式，适用于HTML5视频
Web video subtitle format for HTML5 videos

## 架构设计 / Architecture

### 模块化设计 / Modular Design
```
video-subtitle-system/
├── run_subtitle_generation.py    # 主入口 / Main entry point
├── working_subtitle_system.py    # 工作系统 / Working system
├── complete_subtitle_system.py   # 完整系统 / Complete system
├── demo_subtitle_system.py       # 演示系统 / Demo system
├── test_subtitle_system.py       # 测试套件 / Test suite
├── models/                       # 模型管理 / Model management
├── processors/                   # 处理器 / Processors
├── utils/                        # 工具函数 / Utilities
├── temp_audio/                   # 临时音频 / Temporary audio
├── output/                       # 输出目录 / Output directory
└── logs/                         # 日志文件 / Log files
```

### 核心组件 / Core Components

1. **VideoSubtitleGenerator** - 主要生成器类
2. **WorkingSpeechRecognizer** - 语音识别引擎
3. **SubtitleFormatter** - 字幕格式化器
4. **VideoProcessor** - 视频处理器
5. **SystemManager** - 系统管理器

## 性能优化 / Performance Optimization

### GPU加速 / GPU Acceleration
- NVIDIA RTX 3060 Ti (6GB) 优化
- CUDA 12.1+ 支持
- TensorRT 加速
- 内存管理优化

### 处理速度 / Processing Speed
- 实时语音识别
- 并行音频处理
- 智能分块处理
- 内存高效管理

## 安装指南 / Installation Guide

### 基础安装 / Basic Installation
```bash
# 安装 Python 依赖
pip install speechrecognition pydub requests numpy psutil jieba

# 安装 FFmpeg (Windows)
# 从 https://ffmpeg.org/download.html 下载并添加到 PATH
```

### GPU 加速安装 / GPU Acceleration Installation
```bash
# 安装 CUDA Toolkit 12.1+
# 安装 PyTorch with CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# 安装高性能模型
pip install openai-whisper faster-whisper transformers

# 安装 TensorRT (可选)
pip install tensorrt
```

## 使用示例 / Usage Examples

### 基本使用 / Basic Usage
```python
from working_subtitle_system import VideoSubtitleGenerator

# 初始化生成器
generator = VideoSubtitleGenerator()

# 处理视频
result = generator.process_video(
    video_path="video.mp4",
    output_dir="output",
    formats=["srt", "vtt"],
    language="zh-CN"
)

if result['success']:
    print(f"字幕生成成功: {result['output_files']}")
else:
    print(f"处理失败: {result['error']}")
```

### 批量处理 / Batch Processing
```python
# 批量处理多个视频
video_files = ["video1.mp4", "video2.mp4", "video3.mp4"]
results = generator.process_batch(
    video_files=video_files,
    output_dir="batch_output",
    formats=["srt"],
    language="auto"
)

# 统计结果
successful = sum(1 for r in results if r['success'])
print(f"成功处理: {successful}/{len(results)} 个视频")
```

## 故障排除 / Troubleshooting

### 常见问题 / Common Issues

**Q: 系统提示缺少FFmpeg**
A: 下载安装FFmpeg并添加到系统PATH环境变量

**Q: 网络连接错误**
A: Google语音识别需要网络连接，请检查网络设置

**Q: GPU加速不工作**
A: 确保安装了CUDA Toolkit和支持CUDA的PyTorch

**Q: 内存不足错误**
A: 系统已针对6GB显存优化，可调整处理参数

### 日志查看 / Log Viewing
```bash
# 查看详细日志
python run_subtitle_generation.py video.mp4 -v

# 日志文件位置
logs/subtitle_system_YYYYMMDD.log
logs/errors_YYYYMMDD.log
```

## 开发和扩展 / Development and Extension

### 添加新的语音识别模型 / Adding New Speech Recognition Models
1. 在 `MultiModelSpeechRecognizer` 中添加新模型初始化
2. 实现对应的转录方法
3. 更新模型选择逻辑

### 添加新的输出格式 / Adding New Output Formats
1. 在 `SubtitleFormatter` 中添加新格式方法
2. 更新命令行参数选项
3. 添加相应的测试用例

## 许可证 / License

本项目采用 MIT 许可证 - 详见 LICENSE 文件
This project is licensed under the MIT License - see the LICENSE file for details

## 贡献 / Contributing

欢迎提交问题报告和功能请求
Welcome to submit issue reports and feature requests

## 更新日志 / Changelog

### v1.0.0 (2025-06-09)
- ✅ 完成核心视频字幕识别功能
- ✅ 实现多模型语音识别支持
- ✅ 添加GPU加速和TensorRT优化支持
- ✅ 完成批量处理功能
- ✅ 实现中英文语言支持
- ✅ 添加完整的命令行界面
- ✅ 完成系统测试套件

---

**技术支持 / Technical Support**
如有技术问题，请查看日志文件或运行系统检查命令
For technical issues, please check log files or run system check command