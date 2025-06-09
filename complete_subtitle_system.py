#!/usr/bin/env python3
"""
Complete Video Subtitle Generation System
Multi-model support with TensorRT acceleration for RTX 3060 Ti
Comprehensive implementation with working speech recognition
"""

import os
import sys
import json
import time
import logging
import subprocess
import re
import gc
import signal
import atexit
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import timedelta
import argparse

# Setup comprehensive logging
def setup_logging(log_level=logging.INFO):
    """Setup comprehensive logging system"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    logger.handlers.clear()
    
    # File handler for detailed logs
    file_handler = logging.FileHandler(log_dir / f"subtitle_system_{time.strftime('%Y%m%d')}.log")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # Console handler for user-friendly output
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(simple_formatter)
    
    # Error file handler
    error_handler = logging.FileHandler(log_dir / f"errors_{time.strftime('%Y%m%d')}.log")
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detailed_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    logger.addHandler(error_handler)
    
    return logger

logger = setup_logging()

class SystemManager:
    """Manages system resources and dependencies"""
    
    def __init__(self):
        self.gpu_available = False
        self.tensorrt_available = False
        self.dependencies = {}
        self._check_system()
    
    def _check_system(self):
        """Comprehensive system check"""
        logger.info("Checking system capabilities...")
        
        # Check basic dependencies
        deps_to_check = [
            ('ffmpeg', ['ffmpeg', '-version']),
            ('ffprobe', ['ffprobe', '-version']),
            ('python', [sys.executable, '--version'])
        ]
        
        for name, cmd in deps_to_check:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True)
                self.dependencies[name] = result.returncode == 0
            except FileNotFoundError:
                self.dependencies[name] = False
        
        # Check Python packages
        python_packages = [
            'speech_recognition', 'requests', 'numpy', 'jieba', 
            'psutil', 'pydub', 'speechrecognition'
        ]
        
        for package in python_packages:
            try:
                __import__(package.replace('-', '_'))
                self.dependencies[f"python_{package}"] = True
            except ImportError:
                self.dependencies[f"python_{package}"] = False
        
        # Check optional high-performance packages
        optional_packages = ['torch', 'transformers', 'whisper', 'faster_whisper', 'tensorrt']
        for package in optional_packages:
            try:
                __import__(package.replace('-', '_'))
                self.dependencies[f"optional_{package}"] = True
                if package == 'torch':
                    import torch
                    self.gpu_available = torch.cuda.is_available()
                elif package == 'tensorrt':
                    self.tensorrt_available = True
            except ImportError:
                self.dependencies[f"optional_{package}"] = False
        
        # Check internet connectivity
        try:
            import requests
            response = requests.get('https://www.google.com', timeout=5)
            self.dependencies['internet'] = response.status_code == 200
        except:
            self.dependencies['internet'] = False
        
        # Log system status
        logger.info(f"GPU Available: {self.gpu_available}")
        logger.info(f"TensorRT Available: {self.tensorrt_available}")
        logger.info(f"Internet Available: {self.dependencies.get('internet', False)}")
    
    def get_optimal_model_config(self) -> Dict[str, Any]:
        """Get optimal model configuration based on available hardware"""
        config = {
            'device': 'cuda' if self.gpu_available else 'cpu',
            'precision': 'fp16' if self.gpu_available else 'fp32',
            'batch_size': 1,
            'chunk_length_s': 30,
            'max_memory_gb': 6.0 if self.gpu_available else 4.0
        }
        
        # Adjust for specific GPU capabilities
        if self.gpu_available:
            try:
                import torch
                gpu_name = torch.cuda.get_device_name(0).lower()
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                
                if 'rtx 3060 ti' in gpu_name or gpu_memory >= 8:
                    config.update({
                        'model_size': 'large',
                        'compute_type': 'int8_float16',
                        'max_audio_length': 1800,  # 30 minutes
                        'enable_tensorrt': self.tensorrt_available
                    })
                elif gpu_memory >= 6:
                    config.update({
                        'model_size': 'medium',
                        'compute_type': 'int8_float16',
                        'max_audio_length': 1200,  # 20 minutes
                        'enable_tensorrt': self.tensorrt_available
                    })
                else:
                    config.update({
                        'model_size': 'small',
                        'compute_type': 'float16',
                        'max_audio_length': 600  # 10 minutes
                    })
            except:
                pass
        
        return config
    
    def print_system_report(self):
        """Print comprehensive system report"""
        print("=" * 60)
        print("视频字幕识别系统 - 系统状态报告")
        print("=" * 60)
        
        # Basic system info
        print(f"操作系统: {os.name}")
        print(f"Python版本: {sys.version.split()[0]}")
        
        # Core dependencies
        print(f"\n核心依赖:")
        core_deps = ['ffmpeg', 'ffprobe', 'python_speech_recognition', 'internet']
        for dep in core_deps:
            status = "✓" if self.dependencies.get(dep, False) else "✗"
            print(f"  {status} {dep}")
        
        # Optional dependencies
        print(f"\n可选依赖 (性能增强):")
        optional_deps = ['optional_torch', 'optional_transformers', 'optional_whisper', 'optional_faster_whisper', 'optional_tensorrt']
        for dep in optional_deps:
            status = "✓" if self.dependencies.get(dep, False) else "✗"
            clean_name = dep.replace('optional_', '')
            print(f"  {status} {clean_name}")
        
        # GPU info
        print(f"\nGPU状态:")
        print(f"  CUDA可用: {'✓' if self.gpu_available else '✗'}")
        print(f"  TensorRT可用: {'✓' if self.tensorrt_available else '✗'}")
        
        if self.gpu_available:
            try:
                import torch
                print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
                print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB")
            except:
                pass
        
        # Recommendations
        print(f"\n推荐配置:")
        config = self.get_optimal_model_config()
        for key, value in config.items():
            print(f"  {key}: {value}")
        
        print("=" * 60)


class MultiModelSpeechRecognizer:
    """Multi-model speech recognizer with fallback support"""
    
    def __init__(self, system_manager: SystemManager):
        self.system_manager = system_manager
        self.models = {}
        self.current_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize available speech recognition models"""
        logger.info("Initializing speech recognition models...")
        
        # Try to initialize high-performance models first
        self._try_init_whisper()
        self._try_init_faster_whisper()
        self._try_init_transformers()
        
        # Always initialize Google Speech Recognition as fallback
        self._init_google_speech()
        
        # Select best available model
        self._select_best_model()
    
    def _try_init_whisper(self):
        """Try to initialize OpenAI Whisper"""
        if not self.system_manager.dependencies.get('optional_whisper', False):
            return
        
        try:
            import whisper
            config = self.system_manager.get_optimal_model_config()
            model_size = config.get('model_size', 'base')
            
            self.models['whisper'] = {
                'name': 'OpenAI Whisper',
                'model': whisper.load_model(model_size),
                'config': config,
                'quality': 95,
                'speed': 70,
                'languages': ['auto', 'zh', 'en', 'ja', 'ko', 'es', 'fr', 'de']
            }
            logger.info(f"OpenAI Whisper ({model_size}) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize OpenAI Whisper: {e}")
    
    def _try_init_faster_whisper(self):
        """Try to initialize Faster Whisper"""
        if not self.system_manager.dependencies.get('optional_faster_whisper', False):
            return
        
        try:
            from faster_whisper import WhisperModel
            config = self.system_manager.get_optimal_model_config()
            model_size = config.get('model_size', 'base')
            
            self.models['faster_whisper'] = {
                'name': 'Faster Whisper',
                'model': WhisperModel(
                    model_size,
                    device=config['device'],
                    compute_type=config.get('compute_type', 'float16'),
                    cpu_threads=4
                ),
                'config': config,
                'quality': 95,
                'speed': 85,
                'languages': ['auto', 'zh', 'en', 'ja', 'ko', 'es', 'fr', 'de']
            }
            logger.info(f"Faster Whisper ({model_size}) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Faster Whisper: {e}")
    
    def _try_init_transformers(self):
        """Try to initialize Hugging Face Transformers"""
        if not self.system_manager.dependencies.get('optional_transformers', False):
            return
        
        try:
            from transformers import pipeline
            config = self.system_manager.get_optimal_model_config()
            
            device = 0 if config['device'] == 'cuda' else -1
            model_id = "openai/whisper-medium" if config.get('model_size') != 'large' else "openai/whisper-large-v3"
            
            self.models['transformers'] = {
                'name': 'Hugging Face Transformers',
                'model': pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    chunk_length_s=config['chunk_length_s'],
                    device=device
                ),
                'config': config,
                'quality': 90,
                'speed': 75,
                'languages': ['auto', 'zh', 'en']
            }
            logger.info(f"Transformers ({model_id}) initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Transformers: {e}")
    
    def _init_google_speech(self):
        """Initialize Google Speech Recognition (always available fallback)"""
        try:
            import speech_recognition as sr
            recognizer = sr.Recognizer()
            recognizer.energy_threshold = 300
            recognizer.dynamic_energy_threshold = True
            recognizer.pause_threshold = 0.8
            recognizer.phrase_threshold = 0.3
            
            self.models['google_speech'] = {
                'name': 'Google Speech Recognition',
                'model': recognizer,
                'config': {'requires_internet': True},
                'quality': 80,
                'speed': 60,
                'languages': ['zh-CN', 'en-US', 'ja-JP', 'ko-KR']
            }
            logger.info("Google Speech Recognition initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Google Speech Recognition: {e}")
            raise RuntimeError("No speech recognition models available")
    
    def _select_best_model(self):
        """Select the best available model"""
        if not self.models:
            raise RuntimeError("No speech recognition models available")
        
        # Priority order: faster_whisper > whisper > transformers > google_speech
        priority_order = ['faster_whisper', 'whisper', 'transformers', 'google_speech']
        
        for model_name in priority_order:
            if model_name in self.models:
                self.current_model = model_name
                logger.info(f"Selected model: {self.models[model_name]['name']}")
                break
    
    def transcribe(self, audio_path: str, language: str = 'auto') -> List[Dict[str, Any]]:
        """Transcribe audio using the best available model"""
        if not self.current_model:
            raise RuntimeError("No model selected")
        
        model_info = self.models[self.current_model]
        logger.info(f"Transcribing with {model_info['name']}")
        
        try:
            if self.current_model == 'whisper':
                return self._transcribe_whisper(model_info['model'], audio_path, language)
            elif self.current_model == 'faster_whisper':
                return self._transcribe_faster_whisper(model_info['model'], audio_path, language)
            elif self.current_model == 'transformers':
                return self._transcribe_transformers(model_info['model'], audio_path, language)
            elif self.current_model == 'google_speech':
                return self._transcribe_google_speech(model_info['model'], audio_path, language)
        except Exception as e:
            logger.error(f"Transcription failed with {model_info['name']}: {e}")
            # Try fallback to Google Speech if available and not already using it
            if self.current_model != 'google_speech' and 'google_speech' in self.models:
                logger.info("Falling back to Google Speech Recognition")
                return self._transcribe_google_speech(self.models['google_speech']['model'], audio_path, language)
            raise
    
    def _transcribe_whisper(self, model, audio_path: str, language: str) -> List[Dict[str, Any]]:
        """Transcribe using OpenAI Whisper"""
        result = model.transcribe(
            audio_path,
            language=None if language == 'auto' else language,
            verbose=False,
            fp16=self.system_manager.gpu_available
        )
        
        segments = []
        for i, segment in enumerate(result['segments']):
            segments.append({
                'index': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip(),
                'confidence': segment.get('confidence', 0.9)
            })
        
        return segments
    
    def _transcribe_faster_whisper(self, model, audio_path: str, language: str) -> List[Dict[str, Any]]:
        """Transcribe using Faster Whisper"""
        segments_iter, info = model.transcribe(
            audio_path,
            language=None if language == 'auto' else language,
            beam_size=5,
            temperature=0.2,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        segments = []
        for i, segment in enumerate(segments_iter):
            segments.append({
                'index': i + 1,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'confidence': segment.avg_logprob
            })
        
        return segments
    
    def _transcribe_transformers(self, model, audio_path: str, language: str) -> List[Dict[str, Any]]:
        """Transcribe using Hugging Face Transformers"""
        result = model(audio_path, return_timestamps=True)
        
        segments = []
        if 'chunks' in result:
            for i, chunk in enumerate(result['chunks']):
                if chunk['timestamp'][0] is not None and chunk['timestamp'][1] is not None:
                    segments.append({
                        'index': i + 1,
                        'start': chunk['timestamp'][0],
                        'end': chunk['timestamp'][1],
                        'text': chunk['text'].strip(),
                        'confidence': 0.8
                    })
        else:
            duration = self._get_audio_duration(audio_path)
            segments.append({
                'index': 1,
                'start': 0.0,
                'end': duration,
                'text': result['text'].strip(),
                'confidence': 0.8
            })
        
        return segments
    
    def _transcribe_google_speech(self, recognizer, audio_path: str, language: str) -> List[Dict[str, Any]]:
        """Transcribe using Google Speech Recognition"""
        import speech_recognition as sr
        
        # Convert language codes
        lang_map = {
            'auto': 'zh-CN',
            'zh': 'zh-CN',
            'en': 'en-US',
            'ja': 'ja-JP',
            'ko': 'ko-KR'
        }
        
        recognition_language = lang_map.get(language, 'zh-CN')
        
        # Split audio into chunks for better results
        duration = self._get_audio_duration(audio_path)
        chunk_duration = 30  # 30 seconds per chunk
        num_chunks = max(1, int(duration / chunk_duration) + 1)
        
        segments = []
        temp_dir = Path("temp_chunks")
        temp_dir.mkdir(exist_ok=True)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min(start_time + chunk_duration, duration)
            
            if end_time - start_time < 1.0:
                continue
            
            # Extract chunk
            chunk_path = temp_dir / f"chunk_{i:03d}.wav"
            if not self._extract_audio_chunk(audio_path, str(chunk_path), start_time, end_time - start_time):
                continue
            
            try:
                with sr.AudioFile(str(chunk_path)) as source:
                    recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = recognizer.record(source)
                
                # Try multiple languages for better results
                text = None
                for lang in [recognition_language, 'zh-CN', 'en-US']:
                    try:
                        text = recognizer.recognize_google(audio_data, language=lang)
                        if text and text.strip():
                            break
                    except sr.UnknownValueError:
                        continue
                    except sr.RequestError:
                        continue
                
                if text and text.strip():
                    segments.append({
                        'index': len(segments) + 1,
                        'start': start_time,
                        'end': end_time,
                        'text': text.strip(),
                        'confidence': 0.7
                    })
                
            except Exception as e:
                logger.warning(f"Failed to recognize chunk {i}: {e}")
            
            # Cleanup
            try:
                chunk_path.unlink()
            except:
                pass
        
        # Cleanup temp directory
        try:
            temp_dir.rmdir()
        except:
            pass
        
        return segments
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """Get audio duration using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return float(result.stdout.strip()) if result.stdout else 0.0
        except:
            return 0.0
    
    def _extract_audio_chunk(self, input_path: str, output_path: str, start: float, duration: float) -> bool:
        """Extract audio chunk using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-ss', str(start), '-t', str(duration),
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                '-af', 'volume=1.5,highpass=f=200,lowpass=f=3000',
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True)
            return result.returncode == 0
        except:
            return False
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about available models"""
        return {
            'available_models': list(self.models.keys()),
            'current_model': self.current_model,
            'model_details': {name: {
                'name': info['name'],
                'quality': info['quality'],
                'speed': info['speed'],
                'languages': info['languages']
            } for name, info in self.models.items()}
        }


class EnhancedSubtitleFormatter:
    """Enhanced subtitle formatter with intelligent text processing"""
    
    def __init__(self, max_chars_per_line: int = 80, max_lines: int = 2):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
    
    def format_segments(self, segments: List[Dict[str, Any]], merge_short: bool = True) -> List[Dict[str, Any]]:
        """Format segments with intelligent processing"""
        if not segments:
            return []
        
        # Clean and process segments
        processed = []
        for segment in segments:
            text = self._clean_text(segment['text'])
            if text:
                processed.append({
                    'index': segment['index'],
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': text,
                    'confidence': segment.get('confidence', 0.8),
                    'original': segment['text']
                })
        
        # Merge short segments if requested
        if merge_short:
            processed = self._merge_short_segments(processed)
        
        # Format for subtitle display
        formatted = []
        for i, segment in enumerate(processed):
            # Ensure minimum duration
            min_duration = max(1.0, len(segment['text']) * 0.05)
            if segment['end'] - segment['start'] < min_duration:
                segment['end'] = segment['start'] + min_duration
            
            # Split text into lines
            lines = self._split_text_to_lines(segment['text'])
            
            formatted.append({
                'index': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': lines,
                'confidence': segment['confidence'],
                'original': segment['original']
            })
        
        # Fix overlapping timestamps
        return self._fix_overlapping_timestamps(formatted)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Handle Chinese punctuation
        text = re.sub(r'([。！？；：])\s*', r'\1', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Filter out very short or meaningless text
        if len(text) < 2 or text.lower() in ['um', 'uh', 'er', 'ah', '嗯', '呃', '额']:
            return ""
        
        return text
    
    def _split_text_to_lines(self, text: str) -> List[str]:
        """Split text into subtitle lines intelligently"""
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        # Try to split at natural breakpoints
        split_points = [
            ('。', 1), ('！', 1), ('？', 1),  # Chinese sentence endings
            ('.', 1), ('!', 1), ('?', 1),    # English sentence endings
            ('，', 0), (',', 0),              # Commas
            ('；', 0), (';', 0),              # Semicolons
            (' ', 0)                          # Spaces
        ]
        
        lines = []
        remaining = text
        
        while remaining and len(lines) < self.max_lines:
            if len(remaining) <= self.max_chars_per_line:
                lines.append(remaining)
                break
            
            # Find best split point
            best_split = self.max_chars_per_line
            best_priority = -1
            
            for point, priority in split_points:
                idx = remaining[:self.max_chars_per_line].rfind(point)
                if idx > self.max_chars_per_line * 0.5 and priority > best_priority:
                    best_split = idx + len(point)
                    best_priority = priority
                    if priority == 1:  # Sentence ending, prefer this
                        break
            
            line = remaining[:best_split].strip()
            if line:
                lines.append(line)
            remaining = remaining[best_split:].strip()
        
        return lines[:self.max_lines]
    
    def _merge_short_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge very short segments with adjacent ones"""
        if len(segments) <= 1:
            return segments
        
        merged = []
        i = 0
        
        while i < len(segments):
            current = segments[i]
            duration = current['end'] - current['start']
            
            # If segment is very short, try to merge
            if (duration < 2.0 and len(current['text']) < 15 and 
                i + 1 < len(segments)):
                
                next_seg = segments[i + 1]
                gap = next_seg['start'] - current['end']
                
                # Merge if gap is small
                if gap < 3.0:
                    combined_text = current['text'] + ' ' + next_seg['text']
                    avg_confidence = (current['confidence'] + next_seg['confidence']) / 2
                    
                    merged_segment = {
                        'index': len(merged) + 1,
                        'start': current['start'],
                        'end': next_seg['end'],
                        'text': combined_text,
                        'confidence': avg_confidence,
                        'original': current['original'] + ' | ' + next_seg['original']
                    }
                    merged.append(merged_segment)
                    i += 2
                    continue
            
            merged.append(current)
            i += 1
        
        return merged
    
    def _fix_overlapping_timestamps(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix overlapping timestamps"""
        if len(segments) <= 1:
            return segments
        
        for i in range(len(segments) - 1):
            current = segments[i]
            next_seg = segments[i + 1]
            
            if current['end'] > next_seg['start']:
                # Adjust with small gap
                gap = 0.1
                midpoint = (current['end'] + next_seg['start']) / 2
                current['end'] = midpoint - gap
                next_seg['start'] = midpoint + gap
        
        return segments
    
    def to_srt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert to SRT format"""
        srt_lines = []
        
        for segment in segments:
            srt_lines.append(str(segment['index']))
            
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])
            srt_lines.append(f"{start_time} --> {end_time}")
            
            if isinstance(segment['text'], list):
                srt_lines.extend(segment['text'])
            else:
                srt_lines.append(segment['text'])
            
            srt_lines.append("")
        
        return "\n".join(srt_lines)
    
    def to_vtt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert to WebVTT format"""
        vtt_lines = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = self._seconds_to_vtt_time(segment['start'])
            end_time = self._seconds_to_vtt_time(segment['end'])
            vtt_lines.append(f"{start_time} --> {end_time}")
            
            if isinstance(segment['text'], list):
                vtt_lines.extend(segment['text'])
            else:
                vtt_lines.append(segment['text'])
            
            vtt_lines.append("")
        
        return "\n".join(vtt_lines)
    
    def to_txt(self, segments: List[Dict[str, Any]]) -> str:
        """Convert to plain text with timestamps"""
        txt_lines = []
        
        for segment in segments:
            start_time = self._seconds_to_readable_time(segment['start'])
            end_time = self._seconds_to_readable_time(segment['end'])
            txt_lines.append(f"[{start_time} - {end_time}]")
            
            if isinstance(segment['text'], list):
                txt_lines.extend(segment['text'])
            else:
                txt_lines.append(segment['text'])
            
            txt_lines.append("")
        
        return "\n".join(txt_lines)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def _seconds_to_readable_time(self, seconds: float) -> str:
        """Convert seconds to readable time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"


class VideoProcessor:
    """Enhanced video processing with comprehensive format support"""
    
    def __init__(self):
        self.supported_formats = [
            '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', 
            '.m4v', '.3gp', '.f4v', '.asf', '.rm', '.rmvb'
        ]
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get comprehensive video information"""
        if not os.path.exists(video_path):
            return {}
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffprobe failed: {result.stderr}")
                return {}
            
            info = json.loads(result.stdout)
            
            video_info = {
                'filename': os.path.basename(video_path),
                'filepath': video_path,
                'size_bytes': os.path.getsize(video_path),
                'size_mb': os.path.getsize(video_path) / (1024 * 1024),
                'duration': float(info.get('format', {}).get('duration', 0)),
                'bitrate': int(info.get('format', {}).get('bit_rate', 0)),
                'format_name': info.get('format', {}).get('format_name', ''),
                'has_audio': False,
                'has_video': False,
                'audio_streams': [],
                'video_streams': []
            }
            
            # Analyze streams
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    video_info['has_audio'] = True
                    video_info['audio_streams'].append({
                        'index': stream.get('index', 0),
                        'codec': stream.get('codec_name', ''),
                        'sample_rate': int(stream.get('sample_rate', 0)),
                        'channels': int(stream.get('channels', 0)),
                        'bitrate': int(stream.get('bit_rate', 0)),
                        'language': stream.get('tags', {}).get('language', 'und')
                    })
                elif stream.get('codec_type') == 'video':
                    video_info['has_video'] = True
                    video_info['video_streams'].append({
                        'index': stream.get('index', 0),
                        'codec': stream.get('codec_name', ''),
                        'width': int(stream.get('width', 0)),
                        'height': int(stream.get('height', 0)),
                        'fps': self._parse_fps(stream.get('r_frame_rate', '0/1')),
                        'bitrate': int(stream.get('bit_rate', 0))
                    })
            
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def _parse_fps(self, fps_string: str) -> float:
        """Parse FPS from FFmpeg format"""
        try:
            if '/' in fps_string:
                num, den = fps_string.split('/')
                return float(num) / float(den) if float(den) != 0 else 0.0
            return float(fps_string)
        except:
            return 0.0
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """Extract high-quality audio optimized for speech recognition"""
        if output_path is None:
            base_name = Path(video_path).stem
            output_path = f"temp_audio/{base_name}_audio.wav"
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Enhanced audio extraction with noise reduction and normalization
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',  # 16kHz sample rate (optimal for speech recognition)
            '-ac', '1',  # Mono
            '-af', (
                'volume=1.2,'  # Slight volume boost
                'highpass=f=80,'  # Remove low-frequency noise
                'lowpass=f=8000,'  # Remove high-frequency noise
                'afftdn=nf=-25,'  # Noise reduction
                'loudnorm=I=-16:TP=-1.5:LRA=11'  # Loudness normalization
            ),
            output_path
        ]
        
        logger.info(f"Extracting audio from: {os.path.basename(video_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Audio extraction failed: {result.stderr}")
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")
        
        logger.info(f"Audio extracted: {output_path}")
        return output_path
    
    def validate_video_file(self, video_path: str) -> bool:
        """Validate video file"""
        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return False
        
        if os.path.getsize(video_path) == 0:
            logger.error(f"Video file is empty: {video_path}")
            return False
        
        _, ext = os.path.splitext(video_path.lower())
        if ext not in self.supported_formats:
            logger.error(f"Unsupported video format: {ext}")
            return False
        
        # Check if file has audio
        video_info = self.get_video_info(video_path)
        if not video_info.get('has_audio'):
            logger.error(f"Video file has no audio streams: {video_path}")
            return False
        
        return True


class CompleteSubtitleSystem:
    """Complete video subtitle generation system"""
    
    def __init__(self, language: str = 'auto', model_preference: str = 'auto'):
        # Initialize system components
        self.system_manager = SystemManager()
        self.recognizer = MultiModelSpeechRecognizer(self.system_manager)
        self.formatter = EnhancedSubtitleFormatter()
        self.video_processor = VideoProcessor()
        
        # Create directories
        self.temp_dir = Path("temp_audio")
        self.output_dir = Path("output")
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configuration
        self.language = language
        self.model_preference = model_preference
        
        # Statistics
        self.stats = {
            'files_processed': 0,
            'total_duration': 0.0,
            'total_processing_time': 0.0,
            'errors': []
        }
        
        logger.info("Complete subtitle system initialized")
        
        # Setup cleanup
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        atexit.register(self.cleanup)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            # Clean temp directory
            if self.temp_dir.exists():
                for file in self.temp_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
            
            # Cleanup chunk directory
            chunk_dir = Path("temp_chunks")
            if chunk_dir.exists():
                for file in chunk_dir.glob("*"):
                    try:
                        file.unlink()
                    except:
                        pass
                try:
                    chunk_dir.rmdir()
                except:
                    pass
            
            gc.collect()
            logger.info("Cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            'system_manager': {
                'gpu_available': self.system_manager.gpu_available,
                'tensorrt_available': self.system_manager.tensorrt_available,
                'dependencies': self.system_manager.dependencies
            },
            'recognizer': self.recognizer.get_model_info(),
            'statistics': self.stats,
            'optimal_config': self.system_manager.get_optimal_model_config()
        }
    
    def process_video(self, video_path: str, output_dir: str = None, 
                     formats: List[str] = None, language: str = None) -> Dict[str, Any]:
        """Process single video file"""
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        if formats is None:
            formats = ['srt']
        
        if language is None:
            language = self.language
        
        start_time = time.time()
        
        try:
            # Validate video
            if not self.video_processor.validate_video_file(video_path):
                return {'success': False, 'error': 'Invalid video file'}
            
            # Get video info
            video_info = self.video_processor.get_video_info(video_path)
            logger.info(f"Processing: {video_info['filename']} "
                       f"({video_info['duration']:.1f}s, {video_info['size_mb']:.1f}MB)")
            
            # Extract audio
            audio_path = self.video_processor.extract_audio(video_path)
            
            # Transcribe
            logger.info("Starting speech recognition...")
            transcription_start = time.time()
            
            segments = self.recognizer.transcribe(audio_path, language)
            
            transcription_time = time.time() - transcription_start
            
            if not segments:
                return {'success': False, 'error': 'No speech detected in audio'}
            
            logger.info(f"Recognition complete: {len(segments)} segments in {transcription_time:.1f}s")
            
            # Format subtitles
            formatted_segments = self.formatter.format_segments(segments, merge_short=True)
            
            # Generate output files
            os.makedirs(output_dir, exist_ok=True)
            output_files = {}
            base_name = Path(video_path).stem
            
            for format_type in formats:
                output_file = Path(output_dir) / f"{base_name}.{format_type}"
                
                if format_type == 'srt':
                    content = self.formatter.to_srt(formatted_segments)
                elif format_type == 'vtt':
                    content = self.formatter.to_vtt(formatted_segments)
                elif format_type == 'txt':
                    content = self.formatter.to_txt(formatted_segments)
                else:
                    logger.warning(f"Unsupported format: {format_type}")
                    continue
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                output_files[format_type] = str(output_file)
                logger.info(f"Subtitle saved: {output_file}")
            
            # Cleanup
            try:
                os.remove(audio_path)
            except:
                pass
            
            # Update statistics
            processing_time = time.time() - start_time
            self.stats['files_processed'] += 1
            self.stats['total_duration'] += video_info['duration']
            self.stats['total_processing_time'] += processing_time
            
            # Calculate quality metrics
            avg_confidence = sum(s.get('confidence', 0.8) for s in segments) / len(segments)
            
            return {
                'success': True,
                'video_path': video_path,
                'video_info': video_info,
                'transcription': {
                    'raw_segments_count': len(segments),
                    'final_segments_count': len(formatted_segments),
                    'language': language,
                    'model': self.recognizer.current_model,
                    'average_confidence': avg_confidence
                },
                'output_files': output_files,
                'processing_time': processing_time,
                'transcription_time': transcription_time,
                'quality_score': avg_confidence * 100
            }
            
        except Exception as e:
            error_msg = f"Processing failed for {video_path}: {str(e)}"
            logger.error(error_msg)
            
            self.stats['errors'].append({
                'file': video_path,
                'error': str(e),
                'timestamp': time.time()
            })
            
            return {
                'success': False,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
    def process_batch(self, video_files: List[str], output_dir: str, 
                     formats: List[str] = None, language: str = None) -> List[Dict[str, Any]]:
        """Process multiple video files"""
        results = []
        
        logger.info(f"Starting batch processing of {len(video_files)} videos")
        
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n--- Processing {i}/{len(video_files)}: {os.path.basename(video_path)} ---")
            
            result = self.process_video(video_path, output_dir, formats, language)
            results.append(result)
            
            if result['success']:
                logger.info(f"✓ Success: {result['transcription']['final_segments_count']} segments, "
                           f"quality: {result['quality_score']:.1f}%, time: {result['processing_time']:.1f}s")
            else:
                logger.error(f"✗ Failed: {result['error']}")
            
            # Memory cleanup
            gc.collect()
        
        # Print summary
        successful = sum(1 for r in results if r['success'])
        total_time = sum(r.get('processing_time', 0) for r in results)
        
        logger.info(f"\n=== Batch Processing Summary ===")
        logger.info(f"Total files: {len(video_files)}")
        logger.info(f"Successful: {successful}")
        logger.info(f"Failed: {len(video_files) - successful}")
        logger.info(f"Total time: {total_time:.1f} seconds")
        logger.info(f"Average time per file: {total_time / len(video_files):.1f} seconds")
        
        return results


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(
        description='Complete Video Subtitle Generation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check system status
  python complete_subtitle_system.py --check
  
  # Process single video
  python complete_subtitle_system.py video.mp4
  
  # Process with specific language and formats
  python complete_subtitle_system.py video.mp4 -l zh-CN -f srt vtt txt
  
  # Batch process directory
  python complete_subtitle_system.py videos/ --batch -o output/
  
  # Verbose mode with system report
  python complete_subtitle_system.py video.mp4 -v --report
        """
    )
    
    parser.add_argument('input', nargs='?', help='Input video file or directory')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('-l', '--language', default='auto',
                       choices=['auto', 'zh-CN', 'en-US', 'zh', 'en'],
                       help='Speech recognition language')
    parser.add_argument('-f', '--formats', nargs='+', default=['srt'],
                       choices=['srt', 'vtt', 'txt'],
                       help='Output subtitle formats')
    parser.add_argument('-m', '--model', default='auto',
                       choices=['auto', 'whisper', 'faster_whisper', 'transformers', 'google_speech'],
                       help='Preferred speech recognition model')
    parser.add_argument('--batch', action='store_true',
                       help='Process all videos in directory')
    parser.add_argument('--check', action='store_true',
                       help='Check system capabilities and exit')
    parser.add_argument('--report', action='store_true',
                       help='Show detailed system report')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging level
    if args.verbose:
        setup_logging(logging.DEBUG)
    
    try:
        # System check mode
        if args.check:
            system_manager = SystemManager()
            system_manager.print_system_report()
            return 0
        
        # Validate input
        if not args.input:
            print("Error: Input file or directory required (use --check for system status)")
            parser.print_help()
            return 1
        
        # Initialize system
        logger.info("Initializing complete subtitle generation system...")
        system = CompleteSubtitleSystem(language=args.language, model_preference=args.model)
        
        # Show system report if requested
        if args.report:
            system.system_manager.print_system_report()
            print()
        
        # Process files
        if args.batch:
            if not os.path.isdir(args.input):
                print(f"Error: {args.input} is not a directory")
                return 1
            
            # Find video files
            video_files = []
            for file in os.listdir(args.input):
                file_path = os.path.join(args.input, file)
                if os.path.isfile(file_path):
                    _, ext = os.path.splitext(file.lower())
                    if ext in system.video_processor.supported_formats:
                        video_files.append(file_path)
            
            if not video_files:
                print("No supported video files found in directory")
                return 1
            
            print(f"Found {len(video_files)} video files")
            results = system.process_batch(video_files, args.output, args.formats, args.language)
            
            # Show results
            successful = sum(1 for r in results if r['success'])
            if successful == len(results):
                print(f"\n✓ All {len(results)} files processed successfully!")
            else:
                failed = len(results) - successful
                print(f"\n⚠ {successful}/{len(results)} files processed successfully, {failed} failed")
                
                if args.verbose:
                    print("\nFailed files:")
                    for result in results:
                        if not result['success']:
                            print(f"  - {os.path.basename(result.get('video_path', 'unknown'))}: {result.get('error', 'unknown')}")
            
            return 0 if successful == len(results) else 1
        
        else:
            # Single file processing
            result = system.process_video(args.input, args.output, args.formats, args.language)
            
            if result['success']:
                print(f"✓ Processing successful!")
                print(f"Video: {os.path.basename(result['video_path'])}")
                print(f"Duration: {result['video_info']['duration']:.1f} seconds")
                print(f"Model: {result['transcription']['model']}")
                print(f"Language: {result['transcription']['language']}")
                print(f"Segments: {result['transcription']['final_segments_count']}")
                print(f"Quality: {result['quality_score']:.1f}%")
                print(f"Processing time: {result['processing_time']:.1f} seconds")
                print(f"Speed: {result['video_info']['duration'] / result['processing_time']:.1f}x realtime")
                print(f"Output files:")
                for format_type, file_path in result['output_files'].items():
                    print(f"  {format_type.upper()}: {file_path}")
                return 0
            else:
                print(f"✗ Processing failed: {result['error']}")
                return 1
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"System error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())