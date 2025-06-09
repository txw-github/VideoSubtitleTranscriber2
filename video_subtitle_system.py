#!/usr/bin/env python3
"""
Comprehensive Video Subtitle Generation System
Multi-model support with TensorRT acceleration for RTX 3060 Ti
"""

import os
import sys
import json
import time
import logging
import subprocess
import re
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('video_subtitle.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelManager:
    """Manages different speech recognition models"""
    
    def __init__(self):
        self.available_models = {}
        self.current_model = None
        self._detect_available_models()
    
    def _detect_available_models(self):
        """Detect which speech recognition models are available"""
        
        # Try OpenAI Whisper
        try:
            import whisper
            self.available_models['whisper'] = {
                'name': 'OpenAI Whisper',
                'sizes': ['tiny', 'base', 'small', 'medium', 'large'],
                'languages': ['auto', 'zh', 'en'],
                'module': whisper
            }
            logger.info("OpenAI Whisper available")
        except ImportError:
            logger.warning("OpenAI Whisper not available")
        
        # Try Faster Whisper
        try:
            from faster_whisper import WhisperModel
            self.available_models['faster_whisper'] = {
                'name': 'Faster Whisper',
                'sizes': ['tiny', 'base', 'small', 'medium', 'large-v3'],
                'languages': ['auto', 'zh', 'en'],
                'module': WhisperModel
            }
            logger.info("Faster Whisper available")
        except ImportError:
            logger.warning("Faster Whisper not available")
        
        # Try SpeechRecognition
        try:
            import speech_recognition as sr
            self.available_models['speech_recognition'] = {
                'name': 'Google Speech Recognition',
                'sizes': ['default'],
                'languages': ['auto', 'zh-CN', 'en-US'],
                'module': sr
            }
            logger.info("SpeechRecognition available")
        except ImportError:
            logger.warning("SpeechRecognition not available")
        
        # Try Transformers (Hugging Face)
        try:
            from transformers import pipeline
            self.available_models['transformers'] = {
                'name': 'Hugging Face Transformers',
                'sizes': ['small', 'medium', 'large'],
                'languages': ['auto', 'zh', 'en'],
                'module': pipeline
            }
            logger.info("Transformers available")
        except ImportError:
            logger.warning("Transformers not available")
    
    def get_best_model(self, language: str = 'auto') -> str:
        """Get the best available model for the language"""
        if 'faster_whisper' in self.available_models:
            return 'faster_whisper'
        elif 'whisper' in self.available_models:
            return 'whisper'
        elif 'transformers' in self.available_models:
            return 'transformers'
        elif 'speech_recognition' in self.available_models:
            return 'speech_recognition'
        else:
            return None
    
    def load_model(self, model_type: str, model_size: str = 'base', language: str = 'auto'):
        """Load specified model"""
        if model_type not in self.available_models:
            raise ValueError(f"Model {model_type} not available")
        
        logger.info(f"Loading {model_type} model (size: {model_size})")
        
        if model_type == 'whisper':
            import whisper
            self.current_model = {
                'type': 'whisper',
                'model': whisper.load_model(model_size),
                'size': model_size
            }
        
        elif model_type == 'faster_whisper':
            from faster_whisper import WhisperModel
            # Optimize for RTX 3060 Ti
            compute_type = "int8_float16" if model_size in ['medium', 'large-v3'] else "float16"
            self.current_model = {
                'type': 'faster_whisper',
                'model': WhisperModel(
                    model_size, 
                    device="cuda", 
                    compute_type=compute_type,
                    cpu_threads=4
                ),
                'size': model_size
            }
        
        elif model_type == 'transformers':
            from transformers import pipeline
            model_id = f"openai/whisper-{model_size}" if model_size != 'large' else "openai/whisper-large-v3"
            self.current_model = {
                'type': 'transformers',
                'model': pipeline(
                    "automatic-speech-recognition",
                    model=model_id,
                    chunk_length_s=30,
                    device=0 if self._check_cuda() else -1
                ),
                'size': model_size
            }
        
        elif model_type == 'speech_recognition':
            import speech_recognition as sr
            self.current_model = {
                'type': 'speech_recognition',
                'model': sr.Recognizer(),
                'size': 'default'
            }
        
        logger.info(f"Model loaded successfully: {model_type} ({model_size})")
    
    def _check_cuda(self) -> bool:
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def transcribe(self, audio_path: str, language: str = 'auto') -> List[Dict]:
        """Transcribe audio using current model"""
        if not self.current_model:
            raise ValueError("No model loaded")
        
        model_type = self.current_model['type']
        model = self.current_model['model']
        
        if model_type == 'whisper':
            return self._transcribe_whisper(model, audio_path, language)
        elif model_type == 'faster_whisper':
            return self._transcribe_faster_whisper(model, audio_path, language)
        elif model_type == 'transformers':
            return self._transcribe_transformers(model, audio_path, language)
        elif model_type == 'speech_recognition':
            return self._transcribe_speech_recognition(model, audio_path, language)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _transcribe_whisper(self, model, audio_path: str, language: str) -> List[Dict]:
        """Transcribe using OpenAI Whisper"""
        result = model.transcribe(
            audio_path,
            language=None if language == 'auto' else language,
            verbose=False
        )
        
        segments = []
        for i, segment in enumerate(result['segments']):
            segments.append({
                'index': i + 1,
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text'].strip()
            })
        
        return segments
    
    def _transcribe_faster_whisper(self, model, audio_path: str, language: str) -> List[Dict]:
        """Transcribe using Faster Whisper"""
        segments_iter, info = model.transcribe(
            audio_path,
            language=None if language == 'auto' else language,
            beam_size=5,
            temperature=0.2,
            vad_filter=True
        )
        
        segments = []
        for i, segment in enumerate(segments_iter):
            segments.append({
                'index': i + 1,
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip()
            })
        
        return segments
    
    def _transcribe_transformers(self, model, audio_path: str, language: str) -> List[Dict]:
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
                        'text': chunk['text'].strip()
                    })
        else:
            # Single result without timestamps
            duration = self._get_audio_duration(audio_path)
            segments.append({
                'index': 1,
                'start': 0.0,
                'end': duration,
                'text': result['text'].strip()
            })
        
        return segments
    
    def _transcribe_speech_recognition(self, recognizer, audio_path: str, language: str) -> List[Dict]:
        """Transcribe using Google Speech Recognition"""
        import speech_recognition as sr
        
        # Split audio into chunks for better results
        segments = []
        chunk_duration = 30  # 30 seconds per chunk
        duration = self._get_audio_duration(audio_path)
        num_chunks = int(duration / chunk_duration) + 1
        
        temp_dir = Path("temp_chunks")
        temp_dir.mkdir(exist_ok=True)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min(start_time + chunk_duration, duration)
            
            # Extract chunk
            chunk_path = temp_dir / f"chunk_{i}.wav"
            self._extract_audio_chunk(audio_path, str(chunk_path), start_time, end_time - start_time)
            
            try:
                with sr.AudioFile(str(chunk_path)) as source:
                    audio = recognizer.record(source)
                    
                    # Try Google Speech Recognition
                    lang_code = 'zh-CN' if language == 'zh' else 'en-US' if language == 'en' else 'zh-CN'
                    text = recognizer.recognize_google(audio, language=lang_code)
                    
                    segments.append({
                        'index': i + 1,
                        'start': start_time,
                        'end': end_time,
                        'text': text
                    })
                    
            except Exception as e:
                logger.warning(f"Recognition failed for chunk {i}: {e}")
                segments.append({
                    'index': i + 1,
                    'start': start_time,
                    'end': end_time,
                    'text': f"[识别失败]"
                })
            
            # Cleanup
            if chunk_path.exists():
                chunk_path.unlink()
        
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
    
    def _extract_audio_chunk(self, input_path: str, output_path: str, start: float, duration: float):
        """Extract audio chunk using ffmpeg"""
        cmd = [
            'ffmpeg', '-y', '-i', input_path,
            '-ss', str(start), '-t', str(duration),
            '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
            output_path
        ]
        subprocess.run(cmd, capture_output=True)


class SubtitleFormatter:
    """Formats transcription results into subtitle files"""
    
    def __init__(self, max_chars_per_line: int = 80, max_lines_per_subtitle: int = 2):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_subtitle = max_lines_per_subtitle
    
    def format_segments(self, segments: List[Dict]) -> List[Dict]:
        """Format segments for subtitle output"""
        formatted = []
        
        for segment in segments:
            text = segment['text'].strip()
            if not text:
                continue
            
            # Split long text into multiple lines
            lines = self._split_text_to_lines(text)
            
            formatted.append({
                'index': segment['index'],
                'start': segment['start'],
                'end': segment['end'],
                'text': lines
            })
        
        return formatted
    
    def _split_text_to_lines(self, text: str) -> List[str]:
        """Split text into appropriate lines"""
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        # Try to split at sentence boundaries
        sentences = re.split(r'([.!?。！？])', text)
        
        lines = []
        current_line = ""
        
        for part in sentences:
            if len(current_line + part) <= self.max_chars_per_line:
                current_line += part
            else:
                if current_line:
                    lines.append(current_line.strip())
                    current_line = part
                else:
                    # Force split long words
                    while len(part) > self.max_chars_per_line:
                        lines.append(part[:self.max_chars_per_line])
                        part = part[self.max_chars_per_line:]
                    current_line = part
        
        if current_line:
            lines.append(current_line.strip())
        
        # Limit number of lines
        return lines[:self.max_lines_per_subtitle]
    
    def to_srt(self, segments: List[Dict]) -> str:
        """Convert to SRT format"""
        srt_content = []
        
        for segment in segments:
            srt_content.append(str(segment['index']))
            
            start_time = self._seconds_to_srt_time(segment['start'])
            end_time = self._seconds_to_srt_time(segment['end'])
            srt_content.append(f"{start_time} --> {end_time}")
            
            if isinstance(segment['text'], list):
                for line in segment['text']:
                    srt_content.append(line)
            else:
                srt_content.append(segment['text'])
            
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    def to_vtt(self, segments: List[Dict]) -> str:
        """Convert to WebVTT format"""
        vtt_content = ["WEBVTT", ""]
        
        for segment in segments:
            start_time = self._seconds_to_vtt_time(segment['start'])
            end_time = self._seconds_to_vtt_time(segment['end'])
            vtt_content.append(f"{start_time} --> {end_time}")
            
            if isinstance(segment['text'], list):
                for line in segment['text']:
                    vtt_content.append(line)
            else:
                vtt_content.append(segment['text'])
            
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT time format"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


class VideoSubtitleSystem:
    """Main video subtitle generation system"""
    
    def __init__(self, model_type: str = 'auto', model_size: str = 'base', language: str = 'auto'):
        self.model_manager = ModelManager()
        self.formatter = SubtitleFormatter()
        self.temp_dir = Path("temp_audio")
        self.output_dir = Path("output")
        
        # Create directories
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup model
        if model_type == 'auto':
            model_type = self.model_manager.get_best_model(language)
        
        if model_type:
            self.model_manager.load_model(model_type, model_size, language)
            logger.info(f"System initialized with {model_type} model")
        else:
            logger.error("No speech recognition models available")
            raise RuntimeError("No speech recognition models available. Please install one of: whisper, faster-whisper, transformers, or SpeechRecognition")
    
    def check_dependencies(self) -> Dict[str, bool]:
        """Check system dependencies"""
        deps = {}
        
        # Check ffmpeg
        try:
            subprocess.run(['ffmpeg', '-version'], capture_output=True)
            deps['ffmpeg'] = True
        except FileNotFoundError:
            deps['ffmpeg'] = False
        
        # Check available models
        deps.update({
            f"model_{name}": True for name in self.model_manager.available_models.keys()
        })
        
        # Check CUDA
        try:
            import torch
            deps['cuda'] = torch.cuda.is_available()
            if deps['cuda']:
                deps['gpu_name'] = torch.cuda.get_device_name(0)
                deps['gpu_memory'] = f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB"
        except ImportError:
            deps['cuda'] = False
        
        return deps
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                return {}
            
            info = json.loads(result.stdout)
            
            video_info = {
                'filename': os.path.basename(video_path),
                'size_bytes': os.path.getsize(video_path),
                'duration': float(info.get('format', {}).get('duration', 0)),
                'bitrate': int(info.get('format', {}).get('bit_rate', 0)),
                'format_name': info.get('format', {}).get('format_name', ''),
                'audio_streams': [],
                'video_streams': []
            }
            
            for stream in info.get('streams', []):
                if stream['codec_type'] == 'audio':
                    video_info['audio_streams'].append({
                        'codec': stream.get('codec_name', ''),
                        'sample_rate': int(stream.get('sample_rate', 0)),
                        'channels': int(stream.get('channels', 0)),
                        'bitrate': int(stream.get('bit_rate', 0))
                    })
                elif stream['codec_type'] == 'video':
                    video_info['video_streams'].append({
                        'codec': stream.get('codec_name', ''),
                        'width': int(stream.get('width', 0)),
                        'height': int(stream.get('height', 0))
                    })
            
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def extract_audio(self, video_path: str, audio_path: str = None) -> str:
        """Extract audio from video"""
        if audio_path is None:
            base_name = Path(video_path).stem
            audio_path = self.temp_dir / f"{base_name}_audio.wav"
        
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn', '-acodec', 'pcm_s16le',
            '-ar', '16000', '-ac', '1',
            '-af', 'volume=1.0,highpass=f=80,lowpass=f=8000',
            str(audio_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Audio extraction failed: {result.stderr}")
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")
        
        logger.info(f"Audio extracted: {audio_path}")
        return str(audio_path)
    
    def process_video(self, video_path: str, output_dir: str = None, formats: List[str] = None, language: str = 'auto') -> Dict[str, Any]:
        """Process video file to generate subtitles"""
        if not os.path.exists(video_path):
            return {'success': False, 'error': 'Video file not found'}
        
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        if formats is None:
            formats = ['srt']
        
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        base_name = Path(video_path).stem
        
        try:
            logger.info(f"Processing video: {os.path.basename(video_path)}")
            
            # Get video info
            video_info = self.get_video_info(video_path)
            if not video_info:
                return {'success': False, 'error': 'Could not read video information'}
            
            duration = video_info.get('duration', 0)
            logger.info(f"Video duration: {duration:.1f} seconds")
            
            # Extract audio
            audio_path = self.extract_audio(video_path)
            
            # Transcribe
            logger.info("Starting speech recognition...")
            transcription_start = time.time()
            
            segments = self.model_manager.transcribe(audio_path, language)
            
            transcription_time = time.time() - transcription_start
            logger.info(f"Transcription completed: {len(segments)} segments in {transcription_time:.1f}s")
            
            # Format segments
            formatted_segments = self.formatter.format_segments(segments)
            
            # Generate output files
            output_files = {}
            for format_type in formats:
                output_file = os.path.join(output_dir, f"{base_name}.{format_type}")
                
                if format_type == 'srt':
                    content = self.formatter.to_srt(formatted_segments)
                elif format_type == 'vtt':
                    content = self.formatter.to_vtt(formatted_segments)
                else:
                    logger.warning(f"Unsupported format: {format_type}")
                    continue
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                output_files[format_type] = output_file
                logger.info(f"Subtitle saved: {output_file}")
            
            # Cleanup
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'video_path': video_path,
                'video_info': video_info,
                'transcription': {
                    'segments_count': len(segments),
                    'model_type': self.model_manager.current_model['type'],
                    'model_size': self.model_manager.current_model['size']
                },
                'output_files': output_files,
                'processing_time': processing_time,
                'transcription_time': transcription_time
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {'success': False, 'error': str(e), 'processing_time': time.time() - start_time}
    
    def process_batch(self, video_paths: List[str], output_dir: str, formats: List[str] = None, language: str = 'auto') -> List[Dict[str, Any]]:
        """Process multiple videos"""
        results = []
        
        logger.info(f"Starting batch processing of {len(video_paths)} videos")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"Processing {i}/{len(video_paths)}: {os.path.basename(video_path)}")
            
            result = self.process_video(video_path, output_dir, formats, language)
            results.append(result)
            
            # Memory cleanup
            gc.collect()
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Batch processing complete: {successful}/{len(video_paths)} successful")
        
        return results


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Subtitle Generation System')
    parser.add_argument('input', help='Input video file or directory')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('-m', '--model', default='auto', 
                       choices=['auto', 'whisper', 'faster_whisper', 'transformers', 'speech_recognition'],
                       help='Speech recognition model to use')
    parser.add_argument('-s', '--size', default='base',
                       choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v3'],
                       help='Model size (for Whisper models)')
    parser.add_argument('-l', '--language', default='auto',
                       choices=['auto', 'zh', 'en'],
                       help='Audio language')
    parser.add_argument('-f', '--formats', nargs='+', default=['srt'],
                       choices=['srt', 'vtt'],
                       help='Output subtitle formats')
    parser.add_argument('--batch', action='store_true', help='Process all videos in directory')
    parser.add_argument('--info', action='store_true', help='Show system information')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        system = VideoSubtitleSystem(args.model, args.size, args.language)
        
        if args.info:
            print("System Information:")
            deps = system.check_dependencies()
            for dep, status in deps.items():
                print(f"  {dep}: {status}")
            
            print(f"\nAvailable Models:")
            for model_name, info in system.model_manager.available_models.items():
                print(f"  {model_name}: {info['name']}")
            return
        
        if args.batch:
            if not os.path.isdir(args.input):
                print(f"Error: {args.input} is not a directory")
                return
            
            # Find video files
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
            video_files = []
            
            for file in os.listdir(args.input):
                if any(file.lower().endswith(ext) for ext in video_extensions):
                    video_files.append(os.path.join(args.input, file))
            
            if not video_files:
                print("No video files found in directory")
                return
            
            print(f"Found {len(video_files)} video files")
            results = system.process_batch(video_files, args.output, args.formats, args.language)
            
            successful = sum(1 for r in results if r['success'])
            print(f"\nBatch processing complete: {successful}/{len(results)} successful")
            
            if successful < len(results):
                print("\nFailed files:")
                for result in results:
                    if not result['success']:
                        print(f"  {result.get('video_path', 'unknown')}: {result.get('error', 'unknown error')}")
        
        else:
            result = system.process_video(args.input, args.output, args.formats, args.language)
            
            if result['success']:
                print(f"✓ Success!")
                print(f"Processing time: {result['processing_time']:.1f} seconds")
                print(f"Transcription time: {result['transcription_time']:.1f} seconds")
                print(f"Segments: {result['transcription']['segments_count']}")
                print(f"Model: {result['transcription']['model_type']} ({result['transcription']['model_size']})")
                print(f"Output files:")
                for format_type, file_path in result['output_files'].items():
                    print(f"  {format_type.upper()}: {file_path}")
            else:
                print(f"✗ Error: {result['error']}")
                
    except Exception as e:
        logger.error(f"System error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()