#!/usr/bin/env python3
"""
Working Video Subtitle Generation System
Uses available dependencies for speech recognition and subtitle generation
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
from typing import Dict, List, Any, Optional
from datetime import timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('subtitle_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WorkingSpeechRecognizer:
    """Speech recognizer using available libraries"""
    
    def __init__(self):
        self.recognizer = None
        self._setup_recognizer()
    
    def _setup_recognizer(self):
        """Setup speech recognition"""
        try:
            import speech_recognition as sr
            self.recognizer = sr.Recognizer()
            # Optimize for better accuracy
            self.recognizer.energy_threshold = 300
            self.recognizer.dynamic_energy_threshold = True
            self.recognizer.pause_threshold = 0.8
            self.recognizer.phrase_threshold = 0.3
            logger.info("SpeechRecognition initialized successfully")
        except ImportError:
            logger.error("SpeechRecognition not available")
            raise RuntimeError("SpeechRecognition library required. Please install with: pip install SpeechRecognition")
    
    def transcribe_audio_file(self, audio_path: str, language: str = 'zh-CN') -> List[Dict[str, Any]]:
        """Transcribe audio file using Google Speech Recognition"""
        if not self.recognizer:
            raise RuntimeError("Speech recognizer not initialized")
        
        import speech_recognition as sr
        
        # Get audio duration
        duration = self._get_audio_duration(audio_path)
        logger.info(f"Audio duration: {duration:.1f} seconds")
        
        # Split into manageable chunks (30 seconds each)
        chunk_duration = 30
        num_chunks = max(1, int(duration / chunk_duration) + 1)
        
        segments = []
        temp_dir = Path("temp_chunks")
        temp_dir.mkdir(exist_ok=True)
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min(start_time + chunk_duration, duration)
            
            if end_time - start_time < 1.0:  # Skip very short segments
                continue
            
            # Extract chunk
            chunk_path = temp_dir / f"chunk_{i:03d}.wav"
            success = self._extract_audio_chunk(audio_path, str(chunk_path), start_time, end_time - start_time)
            
            if not success:
                logger.warning(f"Failed to extract chunk {i}")
                continue
            
            try:
                # Transcribe chunk
                with sr.AudioFile(str(chunk_path)) as source:
                    # Adjust for ambient noise
                    self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                    audio_data = self.recognizer.record(source)
                
                # Try recognition with fallback languages
                text = self._recognize_with_fallback(audio_data, language)
                
                if text and text.strip():
                    segments.append({
                        'index': len(segments) + 1,
                        'start': start_time,
                        'end': end_time,
                        'text': text.strip(),
                        'confidence': 0.8  # Approximate confidence
                    })
                    logger.info(f"Chunk {i+1}/{num_chunks}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
                else:
                    logger.warning(f"No text recognized for chunk {i}")
                
            except sr.UnknownValueError:
                logger.warning(f"Could not understand audio in chunk {i}")
            except sr.RequestError as e:
                logger.error(f"Recognition service error for chunk {i}: {e}")
            except Exception as e:
                logger.error(f"Error processing chunk {i}: {e}")
            
            # Cleanup chunk file
            try:
                chunk_path.unlink()
            except:
                pass
        
        # Cleanup temp directory
        try:
            temp_dir.rmdir()
        except:
            pass
        
        logger.info(f"Transcription complete: {len(segments)} segments recognized")
        return segments
    
    def _recognize_with_fallback(self, audio_data, primary_language: str) -> str:
        """Try recognition with multiple language fallbacks"""
        import speech_recognition as sr
        
        languages = [primary_language]
        
        # Add fallback languages
        if primary_language != 'zh-CN':
            languages.append('zh-CN')
        if primary_language != 'en-US':
            languages.append('en-US')
        
        for lang in languages:
            try:
                text = self.recognizer.recognize_google(audio_data, language=lang)
                if text and text.strip():
                    return text
            except sr.UnknownValueError:
                continue
            except sr.RequestError:
                continue
        
        return ""
    
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
                '-af', 'volume=2.0,highpass=f=200,lowpass=f=3000',  # Audio enhancement
                output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            return result.returncode == 0
        except:
            return False


class SubtitleFormatter:
    """Format transcription results into subtitle files"""
    
    def __init__(self, max_chars_per_line: int = 80, max_lines: int = 2):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines = max_lines
    
    def format_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format segments for subtitle display"""
        formatted = []
        
        for segment in segments:
            text = segment['text'].strip()
            if not text:
                continue
            
            # Clean and process text
            text = self._clean_text(text)
            
            # Split into appropriate lines
            lines = self._split_text_to_lines(text)
            
            # Ensure minimum duration
            duration = segment['end'] - segment['start']
            min_duration = max(1.0, len(text) * 0.05)  # ~50ms per character
            
            if duration < min_duration:
                segment['end'] = segment['start'] + min_duration
            
            formatted.append({
                'index': segment['index'],
                'start': segment['start'],
                'end': segment['end'],
                'text': lines,
                'original': text
            })
        
        return self._merge_short_segments(formatted)
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix punctuation spacing
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Handle Chinese punctuation
        text = re.sub(r'([。！？；：])\s*', r'\1', text)
        
        return text.strip()
    
    def _split_text_to_lines(self, text: str) -> List[str]:
        """Split text into subtitle lines"""
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        # Try to split at punctuation or natural breaks
        split_points = ['.', '!', '?', '。', '！', '？', ',', '，', ' ']
        
        lines = []
        remaining = text
        
        while remaining and len(lines) < self.max_lines:
            if len(remaining) <= self.max_chars_per_line:
                lines.append(remaining)
                break
            
            # Find best split point
            best_split = self.max_chars_per_line
            for point in split_points:
                idx = remaining[:self.max_chars_per_line].rfind(point)
                if idx > self.max_chars_per_line * 0.6:  # At least 60% of max length
                    if point in ['.', '!', '?', '。', '！', '？']:
                        best_split = idx + 1
                        break
                    else:
                        best_split = idx
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
            
            # If segment is very short and text is brief, try to merge
            if duration < 2.0 and len(current['original']) < 20 and i + 1 < len(segments):
                next_seg = segments[i + 1]
                
                # Check if we can merge (gap < 3 seconds)
                gap = next_seg['start'] - current['end']
                if gap < 3.0:
                    # Merge segments
                    combined_text = current['original'] + ' ' + next_seg['original']
                    combined_lines = self._split_text_to_lines(combined_text)
                    
                    merged_segment = {
                        'index': len(merged) + 1,
                        'start': current['start'],
                        'end': next_seg['end'],
                        'text': combined_lines,
                        'original': combined_text
                    }
                    merged.append(merged_segment)
                    i += 2  # Skip next segment as it's merged
                    continue
            
            # Re-index
            current['index'] = len(merged) + 1
            merged.append(current)
            i += 1
        
        return merged
    
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
            
            srt_lines.append("")  # Empty line between subtitles
        
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
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


class VideoSubtitleGenerator:
    """Main video subtitle generation system"""
    
    def __init__(self):
        self.recognizer = WorkingSpeechRecognizer()
        self.formatter = SubtitleFormatter()
        self.temp_dir = Path("temp_audio")
        self.output_dir = Path("output")
        
        # Create directories
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        
        logger.info("Video subtitle generator initialized")
    
    def check_system(self) -> Dict[str, bool]:
        """Check system capabilities"""
        status = {}
        
        # Check ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], capture_output=True)
            status['ffmpeg'] = result.returncode == 0
        except FileNotFoundError:
            status['ffmpeg'] = False
        
        # Check ffprobe
        try:
            result = subprocess.run(['ffprobe', '-version'], capture_output=True)
            status['ffprobe'] = result.returncode == 0
        except FileNotFoundError:
            status['ffprobe'] = False
        
        # Check SpeechRecognition
        try:
            import speech_recognition
            status['speech_recognition'] = True
        except ImportError:
            status['speech_recognition'] = False
        
        # Check internet connectivity (needed for Google Speech API)
        try:
            import requests
            response = requests.get('https://www.google.com', timeout=5)
            status['internet'] = response.status_code == 200
        except:
            status['internet'] = False
        
        return status
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information"""
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
                'size_mb': os.path.getsize(video_path) / (1024 * 1024),
                'duration': float(info.get('format', {}).get('duration', 0)),
                'bitrate': int(info.get('format', {}).get('bit_rate', 0)),
                'format': info.get('format', {}).get('format_name', ''),
                'has_audio': False,
                'audio_streams': []
            }
            
            # Check for audio streams
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    video_info['has_audio'] = True
                    video_info['audio_streams'].append({
                        'codec': stream.get('codec_name', ''),
                        'sample_rate': int(stream.get('sample_rate', 0)),
                        'channels': int(stream.get('channels', 0))
                    })
            
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def extract_audio(self, video_path: str, output_path: str = None) -> str:
        """Extract audio from video file"""
        if output_path is None:
            base_name = Path(video_path).stem
            output_path = self.temp_dir / f"{base_name}_audio.wav"
        
        # Enhanced audio extraction with noise reduction
        cmd = [
            'ffmpeg', '-y', '-i', video_path,
            '-vn',  # No video
            '-acodec', 'pcm_s16le',  # 16-bit PCM
            '-ar', '16000',  # 16kHz sample rate
            '-ac', '1',  # Mono
            '-af', 'volume=1.5,highpass=f=200,lowpass=f=3000,dynaudnorm=f=75:g=25',  # Audio filters
            str(output_path)
        ]
        
        logger.info(f"Extracting audio from: {os.path.basename(video_path)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"Audio extraction failed: {result.stderr}")
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")
        
        logger.info(f"Audio extracted: {output_path}")
        return str(output_path)
    
    def process_video(self, video_path: str, output_dir: str = None, 
                     formats: List[str] = None, language: str = 'zh-CN') -> Dict[str, Any]:
        """Process video to generate subtitles"""
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
                return {'success': False, 'error': 'Cannot read video file information'}
            
            if not video_info.get('has_audio'):
                return {'success': False, 'error': 'Video file contains no audio tracks'}
            
            duration = video_info.get('duration', 0)
            logger.info(f"Video duration: {duration:.1f} seconds, Size: {video_info.get('size_mb', 0):.1f} MB")
            
            # Extract audio
            audio_path = self.extract_audio(video_path)
            
            # Transcribe audio
            logger.info("Starting speech recognition...")
            transcription_start = time.time()
            
            segments = self.recognizer.transcribe_audio_file(audio_path, language)
            
            transcription_time = time.time() - transcription_start
            
            if not segments:
                return {'success': False, 'error': 'No speech detected in audio'}
            
            logger.info(f"Speech recognition completed: {len(segments)} segments in {transcription_time:.1f}s")
            
            # Format segments
            formatted_segments = self.formatter.format_segments(segments)
            
            # Generate subtitle files
            output_files = {}
            for format_type in formats:
                output_file = Path(output_dir) / f"{base_name}.{format_type}"
                
                if format_type == 'srt':
                    content = self.formatter.to_srt(formatted_segments)
                elif format_type == 'vtt':
                    content = self.formatter.to_vtt(formatted_segments)
                else:
                    logger.warning(f"Unsupported format: {format_type}")
                    continue
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                output_files[format_type] = str(output_file)
                logger.info(f"Subtitle saved: {output_file}")
            
            # Cleanup temporary audio file
            try:
                os.remove(audio_path)
            except:
                pass
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'video_path': video_path,
                'video_info': video_info,
                'transcription': {
                    'segments_count': len(segments),
                    'formatted_segments_count': len(formatted_segments),
                    'language': language,
                    'method': 'Google Speech Recognition'
                },
                'output_files': output_files,
                'processing_time': processing_time,
                'transcription_time': transcription_time
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {
                'success': False, 
                'error': str(e), 
                'processing_time': time.time() - start_time
            }
    
    def process_batch(self, video_files: List[str], output_dir: str, 
                     formats: List[str] = None, language: str = 'zh-CN') -> List[Dict[str, Any]]:
        """Process multiple video files"""
        results = []
        
        logger.info(f"Starting batch processing of {len(video_files)} videos")
        
        for i, video_path in enumerate(video_files, 1):
            logger.info(f"\n--- Processing {i}/{len(video_files)}: {os.path.basename(video_path)} ---")
            
            result = self.process_video(video_path, output_dir, formats, language)
            results.append(result)
            
            # Log progress
            if result['success']:
                logger.info(f"✓ Completed: {result['transcription']['segments_count']} segments, "
                           f"{result['processing_time']:.1f}s")
            else:
                logger.error(f"✗ Failed: {result['error']}")
            
            # Memory cleanup
            gc.collect()
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"\nBatch processing complete: {successful}/{len(video_files)} successful")
        
        return results


def main():
    """Command line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Subtitle Generation System')
    parser.add_argument('input', help='Input video file or directory')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('-l', '--language', default='zh-CN',
                       choices=['zh-CN', 'en-US', 'auto'],
                       help='Speech recognition language')
    parser.add_argument('-f', '--formats', nargs='+', default=['srt'],
                       choices=['srt', 'vtt'],
                       help='Output subtitle formats')
    parser.add_argument('--batch', action='store_true', 
                       help='Process all videos in directory')
    parser.add_argument('--check', action='store_true', 
                       help='Check system capabilities')
    parser.add_argument('-v', '--verbose', action='store_true', 
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        generator = VideoSubtitleGenerator()
        
        if args.check:
            print("System Status Check:")
            status = generator.check_system()
            for component, available in status.items():
                status_text = "✓ Available" if available else "✗ Not Available"
                print(f"  {component}: {status_text}")
            
            if not status.get('ffmpeg') or not status.get('speech_recognition'):
                print("\nMissing required components:")
                if not status.get('ffmpeg'):
                    print("  - Install FFmpeg from: https://ffmpeg.org/")
                if not status.get('speech_recognition'):
                    print("  - Install SpeechRecognition: pip install SpeechRecognition")
            
            if not status.get('internet'):
                print("\nWarning: No internet connection - Google Speech Recognition requires internet")
            
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
            for file in video_files:
                print(f"  - {os.path.basename(file)}")
            
            print(f"\nStarting batch processing...")
            results = generator.process_batch(video_files, args.output, args.formats, args.language)
            
            # Summary
            successful = sum(1 for r in results if r['success'])
            total_time = sum(r.get('processing_time', 0) for r in results)
            total_segments = sum(r.get('transcription', {}).get('segments_count', 0) for r in results if r['success'])
            
            print(f"\n=== Batch Processing Summary ===")
            print(f"Processed: {len(results)} files")
            print(f"Successful: {successful}")
            print(f"Failed: {len(results) - successful}")
            print(f"Total processing time: {total_time:.1f} seconds")
            print(f"Total segments generated: {total_segments}")
            
            if successful < len(results):
                print(f"\nFailed files:")
                for result in results:
                    if not result['success']:
                        print(f"  - {os.path.basename(result.get('video_path', 'unknown'))}: {result.get('error', 'unknown error')}")
        
        else:
            # Single file processing
            result = generator.process_video(args.input, args.output, args.formats, args.language)
            
            if result['success']:
                print(f"✓ Success!")
                print(f"Video: {os.path.basename(result['video_path'])}")
                print(f"Duration: {result['video_info']['duration']:.1f} seconds")
                print(f"Processing time: {result['processing_time']:.1f} seconds")
                print(f"Transcription time: {result['transcription_time']:.1f} seconds")
                print(f"Segments: {result['transcription']['segments_count']}")
                print(f"Final subtitles: {result['transcription']['formatted_segments_count']}")
                print(f"Output files:")
                for format_type, file_path in result['output_files'].items():
                    print(f"  {format_type.upper()}: {file_path}")
            else:
                print(f"✗ Failed: {result['error']}")
                
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        logger.error(f"System error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()