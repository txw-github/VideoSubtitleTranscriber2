#!/usr/bin/env python3
"""
Simple Video Subtitle Generator
A lightweight implementation for video-to-subtitle conversion
"""

import os
import sys
import subprocess
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleSubtitleGenerator:
    """Simple subtitle generator using available tools"""
    
    def __init__(self):
        self.temp_dir = Path("temp_audio")
        self.temp_dir.mkdir(exist_ok=True)
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
    def check_dependencies(self) -> Dict[str, bool]:
        """Check available dependencies"""
        deps = {}
        
        # Check ffmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True)
            deps['ffmpeg'] = result.returncode == 0
        except FileNotFoundError:
            deps['ffmpeg'] = False
            
        # Check Python packages
        packages = ['numpy', 'jieba', 'psutil']
        for package in packages:
            try:
                __import__(package)
                deps[package] = True
            except ImportError:
                deps[package] = False
                
        return deps
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video information using ffprobe"""
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
                'size_bytes': os.path.getsize(video_path),
                'duration': float(info.get('format', {}).get('duration', 0)),
                'audio_streams': []
            }
            
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    video_info['audio_streams'].append({
                        'codec': stream.get('codec_name', ''),
                        'sample_rate': int(stream.get('sample_rate', 0)),
                        'channels': int(stream.get('channels', 0))
                    })
                    
            return video_info
            
        except Exception as e:
            logger.error(f"Error getting video info: {e}")
            return {}
    
    def extract_audio(self, video_path: str, output_path: str) -> bool:
        """Extract audio from video using ffmpeg"""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', video_path,
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', '16000', '-ac', '1',
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"Audio extraction failed: {result.stderr}")
                return False
                
            logger.info(f"Audio extracted: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error extracting audio: {e}")
            return False
    
    def create_subtitle_from_audio(self, audio_path: str, output_path: str) -> bool:
        """Create subtitle file using available speech recognition"""
        try:
            # Check if we can use speech recognition
            speech_result = self._try_speech_recognition(audio_path)
            
            if speech_result:
                # Use actual speech recognition results
                segments = speech_result
                logger.info(f"Used speech recognition to generate {len(segments)} segments")
            else:
                # Fallback: analyze audio for voice activity
                segments = self._analyze_audio_activity(audio_path)
                logger.info(f"Used voice activity detection to generate {len(segments)} segments")
            
            self._write_srt_file(segments, output_path)
            logger.info(f"Subtitle file created: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating subtitle: {e}")
            return False
    
    def _try_speech_recognition(self, audio_path: str) -> Optional[List[Dict]]:
        """Try to use available speech recognition libraries"""
        # Try OpenAI Whisper if available
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path)
            
            segments = []
            for i, segment in enumerate(result['segments']):
                segments.append({
                    'index': i + 1,
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
            return segments
        except ImportError:
            pass
        
        # Try SpeechRecognition library
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            
            # Split audio into chunks and recognize
            segments = self._recognize_with_sr(audio_path, r)
            return segments
        except ImportError:
            pass
        
        return None
    
    def _analyze_audio_activity(self, audio_path: str) -> List[Dict]:
        """Analyze audio for voice activity using basic methods"""
        try:
            # Get audio duration
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                audio_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            duration = float(result.stdout.strip()) if result.stdout else 60.0
            
            # Use ffmpeg to detect silence and create segments
            return self._detect_voice_segments(audio_path, duration)
            
        except Exception as e:
            logger.warning(f"Could not analyze audio activity: {e}")
            return self._create_time_based_segments(duration)
    
    def _detect_voice_segments(self, audio_path: str, duration: float) -> List[Dict]:
        """Detect voice segments using ffmpeg silencedetect"""
        try:
            cmd = [
                'ffmpeg', '-i', audio_path, '-af',
                'silencedetect=noise=-30dB:duration=0.5',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, stderr=subprocess.STDOUT)
            
            # Parse silence detection output
            import re
            silence_starts = []
            silence_ends = []
            
            for line in result.stderr.split('\n'):
                if 'silence_start' in line:
                    match = re.search(r'silence_start: (\d+\.?\d*)', line)
                    if match:
                        silence_starts.append(float(match.group(1)))
                elif 'silence_end' in line:
                    match = re.search(r'silence_end: (\d+\.?\d*)', line)
                    if match:
                        silence_ends.append(float(match.group(1)))
            
            # Create voice segments between silences
            segments = []
            segment_idx = 1
            
            if not silence_starts:
                # No silence detected, create time-based segments
                return self._create_time_based_segments(duration)
            
            # First segment (start to first silence)
            if silence_starts and silence_starts[0] > 1.0:
                segments.append({
                    'index': segment_idx,
                    'start': 0.0,
                    'end': silence_starts[0],
                    'text': f"[语音片段 {segment_idx}]"
                })
                segment_idx += 1
            
            # Segments between silences
            for i in range(len(silence_ends)):
                start_time = silence_ends[i]
                end_time = silence_starts[i + 1] if i + 1 < len(silence_starts) else duration
                
                if end_time - start_time > 1.0:  # Only create segments longer than 1 second
                    segments.append({
                        'index': segment_idx,
                        'start': start_time,
                        'end': end_time,
                        'text': f"[语音片段 {segment_idx}]"
                    })
                    segment_idx += 1
            
            return segments if segments else self._create_time_based_segments(duration)
            
        except Exception as e:
            logger.warning(f"Voice activity detection failed: {e}")
            return self._create_time_based_segments(duration)
    
    def _create_time_based_segments(self, duration: float) -> List[Dict]:
        """Create time-based segments as fallback"""
        segments = []
        segment_duration = 10.0  # 10 seconds per segment
        num_segments = int(duration / segment_duration) + 1
        
        for i in range(num_segments):
            start_time = i * segment_duration
            end_time = min(start_time + segment_duration, duration)
            
            if end_time - start_time > 1.0:  # Only create segments longer than 1 second
                segments.append({
                    'index': i + 1,
                    'start': start_time,
                    'end': end_time,
                    'text': f"[音频片段 {i + 1:02d}] {self._format_duration(start_time)} - {self._format_duration(end_time)}"
                })
            
        return segments
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in MM:SS format"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def _recognize_with_sr(self, audio_path: str, recognizer) -> List[Dict]:
        """Use SpeechRecognition library to transcribe audio"""
        import speech_recognition as sr
        
        # Convert to WAV if needed and split into chunks
        segments = []
        chunk_duration = 30  # 30 second chunks
        
        # Get duration
        cmd = [
            'ffprobe', '-v', 'error', '-show_entries',
            'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
            audio_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        duration = float(result.stdout.strip()) if result.stdout else 60.0
        
        num_chunks = int(duration / chunk_duration) + 1
        
        for i in range(num_chunks):
            start_time = i * chunk_duration
            end_time = min(start_time + chunk_duration, duration)
            
            # Extract chunk
            chunk_path = self.temp_dir / f"chunk_{i}.wav"
            cmd = [
                'ffmpeg', '-y', '-i', audio_path,
                '-ss', str(start_time), '-t', str(end_time - start_time),
                '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                str(chunk_path)
            ]
            
            subprocess.run(cmd, capture_output=True)
            
            try:
                with sr.AudioFile(str(chunk_path)) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio, language='zh-CN')
                    
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
                    'text': f"[识别失败的音频片段 {i + 1}]"
                })
            
            # Cleanup chunk file
            if chunk_path.exists():
                chunk_path.unlink()
        
        return segments
    
    def _write_srt_file(self, segments: List[Dict], output_path: str):
        """Write segments to SRT file"""
        with open(output_path, 'w', encoding='utf-8') as f:
            for segment in segments:
                f.write(f"{segment['index']}\n")
                start_time = self._seconds_to_srt_time(segment['start'])
                end_time = self._seconds_to_srt_time(segment['end'])
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{segment['text']}\n\n")
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        milliseconds = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def process_video(self, video_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """Process a video file to generate subtitles"""
        if not os.path.exists(video_path):
            return {'success': False, 'error': 'Video file not found'}
        
        if output_dir is None:
            output_dir = str(self.output_dir)
        
        os.makedirs(output_dir, exist_ok=True)
        
        start_time = time.time()
        base_name = Path(video_path).stem
        
        try:
            # Get video info
            logger.info(f"Processing video: {video_path}")
            video_info = self.get_video_info(video_path)
            
            if not video_info:
                return {'success': False, 'error': 'Could not read video information'}
            
            # Extract audio
            audio_path = self.temp_dir / f"{base_name}_audio.wav"
            if not self.extract_audio(video_path, str(audio_path)):
                return {'success': False, 'error': 'Audio extraction failed'}
            
            # Generate subtitles
            subtitle_path = Path(output_dir) / f"{base_name}.srt"
            if not self.create_subtitle_from_audio(str(audio_path), str(subtitle_path)):
                return {'success': False, 'error': 'Subtitle generation failed'}
            
            # Cleanup
            if audio_path.exists():
                audio_path.unlink()
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'video_path': video_path,
                'subtitle_path': str(subtitle_path),
                'video_info': video_info,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Error processing video: {e}")
            return {'success': False, 'error': str(e)}
    
    def process_batch(self, video_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple videos"""
        results = []
        
        logger.info(f"Processing {len(video_paths)} videos")
        
        for i, video_path in enumerate(video_paths, 1):
            logger.info(f"Processing {i}/{len(video_paths)}: {os.path.basename(video_path)}")
            result = self.process_video(video_path, output_dir)
            results.append(result)
        
        successful = sum(1 for r in results if r['success'])
        logger.info(f"Batch processing complete: {successful}/{len(video_paths)} successful")
        
        return results

def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Video Subtitle Generator')
    parser.add_argument('input', help='Input video file or directory')
    parser.add_argument('-o', '--output', default='output', help='Output directory')
    parser.add_argument('--batch', action='store_true', help='Process all videos in directory')
    parser.add_argument('--info', action='store_true', help='Show system information')
    
    args = parser.parse_args()
    
    generator = SimpleSubtitleGenerator()
    
    if args.info:
        print("System Information:")
        deps = generator.check_dependencies()
        for dep, available in deps.items():
            status = "✓" if available else "✗"
            print(f"  {status} {dep}")
        return
    
    if args.batch:
        if not os.path.isdir(args.input):
            print(f"Error: {args.input} is not a directory")
            return
        
        # Find video files
        video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm']
        video_files = []
        
        for file in os.listdir(args.input):
            if any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(os.path.join(args.input, file))
        
        if not video_files:
            print("No video files found in directory")
            return
        
        results = generator.process_batch(video_files, args.output)
        
        successful = sum(1 for r in results if r['success'])
        print(f"\nBatch processing complete: {successful}/{len(results)} successful")
        
    else:
        result = generator.process_video(args.input, args.output)
        
        if result['success']:
            print(f"✓ Success! Subtitle saved to: {result['subtitle_path']}")
            print(f"Processing time: {result['processing_time']:.1f} seconds")
        else:
            print(f"✗ Error: {result['error']}")

if __name__ == '__main__':
    main()