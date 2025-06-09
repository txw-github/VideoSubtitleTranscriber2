"""
Video processing utilities for subtitle generation
"""

import os
import subprocess
import logging
from typing import List, Dict, Any, Optional, Tuple
from processors.audio_processor import AudioProcessor

logger = logging.getLogger(__name__)

class VideoProcessor:
    """Video processing for subtitle generation workflow"""
    
    def __init__(self):
        self.audio_processor = AudioProcessor()
        self.supported_formats = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
        
    def validate_video_file(self, video_path: str) -> bool:
        """Validate video file"""
        if not os.path.exists(video_path):
            logger.error(f"视频文件不存在: {video_path}")
            return False
        
        if os.path.getsize(video_path) == 0:
            logger.error(f"视频文件为空: {video_path}")
            return False
        
        _, ext = os.path.splitext(video_path.lower())
        if ext not in self.supported_formats:
            logger.error(f"不支持的视频格式: {ext}")
            return False
        
        return True
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information using ffprobe"""
        try:
            command = [
                "ffprobe", "-v", "quiet",
                "-print_format", "json",
                "-show_format", "-show_streams",
                video_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"ffprobe执行失败: {result.stderr}")
                return {}
            
            import json
            info = json.loads(result.stdout)
            
            # Extract relevant information
            video_info = {
                "filename": os.path.basename(video_path),
                "size_bytes": os.path.getsize(video_path),
                "duration": 0,
                "video_streams": [],
                "audio_streams": []
            }
            
            # Parse format info
            if "format" in info:
                video_info["duration"] = float(info["format"].get("duration", 0))
                video_info["bitrate"] = int(info["format"].get("bit_rate", 0))
                video_info["format_name"] = info["format"].get("format_name", "")
            
            # Parse streams
            for stream in info.get("streams", []):
                if stream["codec_type"] == "video":
                    video_info["video_streams"].append({
                        "codec": stream.get("codec_name", ""),
                        "width": stream.get("width", 0),
                        "height": stream.get("height", 0),
                        "fps": self._parse_fps(stream.get("r_frame_rate", "0/1")),
                        "bitrate": int(stream.get("bit_rate", 0))
                    })
                elif stream["codec_type"] == "audio":
                    video_info["audio_streams"].append({
                        "codec": stream.get("codec_name", ""),
                        "sample_rate": int(stream.get("sample_rate", 0)),
                        "channels": int(stream.get("channels", 0)),
                        "bitrate": int(stream.get("bit_rate", 0)),
                        "language": stream.get("tags", {}).get("language", "unknown")
                    })
            
            return video_info
            
        except Exception as e:
            logger.error(f"获取视频信息失败: {e}")
            return {}
    
    def _parse_fps(self, fps_string: str) -> float:
        """Parse FPS from FFmpeg format (e.g., '30000/1001')"""
        try:
            if '/' in fps_string:
                num, den = fps_string.split('/')
                return float(num) / float(den)
            return float(fps_string)
        except:
            return 0.0
    
    def extract_audio_tracks(self, video_path: str, output_dir: Optional[str] = None) -> List[str]:
        """Extract all audio tracks from video"""
        if not self.validate_video_file(video_path):
            return []
        
        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        
        os.makedirs(output_dir, exist_ok=True)
        
        video_info = self.get_video_info(video_path)
        audio_streams = video_info.get("audio_streams", [])
        
        if not audio_streams:
            logger.warning("视频中未找到音频流")
            return []
        
        extracted_files = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        for i, stream in enumerate(audio_streams):
            try:
                output_path = os.path.join(output_dir, f"{base_name}_audio_{i}.wav")
                
                logger.info(f"提取音频流 {i}: {stream.get('codec', 'unknown')} "
                           f"({stream.get('channels', 0)}ch, {stream.get('sample_rate', 0)}Hz)")
                
                # Extract specific audio stream
                command = [
                    "ffmpeg", "-y",
                    "-i", video_path,
                    "-map", f"0:a:{i}",  # Map specific audio stream
                    "-acodec", "pcm_s16le",
                    "-ar", "16000",
                    "-ac", "1",
                    "-af", "volume=1.0,highpass=f=80,lowpass=f=8000",
                    output_path
                ]
                
                result = subprocess.run(command, capture_output=True, text=True)
                
                if result.returncode == 0:
                    extracted_files.append(output_path)
                    logger.info(f"音频流提取成功: {os.path.basename(output_path)}")
                else:
                    logger.error(f"音频流 {i} 提取失败: {result.stderr}")
                    
            except Exception as e:
                logger.error(f"提取音频流 {i} 时发生错误: {e}")
        
        return extracted_files
    
    def extract_best_audio(self, video_path: str, output_path: Optional[str] = None) -> str:
        """Extract the best audio track for speech recognition"""
        if output_path is None:
            name, _ = os.path.splitext(video_path)
            output_path = f"{name}_best_audio.wav"
        
        video_info = self.get_video_info(video_path)
        audio_streams = video_info.get("audio_streams", [])
        
        if not audio_streams:
            raise ValueError("视频中未找到音频流")
        
        # Select best audio stream
        best_stream_index = self._select_best_audio_stream(audio_streams)
        
        logger.info(f"选择音频流 {best_stream_index} 进行处理")
        
        try:
            command = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-map", f"0:a:{best_stream_index}",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                "-af", "volume=1.0,highpass=f=80,lowpass=f=8000,loudnorm=I=-16:TP=-1.5:LRA=11",
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True, timeout=3600)
            
            if result.returncode != 0:
                logger.error(f"音频提取失败: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command)
            
            logger.info(f"最佳音频提取完成: {os.path.basename(output_path)}")
            return output_path
            
        except Exception as e:
            logger.error(f"音频提取失败: {e}")
            raise
    
    def _select_best_audio_stream(self, audio_streams: List[Dict]) -> int:
        """Select the best audio stream for speech recognition"""
        if len(audio_streams) == 1:
            return 0
        
        # Scoring criteria
        best_score = -1
        best_index = 0
        
        for i, stream in enumerate(audio_streams):
            score = 0
            
            # Prefer higher sample rates
            sample_rate = stream.get("sample_rate", 0)
            if sample_rate >= 44100:
                score += 3
            elif sample_rate >= 22050:
                score += 2
            elif sample_rate >= 16000:
                score += 1
            
            # Prefer stereo/mono over surround
            channels = stream.get("channels", 0)
            if channels in [1, 2]:
                score += 2
            elif channels > 2:
                score -= 1
            
            # Prefer higher bitrates
            bitrate = stream.get("bitrate", 0)
            if bitrate >= 128000:
                score += 2
            elif bitrate >= 64000:
                score += 1
            
            # Prefer certain codecs
            codec = stream.get("codec", "").lower()
            if codec in ["pcm_s16le", "pcm_s24le", "flac"]:
                score += 3
            elif codec in ["aac", "mp3"]:
                score += 1
            
            if score > best_score:
                best_score = score
                best_index = i
        
        return best_index
    
    def process_video_batch(self, video_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple videos in batch"""
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for video_path in video_paths:
            try:
                logger.info(f"批量处理视频: {os.path.basename(video_path)}")
                
                if not self.validate_video_file(video_path):
                    results.append({
                        "video_path": video_path,
                        "success": False,
                        "error": "视频文件验证失败"
                    })
                    continue
                
                # Extract audio
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                audio_path = os.path.join(output_dir, f"{base_name}_audio.wav")
                
                extracted_audio = self.extract_best_audio(video_path, audio_path)
                
                # Get video info
                video_info = self.get_video_info(video_path)
                
                results.append({
                    "video_path": video_path,
                    "audio_path": extracted_audio,
                    "video_info": video_info,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"批量处理视频失败 {video_path}: {e}")
                results.append({
                    "video_path": video_path,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def create_video_preview(self, video_path: str, output_path: Optional[str] = None,
                           duration: int = 30, start_time: int = 60) -> str:
        """Create a preview clip for testing"""
        if output_path is None:
            name, ext = os.path.splitext(video_path)
            output_path = f"{name}_preview{ext}"
        
        try:
            command = [
                "ffmpeg", "-y",
                "-ss", str(start_time),
                "-i", video_path,
                "-t", str(duration),
                "-c", "copy",
                output_path
            ]
            
            result = subprocess.run(command, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"视频预览创建失败: {result.stderr}")
                raise subprocess.CalledProcessError(result.returncode, command)
            
            logger.info(f"视频预览创建完成: {os.path.basename(output_path)}")
            return output_path
            
        except Exception as e:
            logger.error(f"视频预览创建失败: {e}")
            raise
    
    def estimate_processing_time(self, video_path: str) -> Dict[str, float]:
        """Estimate processing time based on video characteristics"""
        video_info = self.get_video_info(video_path)
        duration = video_info.get("duration", 0)
        
        # Empirical estimates (actual times may vary)
        estimates = {
            "audio_extraction": duration * 0.05,  # ~5% of video duration
            "whisper_small": duration * 0.3,      # ~30% of video duration
            "whisper_medium": duration * 0.5,     # ~50% of video duration  
            "whisper_large": duration * 0.8,      # ~80% of video duration
            "huggingface_medium": duration * 0.7,
            "huggingface_large": duration * 1.2,
            "subtitle_generation": duration * 0.02
        }
        
        estimates["total_estimated"] = (
            estimates["audio_extraction"] + 
            estimates["whisper_medium"] +  # Default model estimate
            estimates["subtitle_generation"]
        )
        
        return estimates
