"""
Subtitle formatting utilities for multiple output formats
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from datetime import timedelta

logger = logging.getLogger(__name__)

class SubtitleFormatter:
    """Subtitle formatter supporting multiple formats"""
    
    def __init__(self, 
                 max_chars_per_line: int = 80,
                 max_lines_per_subtitle: int = 2,
                 min_duration: float = 0.5):
        self.max_chars_per_line = max_chars_per_line
        self.max_lines_per_subtitle = max_lines_per_subtitle
        self.min_duration = min_duration
    
    def format_segments_to_subtitles(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert transcription segments to subtitle format"""
        subtitles = []
        
        for i, segment in enumerate(segments):
            start_time = segment.get("start", 0)
            end_time = segment.get("end", start_time + self.min_duration)
            text = segment.get("text", "").strip()
            
            # Skip empty segments
            if not text:
                continue
            
            # Ensure minimum duration
            if end_time - start_time < self.min_duration:
                end_time = start_time + self.min_duration
            
            # Process text
            processed_text = self._process_text(text)
            
            # Split long text into multiple lines
            lines = self._split_text_to_lines(processed_text)
            
            # Create subtitle entry
            subtitle = {
                "index": i + 1,
                "start": start_time,
                "end": end_time,
                "text": lines,
                "raw_text": text
            }
            
            subtitles.append(subtitle)
        
        # Post-process subtitles
        subtitles = self._post_process_subtitles(subtitles)
        
        return subtitles
    
    def _process_text(self, text: str) -> str:
        """Process and clean text"""
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix common punctuation issues
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        # Handle Chinese punctuation
        text = re.sub(r'([。！？；：])\s*', r'\1', text)
        
        return text
    
    def _split_text_to_lines(self, text: str) -> List[str]:
        """Split text into multiple lines if needed"""
        if len(text) <= self.max_chars_per_line:
            return [text]
        
        # Try to split at sentence boundaries first
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
        if len(lines) > self.max_lines_per_subtitle:
            # Combine lines to fit limit
            combined_lines = []
            for i in range(0, len(lines), self.max_lines_per_subtitle):
                combined_text = " ".join(lines[i:i + self.max_lines_per_subtitle])
                if len(combined_text) > self.max_chars_per_line * self.max_lines_per_subtitle:
                    # Split again if too long
                    words = combined_text.split()
                    current_combined = ""
                    for word in words:
                        if len(current_combined + " " + word) <= self.max_chars_per_line:
                            current_combined += " " + word if current_combined else word
                        else:
                            if current_combined:
                                combined_lines.append(current_combined)
                                current_combined = word
                    if current_combined:
                        combined_lines.append(current_combined)
                else:
                    combined_lines.append(combined_text)
            lines = combined_lines[:self.max_lines_per_subtitle]
        
        return lines
    
    def _post_process_subtitles(self, subtitles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Post-process subtitles for timing and content"""
        if not subtitles:
            return subtitles
        
        # Fix overlapping timestamps
        for i in range(len(subtitles) - 1):
            current = subtitles[i]
            next_subtitle = subtitles[i + 1]
            
            if current["end"] > next_subtitle["start"]:
                # Adjust end time to prevent overlap
                gap = 0.1  # 100ms gap
                current["end"] = max(current["start"] + self.min_duration, 
                                   next_subtitle["start"] - gap)
        
        # Remove very short subtitles
        filtered_subtitles = []
        for subtitle in subtitles:
            duration = subtitle["end"] - subtitle["start"]
            if duration >= self.min_duration:
                filtered_subtitles.append(subtitle)
        
        # Re-index
        for i, subtitle in enumerate(filtered_subtitles):
            subtitle["index"] = i + 1
        
        return filtered_subtitles
    
    def to_srt(self, subtitles: List[Dict[str, Any]]) -> str:
        """Convert subtitles to SRT format"""
        srt_content = []
        
        for subtitle in subtitles:
            # Index
            srt_content.append(str(subtitle["index"]))
            
            # Timestamps
            start_time = self._seconds_to_srt_time(subtitle["start"])
            end_time = self._seconds_to_srt_time(subtitle["end"])
            srt_content.append(f"{start_time} --> {end_time}")
            
            # Text (each line on separate line)
            for line in subtitle["text"]:
                srt_content.append(line)
            
            # Empty line separator
            srt_content.append("")
        
        return "\n".join(srt_content)
    
    def to_vtt(self, subtitles: List[Dict[str, Any]]) -> str:
        """Convert subtitles to WebVTT format"""
        vtt_content = ["WEBVTT", ""]
        
        for subtitle in subtitles:
            # Timestamps
            start_time = self._seconds_to_vtt_time(subtitle["start"])
            end_time = self._seconds_to_vtt_time(subtitle["end"])
            vtt_content.append(f"{start_time} --> {end_time}")
            
            # Text (each line on separate line)
            for line in subtitle["text"]:
                vtt_content.append(line)
            
            # Empty line separator
            vtt_content.append("")
        
        return "\n".join(vtt_content)
    
    def to_txt(self, subtitles: List[Dict[str, Any]]) -> str:
        """Convert subtitles to plain text format"""
        txt_content = []
        
        for subtitle in subtitles:
            # Add timestamp in readable format
            start_time = self._seconds_to_readable_time(subtitle["start"])
            end_time = self._seconds_to_readable_time(subtitle["end"])
            txt_content.append(f"[{start_time} - {end_time}]")
            
            # Add text
            for line in subtitle["text"]:
                txt_content.append(line)
            
            txt_content.append("")  # Empty line
        
        return "\n".join(txt_content)
    
    def _seconds_to_srt_time(self, seconds: float) -> str:
        """Convert seconds to SRT time format (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"
    
    def _seconds_to_vtt_time(self, seconds: float) -> str:
        """Convert seconds to WebVTT time format (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        milliseconds = int((seconds - total_seconds) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"
    
    def _seconds_to_readable_time(self, seconds: float) -> str:
        """Convert seconds to readable time format"""
        td = timedelta(seconds=seconds)
        total_seconds = int(td.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def save_subtitles(self, subtitles: List[Dict[str, Any]], 
                      output_path: str, format_type: str = "srt") -> str:
        """Save subtitles to file"""
        try:
            # Generate content based on format
            if format_type.lower() == "srt":
                content = self.to_srt(subtitles)
                if not output_path.endswith('.srt'):
                    output_path = os.path.splitext(output_path)[0] + '.srt'
            elif format_type.lower() == "vtt":
                content = self.to_vtt(subtitles)
                if not output_path.endswith('.vtt'):
                    output_path = os.path.splitext(output_path)[0] + '.vtt'
            elif format_type.lower() == "txt":
                content = self.to_txt(subtitles)
                if not output_path.endswith('.txt'):
                    output_path = os.path.splitext(output_path)[0] + '.txt'
            else:
                raise ValueError(f"不支持的字幕格式: {format_type}")
            
            # Save to file
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"字幕保存成功: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"字幕保存失败: {e}")
            raise
    
    def merge_subtitles(self, subtitle_files: List[str], output_path: str) -> str:
        """Merge multiple subtitle files"""
        try:
            all_subtitles = []
            
            for subtitle_file in subtitle_files:
                if subtitle_file.endswith('.srt'):
                    subtitles = self._parse_srt(subtitle_file)
                elif subtitle_file.endswith('.vtt'):
                    subtitles = self._parse_vtt(subtitle_file)
                else:
                    logger.warning(f"跳过不支持的字幕文件: {subtitle_file}")
                    continue
                
                all_subtitles.extend(subtitles)
            
            # Sort by start time
            all_subtitles.sort(key=lambda x: x["start"])
            
            # Re-index
            for i, subtitle in enumerate(all_subtitles):
                subtitle["index"] = i + 1
            
            # Save merged subtitles
            format_type = os.path.splitext(output_path)[1][1:].lower()
            return self.save_subtitles(all_subtitles, output_path, format_type)
            
        except Exception as e:
            logger.error(f"字幕合并失败: {e}")
            raise
    
    def _parse_srt(self, srt_path: str) -> List[Dict[str, Any]]:
        """Parse SRT file to subtitle list"""
        subtitles = []
        
        with open(srt_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split into subtitle blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        
        for block in blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                try:
                    index = int(lines[0])
                    time_line = lines[1]
                    text_lines = lines[2:]
                    
                    # Parse timestamps
                    start_str, end_str = time_line.split(' --> ')
                    start_time = self._srt_time_to_seconds(start_str)
                    end_time = self._srt_time_to_seconds(end_str)
                    
                    subtitles.append({
                        "index": index,
                        "start": start_time,
                        "end": end_time,
                        "text": text_lines
                    })
                    
                except Exception as e:
                    logger.warning(f"解析SRT块失败: {e}")
        
        return subtitles
    
    def _srt_time_to_seconds(self, time_str: str) -> float:
        """Convert SRT time string to seconds"""
        time_str = time_str.strip()
        hours, minutes, seconds_ms = time_str.split(':')
        seconds, milliseconds = seconds_ms.split(',')
        
        total_seconds = (int(hours) * 3600 + 
                        int(minutes) * 60 + 
                        int(seconds) + 
                        int(milliseconds) / 1000)
        
        return total_seconds
