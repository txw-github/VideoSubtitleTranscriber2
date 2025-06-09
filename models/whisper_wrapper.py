"""
Whisper model wrapper using faster-whisper
"""

import os
import logging
from typing import Dict, Any, List
from models.model_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)

class WhisperModelWrapper(BaseModelWrapper):
    """Wrapper for faster-whisper models"""
    
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        super().__init__(model_id, device, **kwargs)
        self.compute_type = kwargs.get("compute_type", "int8_float16")
        self.download_root = kwargs.get("download_root", "models_cache")
        self.cpu_threads = kwargs.get("cpu_threads", 4)
        
    def load_model(self):
        """Load faster-whisper model"""
        try:
            from faster_whisper import WhisperModel
            
            # Parse model size from model_id (e.g., "whisper-large-v3" -> "large-v3")
            if self.model_id.startswith("whisper-"):
                model_size = self.model_id.replace("whisper-", "")
            else:
                model_size = self.model_id
            
            # Optimize compute type based on device and model size
            if self.device == "cpu":
                compute_type = "int8"
            elif "large" in model_size:
                compute_type = "int8_float16"  # Conservative for large models
            else:
                compute_type = self.compute_type
            
            logger.info(f"加载Whisper模型: {model_size}, 设备: {self.device}, 精度: {compute_type}")
            
            # Create model with memory optimization
            self.model = WhisperModel(
                model_size,
                device=self.device,
                compute_type=compute_type,
                download_root=self.download_root,
                cpu_threads=self.cpu_threads,
                local_files_only=False
            )
            
            logger.info(f"Whisper模型加载成功: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Whisper模型加载失败: {e}")
            raise
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio using faster-whisper"""
        if not self.validate_audio_file(audio_path):
            return {"segments": [], "language": None, "error": "Invalid audio file"}
        
        try:
            # Get transcription parameters
            language = kwargs.get("language", "auto")
            if language == "auto":
                language = None
            
            beam_size = kwargs.get("beam_size", 5)
            temperature = kwargs.get("temperature", 0.2)
            condition_on_previous_text = kwargs.get("condition_on_previous_text", True)
            
            # Check if audio needs chunking
            max_duration = kwargs.get("max_duration", 1800)  # 30 minutes
            audio_chunks = self.chunk_audio_if_needed(audio_path, max_duration)
            
            all_segments = []
            total_offset = 0.0
            detected_language = None
            
            for chunk_path in audio_chunks:
                logger.info(f"处理音频块: {os.path.basename(chunk_path)}")
                
                with self.memory_manager.memory_context("whisper_transcribe"):
                    segments, info = self.model.transcribe(
                        chunk_path,
                        language=language,
                        beam_size=beam_size,
                        temperature=temperature,
                        condition_on_previous_text=condition_on_previous_text,
                        vad_filter=True,  # Enable voice activity detection
                        vad_parameters=dict(
                            min_silence_duration_ms=500,
                            min_speech_duration_ms=250
                        )
                    )
                    
                    # Store detected language from first chunk
                    if detected_language is None:
                        detected_language = info.language
                    
                    # Convert segments to list and adjust timestamps
                    for segment in segments:
                        segment_dict = {
                            "start": segment.start + total_offset,
                            "end": segment.end + total_offset,
                            "text": segment.text.strip()
                        }
                        all_segments.append(segment_dict)
                    
                    # Update offset for next chunk
                    if audio_chunks != [audio_path]:  # Only if we actually chunked
                        chunk_duration = self.get_audio_duration(chunk_path)
                        total_offset += chunk_duration
                
                # Clean up chunk file if it was created
                if chunk_path != audio_path:
                    try:
                        os.remove(chunk_path)
                    except Exception:
                        pass
            
            # Filter out empty segments
            all_segments = [seg for seg in all_segments if seg["text"].strip()]
            
            logger.info(f"转录完成: {len(all_segments)} 个片段, 语言: {detected_language}")
            
            return {
                "segments": all_segments,
                "language": detected_language,
                "model": self.model_id
            }
            
        except Exception as e:
            logger.error(f"Whisper转录失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "segments": [],
                "language": None,
                "error": str(e)
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Whisper model information"""
        info = super().get_model_info()
        info.update({
            "compute_type": self.compute_type,
            "backend": "faster-whisper",
            "supports_vad": True,
            "supports_timestamps": True
        })
        return info
    
    def transcribe_with_word_timestamps(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe with word-level timestamps"""
        if not self.validate_audio_file(audio_path):
            return {"segments": [], "language": None, "error": "Invalid audio file"}
        
        try:
            language = kwargs.get("language", "auto")
            if language == "auto":
                language = None
            
            with self.memory_manager.memory_context("whisper_word_timestamps"):
                segments, info = self.model.transcribe(
                    audio_path,
                    language=language,
                    beam_size=kwargs.get("beam_size", 5),
                    temperature=kwargs.get("temperature", 0.2),
                    word_timestamps=True,
                    vad_filter=True
                )
                
                result_segments = []
                for segment in segments:
                    segment_dict = {
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "words": []
                    }
                    
                    # Add word-level timestamps if available
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            word_dict = {
                                "start": word.start,
                                "end": word.end,
                                "word": word.word,
                                "probability": getattr(word, 'probability', 1.0)
                            }
                            segment_dict["words"].append(word_dict)
                    
                    result_segments.append(segment_dict)
                
                return {
                    "segments": result_segments,
                    "language": info.language,
                    "model": self.model_id
                }
                
        except Exception as e:
            logger.error(f"Whisper词级时间戳转录失败: {e}")
            return {
                "segments": [],
                "language": None,
                "error": str(e)
            }
