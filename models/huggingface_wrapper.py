"""
Hugging Face transformers model wrapper
"""

import os
import torch
import logging
from typing import Dict, Any, List
from models.model_wrapper import BaseModelWrapper

logger = logging.getLogger(__name__)

class HuggingFaceModelWrapper(BaseModelWrapper):
    """Wrapper for Hugging Face transformers models"""
    
    def __init__(self, model_id: str, device: str = "cuda", **kwargs):
        super().__init__(model_id, device, **kwargs)
        self.torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = None
        self.pipe = None
        
    def load_model(self):
        """Load Hugging Face model"""
        try:
            from transformers import (
                AutoModelForSpeechSeq2Seq, 
                AutoProcessor, 
                pipeline
            )
            
            logger.info(f"加载Hugging Face模型: {self.model_id}")
            
            # Load model with memory optimization
            with self.memory_manager.memory_context("hf_model_loading"):
                self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    self.model_id,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    cache_dir=self.kwargs.get("cache_dir", "models_cache"),
                    trust_remote_code=True
                )
                
                # Move to device
                self.model = self.model.to(self.device)
                
                # Load processor
                self.processor = AutoProcessor.from_pretrained(
                    self.model_id,
                    cache_dir=self.kwargs.get("cache_dir", "models_cache"),
                    trust_remote_code=True
                )
                
                # Create pipeline
                self.pipe = pipeline(
                    "automatic-speech-recognition",
                    model=self.model,
                    tokenizer=self.processor.tokenizer,
                    feature_extractor=self.processor.feature_extractor,
                    max_new_tokens=self.kwargs.get("max_new_tokens", 4096),
                    chunk_length_s=self.kwargs.get("chunk_length_s", 30),
                    batch_size=self.kwargs.get("batch_size", 1),
                    torch_dtype=self.torch_dtype,
                    device=self.device,
                )
            
            logger.info(f"Hugging Face模型加载完成: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Hugging Face模型加载失败: {e}")
            raise
    
    def transcribe(self, audio_path: str, **kwargs) -> Dict[str, Any]:
        """Transcribe audio using Hugging Face pipeline"""
        if not self.validate_audio_file(audio_path):
            return {"segments": [], "language": None, "error": "Invalid audio file"}
        
        try:
            # Setup transcription arguments
            transcription_args = {
                "return_timestamps": True,
                "generate_kwargs": {
                    "temperature": kwargs.get("temperature", 0.2),
                    "num_beams": kwargs.get("beam_size", 5),
                    "do_sample": kwargs.get("do_sample", False),
                    "use_cache": True
                }
            }
            
            # Language-specific settings
            language = kwargs.get("language", "auto")
            if language and language != "auto":
                transcription_args["generate_kwargs"]["language"] = language
                
                # For Chinese models or Chinese language
                if "belle-whisper" in self.model_id.lower() or language == "zh":
                    if hasattr(self.processor, "get_decoder_prompt_ids"):
                        try:
                            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                                language="zh", task="transcribe"
                            )
                            transcription_args["generate_kwargs"]["forced_decoder_ids"] = forced_decoder_ids
                            logger.info("启用中文强制解码")
                        except Exception:
                            pass
            
            # Set task for Whisper models
            if "whisper" in self.model_id.lower():
                transcription_args["generate_kwargs"]["task"] = "transcribe"
            
            # Check if audio needs chunking
            max_duration = kwargs.get("max_duration", 1800)  # 30 minutes
            audio_chunks = self.chunk_audio_if_needed(audio_path, max_duration)
            
            all_segments = []
            total_offset = 0.0
            detected_language = language if language != "auto" else "unknown"
            
            for chunk_path in audio_chunks:
                logger.info(f"处理音频块: {os.path.basename(chunk_path)}")
                
                with self.memory_manager.memory_context("hf_transcribe"):
                    result = self.pipe(chunk_path, **transcription_args)
                    
                    # Process results
                    if "chunks" in result:
                        # Timestamped results
                        for chunk in result["chunks"]:
                            if chunk["timestamp"][0] is not None and chunk["timestamp"][1] is not None:
                                segment = {
                                    "start": chunk["timestamp"][0] + total_offset,
                                    "end": chunk["timestamp"][1] + total_offset,
                                    "text": chunk["text"].strip()
                                }
                                all_segments.append(segment)
                    else:
                        # Single result without timestamps
                        duration = self.get_audio_duration(chunk_path)
                        segment = {
                            "start": total_offset,
                            "end": total_offset + duration,
                            "text": result["text"].strip()
                        }
                        all_segments.append(segment)
                    
                    # Update offset for next chunk
                    if audio_chunks != [audio_path]:
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
            
            # Post-process Chinese text if applicable
            if language == "zh" or "belle-whisper" in self.model_id.lower():
                all_segments = self._post_process_chinese_text(all_segments)
            
            logger.info(f"转录完成: {len(all_segments)} 个片段, 语言: {detected_language}")
            
            return {
                "segments": all_segments,
                "language": detected_language,
                "model": self.model_id
            }
            
        except Exception as e:
            logger.error(f"Hugging Face转录失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                "segments": [],
                "language": None,
                "error": str(e)
            }
    
    def _post_process_chinese_text(self, segments: List[Dict]) -> List[Dict]:
        """Post-process Chinese text with jieba segmentation"""
        try:
            import jieba
            
            for segment in segments:
                text = segment["text"]
                # Add spaces between Chinese words for better readability
                words = jieba.cut(text)
                segment["text"] = " ".join(words)
            
            return segments
            
        except Exception as e:
            logger.warning(f"中文文本后处理失败: {e}")
            return segments
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Hugging Face model information"""
        info = super().get_model_info()
        info.update({
            "torch_dtype": str(self.torch_dtype),
            "backend": "transformers",
            "supports_timestamps": True,
            "supports_languages": True
        })
        return info
    
    def unload_model(self):
        """Unload Hugging Face model"""
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
            
        if self.processor is not None:
            del self.processor
            self.processor = None
            
        super().unload_model()
