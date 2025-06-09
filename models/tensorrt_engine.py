"""
TensorRT optimization engine for model acceleration
"""

import os
import torch
import logging
from typing import Dict, Any, Optional
import numpy as np

logger = logging.getLogger(__name__)

class TensorRTEngine:
    """TensorRT optimization engine for GPU acceleration"""
    
    def __init__(self, model_path: str, device: str = "cuda", precision: str = "fp16"):
        self.model_path = os.path.abspath(model_path)
        self.device = device
        self.precision = precision
        self.engine = None
        self.context = None
        self.stream = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.tensorrt_available = self._check_tensorrt()
        
        if self.tensorrt_available:
            self._initialize_cuda()
    
    def _check_tensorrt(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            logger.info(f"TensorRT可用，版本: {trt.__version__}")
            return True
        except ImportError:
            logger.warning("TensorRT不可用，将使用标准PyTorch推理")
            return False
    
    def _initialize_cuda(self):
        """Initialize CUDA context"""
        try:
            import pycuda.driver as cuda
            import pycuda.autoinit
            cuda.init()
            self.stream = cuda.Stream()
            logger.info("CUDA上下文初始化成功")
        except Exception as e:
            logger.error(f"CUDA初始化失败: {e}")
            self.tensorrt_available = False
    
    def optimize_model(self, model, sample_input: torch.Tensor, **kwargs) -> bool:
        """Optimize model with TensorRT"""
        if not self.tensorrt_available:
            logger.warning("TensorRT不可用，跳过优化")
            return False
        
        try:
            import tensorrt as trt
            
            logger.info("开始TensorRT模型优化...")
            
            # Create TensorRT logger
            trt_logger = trt.Logger(trt.Logger.INFO)
            
            # Create builder and network
            builder = trt.Builder(trt_logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            
            # Configure builder
            config = builder.create_builder_config()
            
            # Set memory pool
            if hasattr(config, 'set_memory_pool_limit'):
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)  # 2GB
            else:
                config.max_workspace_size = 2 << 30
            
            # Set precision
            if self.precision == "fp16" and builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("启用FP16精度")
            elif self.precision == "int8" and builder.platform_has_fast_int8:
                config.set_flag(trt.BuilderFlag.INT8)
                logger.info("启用INT8精度")
            
            # Convert PyTorch model to ONNX first
            onnx_path = self._convert_to_onnx(model, sample_input)
            if not onnx_path:
                return False
            
            # Parse ONNX model
            parser = trt.OnnxParser(network, trt_logger)
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    logger.error("ONNX模型解析失败")
                    for error in range(parser.num_errors):
                        logger.error(parser.get_error(error))
                    return False
            
            # Build engine
            engine_data = builder.build_serialized_network(network, config)
            if not engine_data:
                logger.error("TensorRT引擎构建失败")
                return False
            
            # Save engine
            engine_path = os.path.join(self.model_path, "model.trt")
            os.makedirs(self.model_path, exist_ok=True)
            with open(engine_path, 'wb') as f:
                f.write(engine_data)
            
            logger.info(f"TensorRT引擎已保存: {engine_path}")
            
            # Load engine
            return self._load_engine(engine_path)
            
        except Exception as e:
            logger.error(f"TensorRT优化失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _convert_to_onnx(self, model, sample_input: torch.Tensor) -> Optional[str]:
        """Convert PyTorch model to ONNX"""
        try:
            onnx_path = os.path.join(self.model_path, "model.onnx")
            os.makedirs(self.model_path, exist_ok=True)
            
            # Export to ONNX
            torch.onnx.export(
                model,
                sample_input,
                onnx_path,
                export_params=True,
                opset_version=17,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size', 2: 'sequence'},
                    'output': {0: 'batch_size'}
                }
            )
            
            logger.info(f"ONNX模型已保存: {onnx_path}")
            return onnx_path
            
        except Exception as e:
            logger.error(f"ONNX转换失败: {e}")
            return None
    
    def _load_engine(self, engine_path: str) -> bool:
        """Load TensorRT engine"""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            
            # Create runtime
            runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            
            # Load engine
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if not self.engine:
                logger.error("引擎加载失败")
                return False
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            if not self.context:
                logger.error("执行上下文创建失败")
                return False
            
            # Setup bindings
            self._setup_bindings()
            
            logger.info("TensorRT引擎加载成功")
            return True
            
        except Exception as e:
            logger.error(f"引擎加载失败: {e}")
            return False
    
    def _setup_bindings(self):
        """Setup input/output bindings"""
        try:
            import pycuda.driver as cuda
            
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_shape = self.engine.get_tensor_shape(tensor_name)
                tensor_dtype = self.engine.get_tensor_dtype(tensor_name)
                
                # Calculate size
                size = abs(np.prod(tensor_shape))
                dtype = np.float32 if tensor_dtype == 0 else np.float16  # Simplified
                
                # Allocate memory
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                self.bindings.append(int(device_mem))
                
                if self.engine.get_tensor_mode(tensor_name).name == 'INPUT':
                    self.inputs.append({
                        'name': tensor_name,
                        'host': host_mem,
                        'device': device_mem,
                        'shape': tensor_shape,
                        'dtype': dtype
                    })
                else:
                    self.outputs.append({
                        'name': tensor_name,
                        'host': host_mem,
                        'device': device_mem,
                        'shape': tensor_shape,
                        'dtype': dtype
                    })
            
            logger.info(f"绑定设置完成: {len(self.inputs)}个输入, {len(self.outputs)}个输出")
            
        except Exception as e:
            logger.error(f"绑定设置失败: {e}")
    
    def infer(self, input_data: np.ndarray) -> Optional[np.ndarray]:
        """Run inference with TensorRT engine"""
        if not self.engine or not self.context:
            logger.error("引擎未加载")
            return None
        
        try:
            import pycuda.driver as cuda
            
            # Copy input data to device
            if len(self.inputs) > 0:
                np.copyto(self.inputs[0]['host'], input_data.ravel())
                cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Set tensor addresses
            for i, inp in enumerate(self.inputs):
                self.context.set_tensor_address(inp['name'], int(inp['device']))
            for i, out in enumerate(self.outputs):
                self.context.set_tensor_address(out['name'], int(out['device']))
            
            # Execute inference
            success = self.context.execute_async_v3(stream_handle=self.stream.handle)
            if not success:
                logger.error("推理执行失败")
                return None
            
            # Copy output back to host
            output_data = None
            if len(self.outputs) > 0:
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
                self.stream.synchronize()
                output_data = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
            
            return output_data
            
        except Exception as e:
            logger.error(f"TensorRT推理失败: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if TensorRT optimization is available"""
        return self.tensorrt_available and self.engine is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get TensorRT engine information"""
        info = {
            "tensorrt_available": self.tensorrt_available,
            "engine_loaded": self.engine is not None,
            "precision": self.precision,
            "device": self.device
        }
        
        if self.engine:
            info.update({
                "num_inputs": len(self.inputs),
                "num_outputs": len(self.outputs),
                "input_shapes": [inp['shape'] for inp in self.inputs],
                "output_shapes": [out['shape'] for out in self.outputs]
            })
        
        return info
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.context:
                del self.context
                self.context = None
            
            if self.engine:
                del self.engine
                self.engine = None
            
            # Clean up CUDA memory
            import pycuda.driver as cuda
            for inp in self.inputs:
                if 'device' in inp:
                    inp['device'].free()
            for out in self.outputs:
                if 'device' in out:
                    out['device'].free()
            
            self.inputs = []
            self.outputs = []
            self.bindings = []
            
            logger.info("TensorRT资源清理完成")
            
        except Exception as e:
            logger.error(f"资源清理失败: {e}")
    
    def __del__(self):
        """Destructor"""
        self.cleanup()

# Factory function for TensorRT optimization
def create_tensorrt_engine(model_path: str, **kwargs) -> TensorRTEngine:
    """Create TensorRT engine with optimal settings for RTX 3060 Ti"""
    return TensorRTEngine(
        model_path=model_path,
        device=kwargs.get("device", "cuda"),
        precision=kwargs.get("precision", "fp16")
    )
