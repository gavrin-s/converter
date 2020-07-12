# Converter PyTorch -> ONNX -> TensorRT

### Run docker container
```
 1. chmod +x run.sh
 2. ./run.sh
```

### Preparation
1. In `src/models` define `get_model` function to get model.
2. In `src.trt.config` define all settings.

### Convert model
1. Run `PYTHONPATH=. python src/trt/torch2onnx.py` for convert torch -> onnx
2. Run `PYTHONPATH=. python src/trt/onnx2trt.py` for convert onnx -> tensorrt
3. Run `PYTHONPATH=. python src/trt/check.py` for check conversion
