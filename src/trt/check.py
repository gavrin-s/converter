import numpy as np
import cv2
import torch
import cv2
import onnxruntime

import pycuda.autoinit

from src.trt.config import MODEL_NAME, CHECKPOINT, TRT_MODEL_ENGINE, ONNX_MODEL_FILE,\
    IMAGE_PATH, IMAGE_SIZE, GET_MODEL_FUNCTION

from src.trt.transforms import torch_preprocessing, onnx_preprocessing, trt_preprocessing
import src.trt.common as common
from src.config import DEVICE


def run_torch_inference(input_tensor: np.ndarray) -> np.ndarray:
    """
    Perform inference on Torch model.
    """
    model = GET_MODEL_FUNCTION(MODEL_NAME, checkpoint=CHECKPOINT)
    model.to(DEVICE)
    model.eval()

    input_tensor = torch_preprocessing(input_tensor)
    output_tensor = model(input_tensor)

    if isinstance(output_tensor, torch.Tensor):
        output_tensor = output_tensor[0].cpu().detach().numpy()
    elif isinstance(output_tensor, list) or isinstance(output_tensor, tuple):
        output_tensor = [output_tensor_[0].cpu().detach().numpy() for output_tensor_ in output_tensor]
    else:
        raise Exception(f"Unknown type = {type(output_tensor)}")

    return output_tensor


def run_trt_inference(input_tensor: np.ndarray) -> np.ndarray:
    """
    Perform inference on TensorRT model
    """
    input_tensor = trt_preprocessing(input_tensor)

    with common.get_engine(TRT_MODEL_ENGINE) as engine, \
            engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = input_tensor
        output_tensor = common.do_inference(context, bindings=bindings, inputs=inputs,
                                            outputs=outputs, stream=stream)
    return output_tensor


def run_onnx_inference(input_tensor: np.ndarray) -> np.ndarray:
    """
    Perform inference on ONNX model
    """
    input_tensor = onnx_preprocessing(input_tensor)

    sess_options = onnxruntime.RunOptions()
    sess_options.log_verbosity_level = 0

    ort_session = onnxruntime.InferenceSession(ONNX_MODEL_FILE)
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    ort_outputs = ort_session.run(None, ort_inputs)
    return ort_outputs


if __name__ == '__main__':
    if IMAGE_PATH is not None:
        input_tensor = cv2.imread(IMAGE_PATH)
    else:
        input_tensor = np.random.rand(*IMAGE_SIZE, 3)

    onnx_output = run_onnx_inference(input_tensor.copy())
    # print(f"ONNX output = {onnx_output}\n")

    torch_output = run_torch_inference(input_tensor.copy())
    # print(f"Torch output = {torch_output}\n")

    trt_output = run_trt_inference(input_tensor.copy())
    # print(f"TRT output = {trt_output}\n")

    print("Error for onnx:")
    if isinstance(torch_output, list):
        print([np.abs(torch_output_ - onnx_output_).sum()
              for torch_output_, onnx_output_ in zip(torch_output, onnx_output)])
    else:
        print(np.abs(torch_output - onnx_output[0]).sum())
    print()

    print("Error for trt:")
    if isinstance(torch_output, list):
        print([np.abs(torch_output_ - trt_output_).sum()
              for torch_output_, trt_output_ in zip(torch_output, trt_output)])
    else:
        print(np.abs(torch_output - trt_output[0]).sum())
