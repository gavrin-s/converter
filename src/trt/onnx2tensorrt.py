import numpy as np

import pycuda.autoinit

import src.trt.common as common
from src.trt.config import ONNX_MODEL_FILE, TRT_MODEL_ENGINE, BATCH_SIZE, IMAGE_SIZE


if __name__ == '__main__':
    image = np.random.rand(BATCH_SIZE, *IMAGE_SIZE, 3)
    image = np.array(image, dtype=np.float32, order="C")
    print(image.shape)

    # Do inference with TensorRT
    with common.build_engine(ONNX_MODEL_FILE, TRT_MODEL_ENGINE) as engine,\
            engine.create_execution_context() as context:
        inputs, outputs, bindings, stream = common.allocate_buffers(engine)
        inputs[0].host = image
        trt_outputs = common.do_inference(context, bindings=bindings, inputs=inputs,
                                          outputs=outputs, stream=stream)
    print("Done")
