import torch
import onnx
from onnxsim import simplify

from src.trt.config import MODEL_NAME, BATCH_SIZE, ONNX_MODEL_FILE, CHECKPOINT, SIMPLIFYING,\
    GET_MODEL_FUNCTION, IMAGE_SIZE
from src.config import DEVICE


if __name__ == '__main__':
    model = GET_MODEL_FUNCTION(MODEL_NAME, checkpoint=CHECKPOINT)
    model.to(DEVICE)
    model.eval()
    print(model)

    dummy_input = torch.rand(BATCH_SIZE, 3, *IMAGE_SIZE, device=DEVICE)

    torch.onnx.export(model, dummy_input, ONNX_MODEL_FILE, verbose=True)
    print("Converted")

    # Check model
    onnx_model = onnx.load(ONNX_MODEL_FILE)
    onnx.checker.check_model(onnx_model)
    onnx.helper.printable_graph(onnx_model.graph)
    print("Checked")

    if SIMPLIFYING:
        onnx_model_sim, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(onnx_model_sim, ONNX_MODEL_FILE)
