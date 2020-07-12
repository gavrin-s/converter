FROM nvcr.io/nvidia/pytorch:19.10-py3

RUN pip install onnx-simplifier==0.2.10
WORKDIR /workspace/converter

CMD [ "/bin/bash" ]