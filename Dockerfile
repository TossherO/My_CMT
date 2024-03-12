FROM mmdetection3d:latest

RUN pip install spconv-cu111==2.1.21 flash-attn==0.2.2
