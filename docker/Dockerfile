FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
        ultralytics \
        flask \
        gunicorn \
        pyyaml \
        Pillow

COPY train serve wsgi.py /opt/program/
RUN chmod +x /opt/program/train /opt/program/serve

ENV PATH="/opt/program:${PATH}"
WORKDIR /opt/program

