FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV LANG en_US.UTF-8 

RUN apt-get update -y && apt-get install -y \
    python3.9 \
    python3.9-dev \
    lshw \
    git \
    python3-pip \
    python3-matplotlib \
    gfortran \
    build-essential \
    libatlas-base-dev \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    python3-shapely \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/requirements.clip.txt \
    requirements/requirements.http.txt \
    requirements/requirements.doctr.txt \
    requirements/requirements.groundingdino.txt \
    requirements/requirements.sdk.http.txt \
    requirements/requirements.yolo_world.txt \
    requirements/_requirements.txt \
    ./

RUN python3.9 -m pip install --ignore-installed PyYAML && rm -rf ~/.cache/pip

RUN python3.9 -m pip install --upgrade pip  && python3.9 -m pip install \
    git+https://github.com/pypdfium2-team/pypdfium2 \
    -r _requirements.txt \
    -r requirements.clip.txt \
    -r requirements.http.txt \
    -r requirements.doctr.txt \
    -r requirements.groundingdino.txt \
    -r requirements.sdk.http.txt \
    -r requirements.yolo_world.txt \
    jupyterlab \
    --upgrade \
    && rm -rf ~/.cache/pip

RUN python3.9 -m pip uninstall --yes onnxruntime
RUN wget https://nvidia.box.com/shared/static/5dei4auhjh5ij7rmuvljmdy5q1en3bhf.whl -O onnxruntime_gpu-1.12.1-cp39-cp39-linux_aarch64.whl
RUN python3.9 -m pip install onnxruntime_gpu-1.12.1-cp39-cp39-linux_aarch64.whl "opencv-python-headless>4" \
    && rm -rf ~/.cache/pip \
    && rm onnxruntime_gpu-1.12.1-cp39-cp39-linux_aarch64.whl

ENV VERSION_CHECK_MODE=continuous
ENV PROJECT=roboflow-platform
ENV ORT_TENSORRT_FP16_ENABLE=1
ENV ORT_TENSORRT_ENGINE_CACHE_ENABLE=1
ENV CORE_MODEL_SAM_ENABLED=False
ENV PROJECT=roboflow-platform
ENV NUM_WORKERS=1
ENV HOST=0.0.0.0
ENV PORT=9001
ENV OPENBLAS_CORETYPE=ARMV8 
ENV LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
ENV WORKFLOWS_STEP_EXECUTION_MODE=local
ENV WORKFLOWS_MAX_CONCURRENT_STEPS=1
ENV API_LOGGING_ENABLED=True

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt

CMD ["python", "main.py"]