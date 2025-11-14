ARG CUDA_VERSION=12.8.0
ARG PYTHON_VERSION=3.11.13
FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu22.04

ARG PYTHON_VERSION
ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

# RDMA Python UV
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget git fish libnuma-dev \
    build-essential cmake \
    libibverbs1 libibverbs-dev ibverbs-providers rdma-core \
    vim tmux htop tree silversearcher-ag lsof unzip \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (conda)
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && /opt/conda/bin/conda clean -ay

ENV PATH="/opt/conda/bin:$PATH"

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | env UV_UNMANAGED_INSTALL=/usr/local/bin sh

WORKDIR /root/RAGBoost
RUN uv venv --python ${PYTHON_VERSION}

ENV PATH="/root/RAGBoost/.venv/bin:$PATH"

# Make sure to install ninja to enable fast builds
RUN uv pip install torch==2.8.0+cu128 --extra-index-url https://download.pytorch.org/whl/cu128 wheel packaging ninja \
    && uv cache clean
    
# Install sglang
RUN git clone -b v0.5.5.post2 https://github.com/sgl-project/sglang.git \
    && cd sglang \
    && uv pip install -e "python" \
    && uv cache clean

# Install faiss-gpu-cuvs using conda with rapidsai channel
RUN conda install -y -c rapidsai -c conda-forge -c pytorch pytorch::faiss-gpu-cuvs \
    && conda clean -ay

# Install from RAGBoost source
COPY . /root/RAGBoost

RUN uv pip install -e . -v \
    && uv cache clean

CMD ["sleep", "infinity"]