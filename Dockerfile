FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# ---------- OS deps ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common git wget curl unzip ca-certificates \
    build-essential patchelf \
    libosmesa6-dev libgl1-mesa-dev libglfw3 libglew-dev libegl1 \
    libxrender1 libxext6 libxtst6 libxrandr2 libxi6 \
    ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# ---------- Python 3.9 ----------
RUN add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
      python3.9 python3.9-dev python3.9-venv python3.9-distutils && \
    rm -rf /var/lib/apt/lists/* && \
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.9 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 2 && \
    update-alternatives --install /usr/bin/python  python  /usr/bin/python3.9 2

# ---------- MuJoCo for mujoco-py ----------
ARG MUJOCO_URL=https://github.com/google-deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
RUN mkdir -p /root/.mujoco && cd /root/.mujoco && \
    wget -qO mujoco.tar.gz ${MUJOCO_URL} && tar -xzf mujoco.tar.gz && rm mujoco.tar.gz
ENV MUJOCO_PY_MUJOCO_PATH=/root/.mujoco/mujoco210
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV MUJOCO_GL=egl

# Make torch.load permissive with older checkpoints (optional)
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

# ---------- Python pkgs (mujoco-py + gym + Torch + JAX) ----------
ENV TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --upgrade pip && \
    # keep mujoco-py happy
    python3 -m pip install --no-cache-dir "cython<3" wheel setuptools six && \
    python3 -m pip install --no-cache-dir mujoco-py==2.1.2.14 "gym==0.23.1" pygame einops && \
    # Torch that satisfies CleanDiffuser's torch<2.3 requirement
    python3 -m pip install --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url https://pypi.org/simple \
      "torch==2.2.2+cu121" "torchvision==0.17.2+cu121" && \
    # JAX (works in this env alongside numpy==1.22.4)
    python3 -m pip install --no-cache-dir "jax[cuda12]==0.4.30" "jaxlib==0.4.30" flax optax && \
    # d4rl (best-effort)
    python3 -m pip install --no-cache-dir "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl" || true

# ---------- CleanDiffuser (no-deps) + its runtime deps ----------
ARG CDIFF_COMMIT=05f17fc9dbeae7c19a5e264632c9ae9aaac5994e
RUN python3 -m pip install --no-cache-dir --no-deps \
      "git+https://github.com/CleanDiffuserTeam/CleanDiffuser.git@${CDIFF_COMMIT}" && \
    python3 -m pip install --no-cache-dir \
      "numpy==1.22.4" \
      "zarr==2.16.1" \
      "av>=12.2.0" \
      "dill>=0.3.8" \
      "matplotlib<=3.7.5" \
      "numba<0.60.0" \
      hydra-core imagecodecs opencv-python pymunk \
      "scikit-image<0.23.0" "shapely<2.0.0" wandb

# ---------- Smoke check without importing heavy GPU libs ----------
RUN python3 - <<'PY'
import sys, pkgutil
print("Python:", sys.version.split()[0])
for m in ["gym","mujoco_py","jax","torch","torchvision","cleandiffuser"]:
    print(f"{m}: {'found' if pkgutil.find_loader(m) else 'MISSING'}")
PY

WORKDIR /workspace
CMD ["/bin/bash"]
