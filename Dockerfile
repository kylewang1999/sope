FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 LC_ALL=C.UTF-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MUJOCO_GL=egl \
    D4RL_SUPPRESS_IMPORT_ERROR=1 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1

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
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/root/.mujoco/mujoco210/bin

# ---------- Python pkgs (order matters) ----------
ENV TORCH_CUDA_INDEX_URL=https://download.pytorch.org/whl/cu121

RUN python3 -m pip install --upgrade pip && \
    # 0) Pin NumPy FIRST and keep it
    python3 -m pip install --no-cache-dir "numpy==1.22.4" && \
    # 1) Tooling + mujoco-py + gym (needs Cython<3)
    python3 -m pip install --no-cache-dir "cython<3" wheel setuptools six && \
    python3 -m pip install --no-cache-dir mujoco-py==2.1.2.14 "gym==0.23.1" pygame einops && \
    # 2) PyTorch that satisfies CleanDiffuser's torch<2.3
    python3 -m pip install --no-cache-dir \
      --index-url ${TORCH_CUDA_INDEX_URL} \
      --extra-index-url https://pypi.org/simple \
      "torch==2.2.2+cu121" "torchvision==0.17.2+cu121" && \
    # 3) JAX + Flax + Optax WITHOUT letting pip upgrade NumPy
    python3 -m pip install --no-cache-dir --no-deps \
      jax==0.4.30 jaxlib==0.4.30 jax-cuda12-pjrt==0.4.30 jax-cuda12-plugin==0.4.30 \
      flax==0.8.5 optax==0.2.4 chex==0.1.90 orbax-checkpoint==0.6.4 && \
    #    Explicit deps compatible with NumPy 1.22.x (incl. flax/chex/orbax deps)
    python3 -m pip install --no-cache-dir \
      msgpack==1.0.8 ml_dtypes==0.5.3 opt_einsum==3.4.0 etils==1.5.2 \
      absl-py==2.3.1 tensorstore==0.1.69 dm-tree==0.1.8 \
      rich==14.2.0 humanize==4.13.0 nest-asyncio==1.6.0 toolz==1.0.0 && \
    # 4) d4rl (best-effort)
    python3 -m pip install --no-cache-dir "git+https://github.com/Farama-Foundation/d4rl@master#egg=d4rl" || true

# ---------- Register pip CUDA libs (cuDNN 8, etc.) with the dynamic linker ----------
RUN python3 -m pip install --no-cache-dir \
    nvidia-cudnn-cu12==8.9.2.26 \
    nvidia-cublas-cu12==12.1.3.1 \
    nvidia-cusolver-cu12==11.4.5.107 \
    nvidia-cusparse-cu12==12.1.0.106 \
    nvidia-cufft-cu12==11.0.2.54 \
    nvidia-cuda-runtime-cu12==12.1.105 \
    nvidia-cuda-nvrtc-cu12==12.1.105 \
    nvidia-cuda-cupti-cu12==12.1.105 \
    nvidia-nvtx-cu12==12.1.105 \
    nvidia-nvjitlink-cu12==12.4.127 \
    nvidia-nccl-cu12==2.19.3 \
    nvidia-cuda-nvcc-cu12==12.4.131 && \
    python3 - <<'PY' | tee /etc/ld.so.conf.d/00-pip-nvidia.conf >/dev/null && ldconfig
import importlib, pathlib
mods = [
  "nvidia.cudnn","nvidia.cublas","nvidia.cusolver","nvidia.cusparse","nvidia.cufft",
  "nvidia.cuda_runtime","nvidia.cuda_nvrtc","nvidia.cuda_cupti","nvidia.nvjitlink","nvidia.nccl",
]
for m in mods:
    try:
        lib = pathlib.Path(importlib.import_module(m).__file__).parent / "lib"
        if lib.is_dir():
            print(str(lib))
    except Exception:
        pass
PY

# Expose nvcc/ptxas on PATH for clarity
RUN python3 - <<'PY'
import importlib, pathlib
b=(pathlib.Path(importlib.import_module("nvidia.cuda_nvcc").__file__).parent/"bin")
for exe in ("ptxas","nvcc"):
    src=b/exe
    dst=pathlib.Path("/usr/local/bin")/exe
    if src.exists():
        try:
            if dst.exists() or dst.is_symlink(): dst.unlink()
        except Exception: pass
        dst.symlink_to(src)
print("NVCC bin:", b)
PY

# ---------- CleanDiffuser (no-deps) + its runtime deps ----------
ARG CDIFF_COMMIT=05f17fc9dbeae7c19a5e264632c9ae9aaac5994e
RUN python3 -m pip install --no-cache-dir --no-deps \
      "git+https://github.com/CleanDiffuserTeam/CleanDiffuser.git@${CDIFF_COMMIT}" && \
    python3 -m pip install --no-cache-dir \
      zarr==2.16.1 "av>=12.2.0" dill>=0.3.8 "matplotlib<=3.7.5" "numba<0.60.0" \
      hydra-core imagecodecs opencv-python pymunk "scikit-image<0.23.0" "shapely<2.0.0" wandb && \
    # Re-affirm NumPy pin at the end (belt-and-suspenders)
    python3 -m pip install --no-cache-dir --force-reinstall "numpy==1.22.4"

# ---------- Smoke check (CPU-only to avoid GPU probing during build) ----------
RUN JAX_PLATFORMS=cpu python3 - <<'PY'
import numpy as np, pkgutil, torch, jax
print("numpy", np.__version__)
print("torch", torch.__version__, "cudnn", torch.backends.cudnn.version(), "cuda_available_at_build?", torch.cuda.is_available())
print("jax", jax.__version__, "| cpu devices:", jax.devices('cpu'))
for m in ["gym","mujoco_py","cleandiffuser"]:
    print(f"{m}: {'found' if pkgutil.find_loader(m) else 'MISSING'}")
PY

WORKDIR /workspace
CMD ["/bin/bash"]
