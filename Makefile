IMAGE ?= stitch-gpu
DOCKERFILE ?= Dockerfile
UID := $(shell id -u)
GID := $(shell id -g)

.PHONY: build sanity shell download refresh clean

build:
	docker build -f $(DOCKERFILE) -t $(IMAGE) .

sanity:
	docker run --rm -t --gpus all \
	  --user $(UID):$(GID) \
	  -e MUJOCO_GL=egl \
	  -v "$$PWD":/workspace -w /workspace \
	  $(IMAGE) python3 opelab/sanity.py --steps 5

shell:
	docker run --rm -it --gpus all --user $(UID):$(GID) \
	  -e MUJOCO_GL=egl \
	  -v "$$PWD":/workspace -w /workspace \
	  $(IMAGE) bash

ZIP_URL ?= https://drive.google.com/file/d/13uoTd6Yw5BIM7UUxrLGErMrIta7bl27K/view?usp=sharing
STAMP   ?= .cache/download.stamp

download:
	docker run --rm -t --gpus all --user $(UID):$(GID) \
	  -e MUJOCO_GL=egl \
	  -e ZIP_LOCAL=$(ZIP_LOCAL) -e ZIP_URL=$(ZIP_URL) -e ZIP_ID=$(ZIP_ID) \
	  -v "$$PWD":/workspace -w /workspace \
	  $(IMAGE) bash -lc 'bash scripts/download_assets.sh'

refresh:
	docker run --rm -t --gpus all --user $(UID):$(GID) \
	  -e MUJOCO_GL=egl \
	  -e ZIP_URL=$(ZIP_URL) \
	  -e ZIP_LOCAL=$(ZIP_LOCAL) \
	  -e STAMP=$(STAMP) \
	  -e REFRESH=1 \
	  -v "$$PWD":/workspace -w /workspace \
	  $(IMAGE) bash -lc 'bash scripts/download_assets.sh'

clean:
	@echo "[clean] removing downloaded assets (models/policies/dataset) …"
	@set -e; \
	for d in \
	  opelab/examples/d4rl/models \
	  opelab/examples/d4rl/policy \
	  opelab/examples/gym/models  \
	  opelab/examples/gym/policy  \
	  opelab/examples/diffusion_policy/policy \
	  dataset \
	; do \
	  if [ -d "$$d" ]; then \
	    find "$$d" -mindepth 1 -maxdepth 1 -exec rm -rf {} +; \
	  fi; \
	done; \
	rm -f .cache/download.stamp || true
	@echo "[clean] done."
