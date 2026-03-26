# Realman training pipeline - Makefile
#
# 使い方: make <target>
# 例: make train-act    # ACTポリシーで学習
#     make test-all     # 全テスト実行

# lerobot の uv 仮想環境（lerobot-train 等のコマンドを含む）
VENV := .lerobot_venv/bin
LEROBOT_TRAIN := $(VENV)/lerobot-train
LEROBOT_INFO := $(VENV)/lerobot-info
PYTHON := $(VENV)/python

# Dataset settings
DATASET_REPO := robocoin/Realman_RMC_AIDA_L_storage_block_basket
DATASET_ROOT := robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket
DATASET_OLD_REPO := robocoin/Realman_RMC_AIDA_L_storage_block_basket_old
DATASET_OLD_ROOT := robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old

# Common training defaults
COMMON_ARGS := \
	--dataset.repo_id=$(DATASET_REPO) \
	--dataset.root=$(DATASET_ROOT) \
	--policy.push_to_hub=false \
	--num_workers=0

.PHONY: help setup test-import test-dataset test-train test-all \
	train-act train-diffusion convert-v21-to-v30 \
	check-info check-dataset-features check-dataset-version

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-25s\033[0m %s\n", $$1, $$2}'

# === Setup ===

setup: ## Create lerobot venv and install dependencies
	uv venv .lerobot_venv --python 3.12
	uv pip install -e lerobot/ --python $(PYTHON)
	@echo "Setup complete. Run 'make check-info' to verify."

# === Verification targets ===

check-info: ## Show lerobot environment info
	$(LEROBOT_INFO)

test-import: ## Verify realman/bi_realman imports
	$(PYTHON) -c "from lerobot.robots.realman import RealmanFollowerConfig, RealmanFollower; print('realman import OK')"
	$(PYTHON) -c "from lerobot.robots.bi_realman import BiRealmanFollowerConfig, BiRealmanFollower; print('bi_realman import OK')"

test-dataset: ## Verify dataset loads correctly
	$(PYTHON) -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset(repo_id='$(DATASET_REPO)', root='$(DATASET_ROOT)'); print('Dataset loaded:', len(ds), 'frames'); print('Features:', list(ds.meta.info['features'].keys()))"

test-train: ## Run 2-step training smoke test (ACT)
	$(LEROBOT_TRAIN) \
		$(COMMON_ARGS) \
		--policy.type=act \
		--batch_size=2 \
		--steps=2 \
		--output_dir=outputs/test_smoke

test-all: test-import test-dataset test-train ## Run all verification tests

# === Training targets ===

train-act: ## Train with ACT policy (100k steps)
	$(LEROBOT_TRAIN) \
		$(COMMON_ARGS) \
		--policy.type=act \
		--batch_size=8 \
		--steps=100000 \
		--output_dir=outputs/act_realman

train-diffusion: ## Train with Diffusion policy (100k steps)
	$(LEROBOT_TRAIN) \
		$(COMMON_ARGS) \
		--policy.type=diffusion \
		--batch_size=8 \
		--steps=100000 \
		--output_dir=outputs/diffusion_realman

# === Dataset targets ===

convert-v21-to-v30: ## Convert old v2.1 dataset to v3.0
	$(PYTHON) lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
		--repo-id=$(DATASET_OLD_REPO) \
		--root=$(DATASET_OLD_ROOT) \
		--push-to-hub=false

check-dataset-version: ## Check dataset codebase version
	@cat $(DATASET_ROOT)/meta/info.json | python3 -m json.tool | grep codebase_version

check-dataset-features: ## Show dataset features
	@$(PYTHON) -c "import json; \
		info = json.load(open('$(DATASET_ROOT)/meta/info.json')); \
		[print(f'{k}: dtype={v[\"dtype\"]}, shape={v[\"shape\"]}') for k, v in info['features'].items()]"
