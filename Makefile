# Realman training pipeline - Makefile
#
# 使い方: make <target>
# 例: make train-act    # ACTポリシーで学習
#     make test-import  # importテスト

PYTHON := .venv/bin/python
TRAIN_SCRIPT := train_realman.py

# Dataset settings
DATASET_REPO := robocoin/Realman_RMC_AIDA_L_storage_block_basket
DATASET_ROOT := robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket

# Common training defaults
COMMON_ARGS := \
	--dataset.repo_id=$(DATASET_REPO) \
	--dataset.root=$(DATASET_ROOT) \
	--policy.push_to_hub=false \
	--num_workers=0

.PHONY: test-import test-dataset test-train train-act train-diffusion help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# === Verification targets ===

test-import: ## Verify realman/bi_realman imports
	$(PYTHON) -c "import sys; sys.path.insert(0, 'lerobot/src'); \
		from lerobot.robots.realman import RealmanFollowerConfig, RealmanFollower; \
		print('realman import OK')"
	$(PYTHON) -c "import sys; sys.path.insert(0, 'lerobot/src'); \
		from lerobot.robots.bi_realman import BiRealmanFollowerConfig, BiRealmanFollower; \
		print('bi_realman import OK')"

test-dataset: ## Verify dataset loads correctly
	$(PYTHON) -c "import sys; sys.path.insert(0, 'lerobot/src'); \
		from lerobot.datasets.lerobot_dataset import LeRobotDataset; \
		ds = LeRobotDataset(repo_id='$(DATASET_REPO)', root='$(DATASET_ROOT)'); \
		print('Dataset loaded:', len(ds), 'frames'); \
		print('Features:', list(ds.meta.info['features'].keys()))"

test-train: ## Run 2-step training smoke test (ACT)
	$(PYTHON) $(TRAIN_SCRIPT) \
		$(COMMON_ARGS) \
		--policy.type=act \
		--batch_size=2 \
		--steps=2 \
		--output_dir=outputs/test_smoke

test-all: test-import test-dataset test-train ## Run all verification tests

# === Training targets ===

train-act: ## Train with ACT policy (default)
	$(PYTHON) $(TRAIN_SCRIPT) \
		$(COMMON_ARGS) \
		--policy.type=act \
		--batch_size=8 \
		--steps=100000 \
		--output_dir=outputs/act_realman

train-diffusion: ## Train with Diffusion policy
	$(PYTHON) $(TRAIN_SCRIPT) \
		$(COMMON_ARGS) \
		--policy.type=diffusion \
		--batch_size=8 \
		--steps=100000 \
		--output_dir=outputs/diffusion_realman

# === Dataset targets ===

check-dataset-version: ## Check dataset codebase version
	@cat $(DATASET_ROOT)/meta/info.json | python3 -m json.tool | grep codebase_version

check-dataset-features: ## Show dataset features
	@$(PYTHON) -c "import json; \
		info = json.load(open('$(DATASET_ROOT)/meta/info.json')); \
		[print(f'{k}: dtype={v[\"dtype\"]}, shape={v[\"shape\"]}') for k, v in info['features'].items()]"
