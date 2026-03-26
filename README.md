# qwen_vla_project

Qwen3.5 を VLM Backbone に、pi0.5 型の Vision-Language-Action (VLA) モデルを構築するプロジェクト。
RoboCOIN データセット + LeRobot で Realman ロボットの学習パイプラインを実装済み。

## プロジェクト構成

```
qwen_vla_project/
├── Makefile                         # 学習・検証コマンド集（make help で一覧）
├── pyproject.toml                   # Qwen 環境の依存定義
├── scripts/
│   ├── chat.py                      # Qwen3.5-4B チャット推論
│   ├── train_realman.py             # 学習ラッパー（monkey-patch、通常不要）
│   └── fix_parquet_list_scalars.py  # Parquet スキーマ修正ツール
├── docs/
│   ├── PIPELINE_GUIDE.md            # 学習パイプラインの手順書
│   ├── TROUBLESHOOTING.md           # トラブルシューティング集
│   ├── Plan.md                      # プロジェクト計画
│   └── VLA_BACKBONE_ANALYSIS.md     # VLA Backbone 設計文書
├── lerobot/                         # LeRobot（git submodule, Realman 対応済み）
├── transformers-qwen3_5/            # Qwen3.5 対応 transformers カスタム実装
├── hf_qwen/                         # モデル重み（gitignored）
├── robocoin/                        # データセット（gitignored）
└── outputs/                         # 学習出力（gitignored）
```

## 2つの Python 環境

本プロジェクトには用途別に **2つの仮想環境** がある。

| 環境 | 用途 | 仮想環境 | activate |
|---|---|---|---|
| **Qwen** | VLM 推論・VLA 開発 | `.venv/` | `source .venv/bin/activate` |
| **LeRobot** | データセット変換・学習 | `.lerobot_venv/` | `source .lerobot_venv/bin/activate` |

```bash
# === Qwen 環境（VLM 推論）===
source .venv/bin/activate
python scripts/chat.py

# === LeRobot 環境（学習パイプライン）===
source .lerobot_venv/bin/activate
lerobot-train --dataset.repo_id=... --policy.type=act ...

# activate せずに直接実行する場合
.lerobot_venv/bin/lerobot-train ...

# Makefile 経由なら activate 不要（パスを自動解決）
make train-act
```

## セットアップ

### 前提条件

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- CUDA 対応 GPU

### クローンと初期セットアップ

```bash
git clone https://github.com/sige0002/qwen_vla_project.git
cd qwen_vla_project
git submodule update --init --recursive     # lerobot サブモジュール取得
```

### Qwen 環境

```bash
uv sync                                    # .venv/ に依存をインストール
huggingface-cli download Qwen/Qwen3.5-4B-Base --local-dir hf_qwen
```

### LeRobot 環境

```bash
make setup
# 内部で以下を実行:
#   uv venv .lerobot_venv --python 3.12
#   uv pip install -e lerobot/ --python .lerobot_venv/bin/python
```

## 使い方

### Qwen チャット

```bash
source .venv/bin/activate
python scripts/chat.py              # thinking あり
python scripts/chat.py --no-think   # thinking なし
```

### LeRobot 学習

```bash
source .lerobot_venv/bin/activate

# smoke test（2ステップ）
make test-all

# ACT ポリシーで本番学習
make train-act

# 全コマンド一覧
make help
```

詳細は [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) を参照。

## 参考資料

- [pi0.5 論文](https://arxiv.org/abs/2504.16054)
- [VLA Backbone 設計文書](docs/VLA_BACKBONE_ANALYSIS.md)
- [学習パイプライン手順書](docs/PIPELINE_GUIDE.md)
- [トラブルシューティング](docs/TROUBLESHOOTING.md)
