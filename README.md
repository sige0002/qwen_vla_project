# qwen_vla_project

Qwen3.5 を VLM Backbone として活用し、pi0.5 型の Vision-Language-Action (VLA) モデルを構築するプロジェクトです。

## 概要

- **目標**: Qwen3.5-4B をベースに、pi0.5 アーキテクチャに準拠した VLA モデルを実装する
- **VLM Backbone**: Qwen3.5-4B（Vision Encoder + LLM）
- **Action Expert**: VLM Backbone の hidden states を受け取り、ロボット操作用のアクションを出力
- **データセット**: [RoboCOIN](https://huggingface.co/datasets/) によるロボット操作データ

## プロジェクト構成

```
qwen_vla_project/
├── chat.py                  # Qwen3.5-4B の推論スクリプト
├── transformers-qwen3_5/    # Qwen3.5 対応の transformers カスタム実装
├── lerobot/                 # LeRobot（サブモジュール）
├── hf_qwen/                 # モデル重み（.gitignore で除外）
├── robocoin/                # データセット（.gitignore で除外）
├── VLA_BACKBONE_ANALYSIS.md # VLA Backbone 設計文書
└── pyproject.toml           # プロジェクト設定・依存関係
```

## セットアップ

### 前提条件

- Python >= 3.12
- [uv](https://docs.astral.sh/uv/)
- CUDA 対応 GPU

### インストール

```bash
# 依存関係のインストール
uv sync

# サブモジュールの取得
git submodule update --init --recursive
```

### モデルの準備

`hf_qwen/` ディレクトリにモデルの重みを配置してください。

```bash
# 例: Hugging Face Hub からダウンロード
huggingface-cli download Qwen/Qwen3.5-4B-Base --local-dir hf_qwen
```

## 使い方

### チャットスクリプト

```bash
# 通常モード（thinking あり）
uv run python chat.py

# thinking を無効にする
uv run python chat.py --no-think
```

対話形式で Qwen3.5-4B と会話できます。`quit` または `exit` で終了します。

## 依存関係

- [transformers](https://github.com/huggingface/transformers)
- [torch](https://pytorch.org/)
- [accelerate](https://github.com/huggingface/accelerate)
- [huggingface_hub](https://github.com/huggingface/huggingface-hub)
- [LeRobot](https://github.com/huggingface/lerobot)（サブモジュール）

## 参考資料

- [pi0.5 論文](https://arxiv.org/abs/2504.16054)
- [VLA Backbone 設計文書](VLA_BACKBONE_ANALYSIS.md)
