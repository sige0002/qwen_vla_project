# 学習パイプラインガイド

RoboCOIN データセットを用いた LeRobot の学習パイプライン手順書。
全コマンドは lerobot CLI (`lerobot-train`, `lerobot-info` 等) を使用する。

> トラブルが起きたら → [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

---

## 1. 環境構築

```bash
# lerobot 仮想環境の作成とインストール
make setup

# 確認
source .lerobot_venv/bin/activate
lerobot-info
```

CUDA 版 PyTorch は `lerobot/pyproject.toml` の `[tool.uv.index]` で制御済み。
別の CUDA バージョンを使う場合は `cu130` を変更する（例: `cu124`, `cu126`）。

```bash
# CUDA 確認
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

---

## 2. データセット

### 2つのデータセット

| 名前 | パス | バージョン | features | 拡張フィールド |
|---|---|---|---|---|
| 既存 | `Realman_RMC_AIDA_L_storage_block_basket/` | v3.0 | 10 | なし |
| old | `Realman_RMC_AIDA_L_storage_block_basket_old/` | v2.1→v3.0 | 26 | あり (16列) |

### features 確認

```bash
make check-dataset-features
```

出力例 (既存 v3.0):
```
observation.images.cam_head_rgb: dtype=video, shape=[480, 640, 3]
observation.images.cam_left_wrist_rgb: dtype=video, shape=[480, 640, 3]
observation.images.cam_right_wrist_rgb: dtype=video, shape=[480, 640, 3]
observation.state: dtype=float32, shape=[28]
action: dtype=float32, shape=[28]
timestamp / frame_index / episode_index / index / task_index
```

### action (28次元) の内訳

| 次元 | 内容 |
|---|---|
| 0--6 | 右腕 7関節 (rad) |
| 7 | 右グリッパー |
| 8--13 | 右手先 位置 (3) + 姿勢 (3) |
| 14--20 | 左腕 7関節 (rad) |
| 21 | 左グリッパー |
| 22--27 | 左手先 位置 (3) + 姿勢 (3) |

カメラ 3台: `cam_head_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb` (480x640, AV1)

---

## 3. データセット変換 (v2.1 → v3.0)

v2.1 データを v3.0 に変換し、parquet のスキーマを修正する。

```bash
# Makefile で一発
make convert-v21-to-v30

# 手動で実行する場合:
python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=robocoin/Realman_RMC_AIDA_L_storage_block_basket_old \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old \
    --push-to-hub=false

python scripts/fix_parquet_list_scalars.py \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old
```

### v2.1 → v3.0 の変更点

| 項目 | v2.1 | v3.0 |
|---|---|---|
| data | `data/chunk-000/episode_NNNNNN.parquet` | `data/chunk-NNN/file-NNN.parquet` |
| video | `videos/chunk-000/CAMERA/episode_NNNNNN.mp4` | `videos/{key}/chunk-NNN/file-NNN.mp4` |
| episodes | `meta/episodes.jsonl` | `meta/episodes/chunk-000/file_000.parquet` |
| tasks | `meta/tasks.jsonl` | `meta/tasks.parquet` |

### RoboCOIN 拡張フィールド

変換スクリプトは拡張フィールド (subtask_annotation, scene_annotation, eef_*, gripper_* 等) を保持する。
ただし `scene_annotation` (shape=[1]) が parquet 上で `list<int32>` のまま残るため、
`scripts/fix_parquet_list_scalars.py` でスカラーに変換する必要がある。

---

## 4. 学習

### smoke test

```bash
make test-all          # import → dataset → train (2step) を順に検証
```

### 本番学習

```bash
make train-act         # ACT ポリシー (100k steps)
make train-diffusion   # Diffusion ポリシー (100k steps)
```

### 直接コマンド

```bash
lerobot-train \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=4 \
    --batch_size=8 \
    --steps=100000 \
    --output_dir=outputs/act_realman
```

### 主要オプション

| オプション | 説明 | 備考 |
|---|---|---|
| `--dataset.repo_id` | データセット識別子 | ローカルでも Hub 形式の名前が必要 |
| `--dataset.root` | ローカルパス | `meta/`, `data/`, `videos/` を含むディレクトリ |
| `--policy.type` | ポリシー種別 | `act`, `diffusion` 等 |
| `--policy.push_to_hub=false` | Hub push 無効化 | 省略すると repo_id 未指定エラー |
| `--num_workers` | DataLoader ワーカー数 | SHM 不足時は `0` |
| `--batch_size` | バッチサイズ | GPU メモリに応じて調整 |
| `--steps` | 学習ステップ数 | |
| `--output_dir` | 出力先 | |

---

## 5. 可視化

```bash
lerobot-dataset-viz \
    --repo-id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --episode-index=0 \
    --num-workers=0 \
    --save=1 \
    --output-dir=outputs/viz_v30
```

`outputs/viz_v30/*.rrd` を Rerun Viewer で開いて確認。

---

## 6. Realman Config

### ファイル構成

```
lerobot/src/lerobot/robots/
├── realman/          # 単腕 (realman_follower)
└── bi_realman/       # 双腕 (bi_realman_follower)
```

### 設計上の注意

- `robot.type` は学習に **不要**。学習はデータセットの features から入出力次元を自動決定する
- robot config はデプロイ (推論→実機制御) フェーズで必要
- 学習データ (RoboCOIN): 28次元、BiRealmanFollowerConfig: 16次元 — デプロイ時にリネーム・次元変換が必要

---

## 7. lerobot CLI 一覧

| コマンド | 説明 |
|---|---|
| `lerobot-train` | ポリシーの学習 |
| `lerobot-eval` | ポリシーの評価 |
| `lerobot-info` | 環境情報の表示 |
| `lerobot-dataset-viz` | データセットの可視化 |
| `lerobot-record` | ロボット操作の記録 |
| `lerobot-replay` | 記録の再生 |
| `lerobot-calibrate` | キャリブレーション |
| `lerobot-teleoperate` | テレオペレーション |
| `lerobot-find-cameras` | カメラ検出 |
| `lerobot-find-port` | ポート検出 |

---

*最終更新: 2026-03-26*
