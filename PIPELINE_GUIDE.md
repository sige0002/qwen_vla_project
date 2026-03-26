# RoboCOIN データセット → LeRobot 学習 パイプラインガイド

本ドキュメントは、RoboCOINデータセットを用いたLeRobotの学習パイプラインをまとめたものです。
全コマンドは lerobot の公式 CLI (`lerobot-train`, `lerobot-info` 等) を使用します。

---

## 0. 環境構築

### uv で lerobot 仮想環境を作成

```bash
cd /workspace/qwen_vla_project

# lerobot 用の仮想環境を作成
uv venv .lerobot_venv --python 3.12

# lerobot をエディタブルインストール
uv pip install -e lerobot/ --python .lerobot_venv/bin/python

# CUDA 版 PyTorch が必要な場合（Jetson / GPU 環境）:
# uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
#   --python .lerobot_venv/bin/python

# コマンド確認
.lerobot_venv/bin/lerobot-info --help
```

**重要**: lerobot の `pyproject.toml` には `torch`, `torchvision`, `av` 等が全て含まれている。
aarch64 (Jetson) では `torchcodec` が自動除外され、`pyav` バックエンドにフォールバックする。
正しくインストールすれば monkey-patch なしで `lerobot-train` が動作する。

### 環境確認

```bash
# lerobot の情報を表示
.lerobot_venv/bin/lerobot-info

# 期待される出力例:
# - LeRobot version: 0.5.0
# - Platform: Linux-...-aarch64-with-glibc2.39
# - PyTorch version: 2.10.0
# - lerobot scripts: ['lerobot-train', 'lerobot-eval', ...]
```

---

## 1. データセット変換（v2.1 → v3.0）

### バージョン確認

```bash
# v2.1 のデータセット (old) の info.json を確認
cat robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old/meta/info.json \
    | python3 -m json.tool | grep codebase_version
# → "codebase_version": "v2.1"

# v3.0 に変換済みのデータセットを確認
cat robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket/meta/info.json \
    | python3 -m json.tool | grep codebase_version
# → "codebase_version": "v3.0"
```

### v2.1 → v3.0 の主な変更点

| 項目 | v2.1 | v3.0 |
|---|---|---|
| data_path | `data/chunk-000/episode_000000.parquet` | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| video_path | `videos/chunk-000/CAMERA/episode_000000.mp4` | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |
| episodes | `meta/episodes.jsonl` | `meta/episodes/chunk-000/file_000.parquet` |
| tasks | `meta/tasks.jsonl` | `meta/tasks.parquet` |

### 変換コマンド（ローカル実行、Hub push なし）

```bash
.lerobot_venv/bin/python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=robocoin/Realman_RMC_AIDA_L_storage_block_basket_old \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old \
    --push-to-hub=false
```

**実行結果 (2026-03-26):**
```
Converting info from ...old to ...old_v30
Converting tasks from ...old to ...old_v30
Converting data files from 50 episodes
convert data files: 100%|██████████| 50/50
Converting videos from ...old to ...old_v30
convert videos: 100%|██████████| 50/50
Converting episodes metadata ...
```

変換後:
- 元データは `*_old` にバックアップされる
- 変換済みデータが元のパスに置かれる
- `meta/info.json` の `codebase_version` が `"v3.0"` に更新

### Hub 上のデータセットを変換する場合

```bash
.lerobot_venv/bin/python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=your_org/your_dataset \
    --push-to-hub=true
```

### RoboCOIN 固有フィールドの注意点

RoboCOIN v2.1 データセットには標準 LeRobot にない拡張フィールドが含まれる:
- `subtask_annotation` (shape=[5], dtype=int32)
- `scene_annotation` (shape=[1], dtype=int32)
- `eef_sim_pose_state/action`, `eef_direction_*`, `eef_velocity_*`, `eef_acc_mag_*`
- `gripper_open_scale_*`, `gripper_mode_*`, `gripper_activity_*`

これらのフィールドはパーケットのスキーマ不整合（`list<int32>` vs `int32`）を引き起こす場合がある。
既に v3.0 で提供されているデータセットを使うか、変換後にスキーマを修正する必要がある。

### action feature（28次元）の構成

| 次元 | フィールド名 | 説明 |
|---|---|---|
| 0--6 | `right_arm_joint_1_rad` 〜 `right_arm_joint_7_rad` | 右腕関節角度（ラジアン） |
| 7 | `right_gripper_open` | 右グリッパー開閉 |
| 8--13 | `right_eef_pos_{x,y,z}_m`, `right_eef_rot_euler_{x,y,z}_rad` | 右手先位置・姿勢 |
| 14--20 | `left_arm_joint_1_rad` 〜 `left_arm_joint_7_rad` | 左腕関節角度（ラジアン） |
| 21 | `left_gripper_open` | 左グリッパー開閉 |
| 22--27 | `left_eef_pos_{x,y,z}_m`, `left_eef_rot_euler_{x,y,z}_rad` | 左手先位置・姿勢 |

カメラ3台: `cam_head_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb` (480x640, AV1)

---

## 2. Realman Config の実装

### ファイル構成

```
lerobot/src/lerobot/robots/
├── realman/
│   ├── __init__.py             # RealmanFollowerConfig, RealmanFollower
│   ├── config_realman.py       # @RobotConfig.register_subclass("realman_follower")
│   └── realman.py              # 単腕 Robot 実装
├── bi_realman/
│   ├── __init__.py             # BiRealmanFollowerConfig, BiRealmanFollower
│   ├── config_bi_realman.py    # @RobotConfig.register_subclass("bi_realman_follower")
│   └── bi_realman.py           # 双腕 Robot 実装
└── utils.py                    # make_robot_from_config() に分岐追加
```

### 登録（utils.py への追加）

```python
elif config.type == "realman_follower":
    from .realman import RealmanFollower
    return RealmanFollower(config)
elif config.type == "bi_realman_follower":
    from .bi_realman import BiRealmanFollower
    return BiRealmanFollower(config)
```

### 次元の注意

- 学習データ（RoboCOIN）: 28次元（左右各7関節 + 1グリッパー + 6次元EEF）
- BiRealmanFollowerConfig: 16次元（左右各7関節 + 1グリッパー）
- デプロイ時にはキー名リネームと次元変換が必要

---

## 3. lerobot コマンドによる一連の実行例

以下は全て 2026-03-26 に実際に実行して検証したコマンドと結果です。

### 3.1 環境情報の確認

```bash
.lerobot_venv/bin/lerobot-info
```

```
- LeRobot version: 0.5.0
- Platform: Linux-6.14.0-1015-nvidia-aarch64-with-glibc2.39
- Python version: 3.12.3
- PyTorch version: 2.10.0
- lerobot scripts: ['lerobot-calibrate', 'lerobot-dataset-viz', 'lerobot-edit-dataset',
  'lerobot-eval', 'lerobot-find-cameras', 'lerobot-find-joint-limits', 'lerobot-find-port',
  'lerobot-imgtransform-viz', 'lerobot-info', 'lerobot-record', 'lerobot-replay',
  'lerobot-setup-can', 'lerobot-setup-motors', 'lerobot-teleoperate', 'lerobot-train',
  'lerobot-train-tokenizer']
```

### 3.2 データセットの features 確認

```bash
.lerobot_venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    repo_id='robocoin/Realman_RMC_AIDA_L_storage_block_basket',
    root='robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket'
)
print(f'Frames: {len(ds)}')
print(f'Episodes: {ds.meta.info[\"total_episodes\"]}')
print(f'FPS: {ds.meta.info[\"fps\"]}')
for k, v in ds.meta.info['features'].items():
    print(f'  {k}: dtype={v[\"dtype\"]}, shape={v[\"shape\"]}')
"
```

```
Frames: 19083
Episodes: 50
FPS: 30
  observation.images.cam_head_rgb: dtype=video, shape=[480, 640, 3]
  observation.images.cam_left_wrist_rgb: dtype=video, shape=[480, 640, 3]
  observation.images.cam_right_wrist_rgb: dtype=video, shape=[480, 640, 3]
  observation.state: dtype=float32, shape=[28]
  action: dtype=float32, shape=[28]
  timestamp: dtype=float32, shape=[1]
  frame_index: dtype=int64, shape=[1]
  episode_index: dtype=int64, shape=[1]
  index: dtype=int64, shape=[1]
  task_index: dtype=int64, shape=[1]
```

### 3.3 v2.1 → v3.0 データセット変換

```bash
# old (v2.1) データを v3.0 に変換
.lerobot_venv/bin/python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=robocoin/Realman_RMC_AIDA_L_storage_block_basket_old \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old \
    --push-to-hub=false
```

```
Converting info ...
Converting tasks ...
Converting data files from 50 episodes
convert data files: 100%|██████████| 50/50
Converting videos ...
convert videos: 100%|██████████| 50/50
Converting episodes metadata ...
```

### 3.4 学習（ACT ポリシー、smoke test: 2ステップ）

```bash
.lerobot_venv/bin/lerobot-train \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=0 \
    --batch_size=2 \
    --steps=2 \
    --output_dir=outputs/test_smoke
```

```
Using video codec: libsvtav1
Creating policy
num_learnable_params=51642268 (52M)
Training: 100%|██████████| 2/2 [00:08<00:00, 4.19s/step]
End of training
```

### 3.5 学習（ACT ポリシー、本番: 100000ステップ）

```bash
.lerobot_venv/bin/lerobot-train \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=4 \
    --batch_size=8 \
    --steps=100000 \
    --output_dir=outputs/act_realman
```

> `--num_workers`: コンテナ環境で SHM 不足エラーが出る場合は `0` に設定。
> `--batch_size`: GPU メモリに応じて調整（128GB GPU なら 64 以上も可能）。

### 3.6 学習（Diffusion ポリシー）

```bash
.lerobot_venv/bin/lerobot-train \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.type=diffusion \
    --policy.push_to_hub=false \
    --num_workers=0 \
    --batch_size=8 \
    --steps=100000 \
    --output_dir=outputs/diffusion_realman
```

---

## 4. Makefile による簡易実行

```bash
make help          # 利用可能なコマンド一覧
make test-all      # import / dataset / train の全検証
make train-act     # ACTポリシーで本番学習
make train-diffusion  # Diffusionポリシーで本番学習
make check-dataset-features  # データセット構造確認
```

---

## 5. 重要なオプション一覧

| オプション | 説明 | 備考 |
|---|---|---|
| `--dataset.repo_id` | データセット識別子 | ローカルでもHub形式の名前が必要 |
| `--dataset.root` | データセットのローカルパス | `meta/`, `data/`, `videos/` を含むディレクトリ |
| `--policy.type` | ポリシー種別 | `act`, `diffusion` 等 |
| `--policy.push_to_hub=false` | Hub へのモデル自動アップロードを無効化 | 省略すると `repo_id` 未指定エラー |
| `--num_workers` | DataLoader ワーカー数 | SHM不足時は `0` に設定 |
| `--batch_size` | バッチサイズ | GPU メモリに応じて調整 |
| `--steps` | 学習ステップ数 | |
| `--output_dir` | チェックポイント・ログの保存先 | |
| `--dataset.video_backend` | 動画デコードバックエンド | 通常は自動検出（torchcodec or pyav） |

---

## 6. 設計メモ

### robot.type は学習に不要

`TrainPipelineConfig` に `robot` フィールドは**存在しない**。
学習はデータセットの `features`（`meta/info.json`）のみからポリシーの入出力次元を自動決定する。
robot config はデプロイ（推論→実機制御）フェーズで必要。

### video backend の自動選択

lerobot の `pyproject.toml` で:
- `torchcodec`: x86_64 環境で自動インストール
- aarch64 (Jetson) では `torchcodec` が除外され、`pyav` (torchvision 経由) にフォールバック
- `torchvision` が正しくインストールされていれば VideoReader が使える

**重要**: lerobot を `uv pip install -e lerobot/` で正しくインストールすること。
`sys.path` 操作だけでは torchvision のバージョンが合わず VideoReader が使えない場合がある。

### train_realman.py（旧ラッパー）について

`train_realman.py` は lerobot が正しくインストールされていない環境で
`torchvision.io.VideoReader` がない場合の workaround として PyAV を直接使う monkey-patch を提供する。
**lerobot を正しく uv でインストールすれば不要**。`lerobot-train` コマンドを推奨。

---

## 7. トラブルシューティング

### SHM 不足エラー

```
RuntimeError: DataLoader worker is killed by signal: Bus error
```

→ `--num_workers=0` を指定。`df -h /dev/shm` で SHM サイズ確認。

### push_to_hub エラー

```
ValueError: 'policy.repo_id' argument missing.
```

→ `--policy.push_to_hub=false` を指定。

### egg-info 権限エラー（インストール時）

```
error: Cannot update time stamp of directory 'src/lerobot.egg-info'
```

→ 別ユーザー（root 等）で `uv pip install` した際に `.lerobot_venv/` や
`lerobot/src/lerobot.egg-info/` がそのユーザー所有で作られている。
venv ごと削除して作り直す:

```bash
sudo rm -rf .lerobot_venv lerobot/src/lerobot.egg-info
uv venv .lerobot_venv --python 3.12
uv pip install -e lerobot/ --python .lerobot_venv/bin/python
```

### video decoding エラー

```
AttributeError: module 'torchvision.io' has no attribute 'VideoReader'
```

→ lerobot が正しくインストールされていない。`uv pip install -e lerobot/` を再実行。
それでも解決しない場合は `train_realman.py` (monkey-patch ラッパー) を使用。

### v2.1 変換時のスキーマエラー

```
TypeError: Couldn't cast array of type list<int32> to int32
```

→ RoboCOIN 固有の拡張フィールド（`subtask_annotation` 等）のスキーマ不整合。
既に v3.0 で提供されているデータセットを使うか、変換後に info.json の features から
RoboCOIN 固有フィールドを除去する。

---

## 8. lerobot CLI コマンド一覧

| コマンド | 説明 |
|---|---|
| `lerobot-train` | ポリシーの学習 |
| `lerobot-eval` | ポリシーの評価 |
| `lerobot-info` | 環境情報の表示 |
| `lerobot-dataset-viz` | データセットの可視化 |
| `lerobot-edit-dataset` | データセットの編集 |
| `lerobot-record` | ロボット操作の記録 |
| `lerobot-replay` | 記録の再生 |
| `lerobot-calibrate` | ロボットのキャリブレーション |
| `lerobot-teleoperate` | テレオペレーション |
| `lerobot-find-cameras` | カメラの検出 |
| `lerobot-find-port` | ポートの検出 |
| `lerobot-setup-motors` | モーターのセットアップ |
| `lerobot-setup-can` | CAN バスのセットアップ |
| `lerobot-train-tokenizer` | トークナイザーの学習 |
| `lerobot-imgtransform-viz` | 画像変換の可視化 |
| `lerobot-find-joint-limits` | 関節可動域の検出 |

---

*最終更新: 2026-03-26*
