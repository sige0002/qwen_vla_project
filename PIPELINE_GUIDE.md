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

### PyTorch CUDA 版の注意（DGX Spark 等）

`uv pip install` はデフォルトで PyPI の CPU 版 PyTorch をインストールしてしまう。
CUDA 版を確実に入れるため、`lerobot/pyproject.toml` に以下を追記済み:

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
```

別の CUDA バージョンの場合は `cu130` を変更する（例: `cu124`, `cu126`）。

確認方法:
```bash
.lerobot_venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# → True 13.0
```

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

### 変換後の parquet 修正（shape=[1] リスト型の修正）

変換後、RoboCOIN 固有フィールドのうち `scene_annotation` (shape=[1]) が parquet 上で
`list<int32>` として格納されており、LeRobot の読み込みで `int32` へのキャスト失敗エラーが出る。
`fix_parquet_list_scalars.py` で修正する:

```bash
.lerobot_venv/bin/python fix_parquet_list_scalars.py \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old
```

このスクリプトは info.json の shape=[1] フィールドのうち parquet で list 型のものだけを
スカラーに変換する。拡張フィールドは削除しない。

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

変換スクリプト自体はこれらのフィールドを保持するが、`scene_annotation` (shape=[1]) が
parquet 上で `list<int32>` のまま残り、LeRobot の読み込み時にスキーマ不整合が起きる。
変換後に `fix_parquet_list_scalars.py` を実行して修正する（上記参照）。

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

**注意: 2種類のデータセットについて**

| データセット | パス | バージョン | features 数 | 拡張フィールド |
|---|---|---|---|---|
| 既存 v3.0 | `Realman_RMC_AIDA_L_storage_block_basket/` | v3.0 | 10 | なし |
| old (v2.1→v3.0変換) | `Realman_RMC_AIDA_L_storage_block_basket_old/` | v2.1 | 26 | あり（16列） |

既存 v3.0 データは拡張フィールドが最初から含まれていない。
old データを変換すると拡張フィールドを保持した v3.0 データが得られる。

### 3.1 環境情報の確認

```bash
.lerobot_venv/bin/lerobot-info
```

```
- LeRobot version: 0.5.0
- Platform: Linux-6.14.0-1015-nvidia-aarch64-with-glibc2.39
- Python version: 3.12.3
- PyTorch version: 2.10.0
```

### 3.2 データセットの features 確認（既存 v3.0、拡張フィールドなし）

**使用データ: `Realman_RMC_AIDA_L_storage_block_basket/` (既存 v3.0、10 features)**

```bash
.lerobot_venv/bin/python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    repo_id='robocoin/Realman_RMC_AIDA_L_storage_block_basket',
    root='robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket'
)
print(f'Frames: {len(ds)}')
for k, v in ds.meta.info['features'].items():
    print(f'  {k}: dtype={v[\"dtype\"]}, shape={v[\"shape\"]}')
"
```

```
Frames: 19083
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

> このデータには RoboCOIN 拡張フィールド（subtask_annotation, eef_* 等）は含まれていない。

### 3.3 学習 smoke test（既存 v3.0、拡張フィールドなし）

**使用データ: `Realman_RMC_AIDA_L_storage_block_basket/` (既存 v3.0、10 features)**

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
num_learnable_params=51642268 (52M)
Training: 100%|██████████| 2/2 [00:08<00:00, 4.19s/step]
End of training
```

> **結果: 成功。** ただしこのデータには拡張フィールドがないため、
> 拡張フィールドの互換性は検証できていない。

### 3.4 v2.1 → v3.0 データセット変換（拡張フィールド保持）

**使用データ: `Realman_RMC_AIDA_L_storage_block_basket_old/` (v2.1、26 features)**

```bash
# Step 1: v2.1 → v3.0 変換（拡張フィールドは保持される）
.lerobot_venv/bin/python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=robocoin/Realman_RMC_AIDA_L_storage_block_basket_old \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old \
    --push-to-hub=false

# Step 2: shape=[1] リスト型をスカラーに修正
# （scene_annotation が list<int32> のまま残り LeRobot 読み込みでエラーになるため）
.lerobot_venv/bin/python fix_parquet_list_scalars.py \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old

# または Makefile で一発:
# make convert-v21-to-v30
```

```
# Step 1 出力:
Converting data files from 50 episodes
convert data files: 100%|██████████| 50/50
convert videos: 100%|██████████| 50/50
Converting episodes metadata ...

# Step 2 出力:
shape=[1] features to check: ['timestamp', 'frame_index', 'episode_index', 'index', 'task_index', 'scene_annotation']
  Fixed: scene_annotation (list<int32> -> int32)
Done. Fixed 1/1 file(s).
```

> **変換後の info.json には 26 features が保持される（拡張フィールド 16 列含む）。**
> **parquet にも 23 カラム全て保持される。**
>
> Step 2 を省略すると以下のエラーが出る:
> `TypeError: Couldn't cast array of type list<int32> to int32`

### 3.5 学習 smoke test（変換後 v3.0、拡張フィールドあり）

**使用データ: v2.1 から変換した v3.0 データ（26 features、fix 適用済み）**

```bash
.lerobot_venv/bin/lerobot-train \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket_old \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket_old \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=0 \
    --batch_size=2 \
    --steps=2 \
    --output_dir=outputs/test_smoke_with_ext
```

```
num_learnable_params=51642268 (52M)
Training: 100%|██████████| 2/2 [00:08<00:00, 4.12s/step]
End of training
```

> **結果: 成功。** 拡張フィールド付きデータでも学習が動作することを確認。

### 3.6 可視化

**使用データ: `Realman_RMC_AIDA_L_storage_block_basket/` (既存 v3.0)**

```bash
.lerobot_venv/bin/lerobot-dataset-viz \
    --repo-id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --episode-index=0 \
    --num-workers=0 \
    --save=1 \
    --output-dir=outputs/viz_v30
```

```
100%|██████████| 10/10 [00:08<00:00, 1.13it/s]
```

→ `outputs/viz_v30/*.rrd` が出力される。Rerun Viewer で開いて確認可能。

### 3.7 本番学習コマンド例

```bash
# ACT ポリシー
.lerobot_venv/bin/lerobot-train \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=4 \
    --batch_size=8 \
    --steps=100000 \
    --output_dir=outputs/act_realman

# Diffusion ポリシー
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

> `--num_workers`: コンテナ環境で SHM 不足エラーが出る場合は `0` に設定。
> `--batch_size`: GPU メモリに応じて調整（128GB GPU なら 64 以上も可能）。

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

### コンテナ内で作成したファイルがホストから操作できない

```
Permission denied
```

→ コンテナ内で root として作成されたファイルは、ホスト側のユーザーから操作できない。
コンテナ内で全ファイルの所有者を統一する:

```bash
# コンテナ内で実行
sudo chown -R $(whoami):$(whoami) /workspace/qwen_vla_project/
```

### インストール時の Permission denied エラー

以下のいずれかのエラーが出る場合:

```
error: Cannot update time stamp of directory 'src/lerobot.egg-info'
```
```
error: failed to remove file `.lerobot_venv/lib/.../bin/lerobot-calibrate`: Permission denied
```
```
rm: cannot remove 'lerobot/src/lerobot.egg-info/PKG-INFO': Permission denied
```

→ **原因**: 別ユーザー（root、コンテナ内の別ユーザー等）で `uv venv` や `uv pip install`
を実行した際に `.lerobot_venv/` と `lerobot/src/lerobot.egg-info/` がそのユーザー所有で
作られている。egg-info だけ消しても venv 内の `bin/lerobot-*` スクリプトも同じ所有者のため
再インストールも失敗する。

→ **対処**: venv ごと `sudo` で削除して、自分のユーザーで作り直す:

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

### v2.1 変換後のスキーマエラー

```
TypeError: Couldn't cast array of type list<int32> to int32
```

→ RoboCOIN 固有フィールドのうち shape=[1] のもの（`scene_annotation`）が parquet で
`list<int32>` のまま残り、LeRobot が `Value(int32)` として読もうとして失敗する。
変換後に `fix_parquet_list_scalars.py` を実行する:

```bash
.lerobot_venv/bin/python fix_parquet_list_scalars.py --root=<dataset_root>
```

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
