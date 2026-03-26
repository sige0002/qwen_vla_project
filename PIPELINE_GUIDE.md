# RoboCOIN データセット → LeRobot 学習 パイプラインガイド

本ドキュメントは、RoboCOINデータセットを用いたLeRobotの学習パイプライン（データ変換・Config実装・学習実行・検証・トラブルシューティング）をまとめたものです。

---

## 1. データセット変換（v2.1 → v3.0）

### 変換スクリプトの場所

```
lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py
```

### v2.1 → v3.0 の主な変更点

| 項目 | v2.1 (OLD) | v3.0 (NEW) |
|---|---|---|
| `data_path` | `data/chunk-000/episode_000000.parquet` | `data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet` |
| `video_path` | `videos/chunk-000/CAMERA/episode_000000.mp4` | `videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4` |
| `episodes` | `episodes.jsonl`（JSONL形式） | `meta/episodes/chunk-000/file_000.parquet`（Parquet形式、統計も含む） |
| `tasks` | `tasks.jsonl`（JSONL形式） | `meta/tasks.parquet`（Parquet形式） |
| `stats` | `stats.json`（ファイル全体の統計） | `episodes_stats.jsonl`（エピソード毎の統計、その後Parquetに統合） |

### 実行コマンド例

**ハブ上のデータセットを変換する場合:**
```bash
python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=your_org/your_dataset
```

**ローカルデータセットを変換する場合（ハブへのpushなし）:**
```bash
python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
    --repo-id=your_org/your_dataset \
    --root=/path/to/local/dataset \
    --push-to-hub=false
```

> `--root` はデータセットの **ルートディレクトリ**（`meta/`, `data/`, `videos/` を含むフォルダ）を指定する。

### RoboCOIN固有フィールドの注意点

RoboCOINデータセット（例: `Realman_RMC_AIDA_L_storage_block_basket`）は既に **v3.0** 形式で提供されているため、変換スクリプトの実行は不要な場合がある。

確認方法:
```bash
make check-dataset-version
# または手動で:
cat robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket/meta/info.json \
    | python3 -m json.tool | grep codebase_version
# "codebase_version": "v3.0" であれば変換済み
```

**action featureの注意点（28次元）:**

RoboCOINデータセットの `action` は28次元で構成されている:

| 次元 | フィールド名 | 説明 |
|---|---|---|
| 0--6 | `right_arm_joint_1_rad` 〜 `right_arm_joint_7_rad` | 右腕関節角度（ラジアン） |
| 7 | `right_gripper_open` | 右グリッパー開閉 |
| 8--13 | `right_eef_pos_{x,y,z}_m`, `right_eef_rot_euler_{x,y,z}_rad` | 右手先位置・姿勢 |
| 14--20 | `left_arm_joint_1_rad` 〜 `left_arm_joint_7_rad` | 左腕関節角度（ラジアン） |
| 21 | `left_gripper_open` | 左グリッパー開閉 |
| 22--27 | `left_eef_pos_{x,y,z}_m`, `left_eef_rot_euler_{x,y,z}_rad` | 左手先位置・姿勢 |

カメラは3台（`cam_head_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`）、解像度は 480x640、コーデックは AV1。

---

## 2. Realman Config の実装

### ファイル構成

```
lerobot/src/lerobot/robots/
├── realman/
│   ├── __init__.py             # RealmanFollowerConfig, RealmanFollower をエクスポート
│   ├── config_realman.py       # 単腕 Config (@RobotConfig.register_subclass("realman_follower"))
│   └── realman.py              # 単腕 Robot 実装
└── bi_realman/
    ├── __init__.py             # BiRealmanFollowerConfig, BiRealmanFollower をエクスポート
    ├── config_bi_realman.py    # 双腕 Config (@RobotConfig.register_subclass("bi_realman_follower"))
    └── bi_realman.py           # 双腕 Robot 実装
```

### 単腕（realman_follower）の設定

`config_realman.py` の主要パラメータ:

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `id` | `"default"` | ロボット識別子 |
| `ip` | `"169.254.128.18"` | Realman コントローラ IP |
| `port` | `8080` | Realman コントローラ ポート |
| `velocity` | `30` | 関節移動速度（0--100） |
| `block` | `False` | SDKコマンドのブロッキングモード |
| `wait_second` | `0.1` | 非ブロッキング時の待機秒数 |
| `joint_names` | `["joint_1", ..., "joint_7", "gripper"]` | 関節名リスト（8次元） |
| `init_state` | `[-0.84, -2.03, 1.15, ...]` | 初期関節角度（deg, gripper value） |

### 双腕（bi_realman_follower）の設定

`config_bi_realman.py` の主要パラメータ:

| パラメータ | デフォルト値 | 説明 |
|---|---|---|
| `ip_left` | `"169.254.128.18"` | 左腕コントローラ IP |
| `ip_right` | `"169.254.128.19"` | 右腕コントローラ IP |
| `port_left` / `port_right` | `8080` | 各コントローラ ポート |
| `joint_names` | `["joint_1", ..., "joint_7", "gripper"]` | 片腕分の関節名（実行時に left_/right_ プレフィックス付与） |
| `init_state_left` | `[-0.84, -2.03, ...]` | 左腕初期状態 |
| `init_state_right` | `[1.16, 2.01, ...]` | 右腕初期状態 |

**次元不一致の注意:**
- 学習データ（RoboCOIN）は28次元（左右各7関節 + 1グリッパー + 6次元EEF）
- `BiRealmanFollowerConfig` が定義するのは16次元（左右各8 = 7関節 + 1グリッパー）のみ
- デプロイ時にはキー名リネームと次元変換が必要

### 登録手順（utils.py への追加）

`lerobot/src/lerobot/robots/utils.py` の `make_robot_from_config()` に以下を追加:

```python
elif config.type == "realman_follower":
    from .realman import RealmanFollower

    return RealmanFollower(config)
elif config.type == "bi_realman_follower":
    from .bi_realman import BiRealmanFollower

    return BiRealmanFollower(config)
```

---

## 3. 学習パイプライン

### なぜ lerobot_train 直接実行ではなく train_realman.py ラッパーが必要か

LeRobotは `DatasetConfig.video_backend` で `pyav` を指定可能だが、**内部実装に問題がある**：

```
video_backend="pyav"
  → decode_video_frames() が呼ばれる
    → decode_video_frames_torchvision(backend="pyav") にルーティング
      → torchvision.set_video_backend("pyav") を設定
        → torchvision.io.VideoReader を呼び出す  ← ここでクラッシュ
```

つまり LeRobot の `pyav` バックエンドは「PyAVライブラリを直接使う」のではなく「torchvision の pyav バックエンドを使う」という意味であり、`torchvision.io.VideoReader` が必要。VideoReader 未搭載環境（aarch64、特定のGPU非搭載ビルド等）では以下のエラーが出る：

```
AttributeError: module 'torchvision.io' has no attribute 'VideoReader'
```

**検証結果（2026-03-26実施）：**

| テスト | コマンド | 結果 |
|--------|---------|------|
| lerobot_train 直接実行 | `PYTHONPATH=lerobot/src python -m lerobot.scripts.lerobot_train --dataset.video_backend=pyav ...` | **FAIL** (`VideoReader` 未搭載) |
| train_realman.py 経由 | `python train_realman.py ...` | **PASS** (2ステップ完走) |

したがって `train_realman.py` の monkey-patch（PyAV を直接使う `decode_video_frames_av` で置き換え）は、この環境では**唯一の有効な手段**である。

### train_realman.py の構成

```
train_realman.py
├── 1. sys.path に lerobot/src を追加
├── 2. lerobot.datasets.video_utils をインポート
├── 3. decode_video_frames_av() を定義（PyAV を直接使用）
├── 4. video_utils.decode_video_frames = decode_video_frames_av  ← monkey-patch
└── 5. lerobot.scripts.lerobot_train.train() を呼び出し
```

**decode_video_frames_av の動作:**
- `av.open()` で動画ファイルを開く
- 最初のリクエストタイムスタンプの1秒前にシーク
- フレームをデコードして `torch.Tensor (N, C, H, W)` で返す（float32, [0, 1]）
- 許容誤差 `tolerance_s` を超えた場合は `FrameTimestampError` を送出

**注意: `robot.type` について**
- `TrainPipelineConfig` に `robot` フィールドは**存在しない**
- 学習にはデータセットの features 情報のみが使われ、robot config は不要
- robot config はデプロイ（推論→実機送信）時に必要になる

### Makefile によるコマンド管理

デフォルトオプション（`--policy.push_to_hub=false`, `--num_workers=0` 等）は Makefile の `COMMON_ARGS` で一元管理する。

```bash
cd /workspace/qwen_vla_project

# 利用可能なコマンド一覧を表示
make help
```

| Makeターゲット | 説明 | 用途 |
|---------------|------|------|
| `make test-import` | realman/bi_realman の import テスト | Config/Robot クラスが正しく読み込めるか確認 |
| `make test-dataset` | データセットの読み込みテスト | LeRobotDataset でフレーム数・features を確認 |
| `make test-train` | 2ステップの smoke test（ACT） | 学習パイプラインが最低限動くか確認 |
| `make test-all` | 上記3つを順番に実行 | CI/手動検証用 |
| `make train-act` | ACTポリシーで本番学習 | batch_size=8, steps=100000 |
| `make train-diffusion` | Diffusionポリシーで本番学習 | batch_size=8, steps=100000 |
| `make check-dataset-version` | データセットの codebase_version を表示 | v3.0 かどうか確認 |
| `make check-dataset-features` | データセットの全 features を表示 | 次元・dtype の確認 |

### 直接実行する場合のコマンド例

Makefileを使わず直接実行する場合:

```bash
cd /workspace/qwen_vla_project

# ACTポリシーで学習
.venv/bin/python train_realman.py \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=0 \
    --batch_size=8 \
    --steps=100000 \
    --output_dir=outputs/act_realman
```

### 重要なオプション

| オプション | 推奨値 | 理由 |
|---|---|---|
| `--policy.push_to_hub=false` | `false` | HuggingFaceへのモデル自動アップロードを無効化。省略すると `repo_id` 未指定エラー |
| `--num_workers=0` | `0` | DataLoader のワーカープロセスを無効化。SHM不足エラーを回避 |
| `--policy.type=act` | 任意 | ポリシー種別。`act`, `diffusion` 等を指定 |
| `--batch_size=N` | `8` | GPUメモリに応じて調整 |
| `--steps=N` | `100000` | 学習ステップ数 |
| `--output_dir=PATH` | 任意 | チェックポイント・ログの保存先 |

---

## 4. 検証手順（実行済みテストコマンドと結果）

以下は 2026-03-26 に実際に実行して検証したコマンドと結果です。

### 4.1 Import テスト

```bash
cd /workspace/qwen_vla_project

# 単腕 Realman
.venv/bin/python -c "
import sys; sys.path.insert(0, 'lerobot/src')
from lerobot.robots.realman import RealmanFollowerConfig, RealmanFollower
print('realman import OK')
"
# 結果: realman import OK

# 双腕 BiRealman
.venv/bin/python -c "
import sys; sys.path.insert(0, 'lerobot/src')
from lerobot.robots.bi_realman import BiRealmanFollowerConfig, BiRealmanFollower
print('bi_realman import OK')
"
# 結果: bi_realman import OK
```

### 4.2 データセット読み込みテスト

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'lerobot/src')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(
    repo_id='robocoin/Realman_RMC_AIDA_L_storage_block_basket',
    root='robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket'
)
print('Dataset loaded:', len(ds), 'frames')
print('Features:', list(ds.meta.info['features'].keys()))
"
# 結果:
#   Dataset loaded: 19083 frames
#   Features: ['observation.images.cam_head_rgb', 'observation.images.cam_left_wrist_rgb',
#              'observation.images.cam_right_wrist_rgb', 'observation.state', 'action',
#              'timestamp', 'frame_index', 'episode_index', 'index', 'task_index']
```

### 4.3 video_backend 自動検出テスト

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'lerobot/src')
from lerobot.configs.default import get_safe_default_codec
print('Default codec:', get_safe_default_codec())
"
# 結果:
#   WARNING: 'torchcodec' is not available in your platform, falling back to 'pyav'
#   Default codec: pyav
```

### 4.4 lerobot_train 直接実行テスト（ラッパーなし）

```bash
PYTHONPATH=lerobot/src:$PYTHONPATH .venv/bin/python -m lerobot.scripts.lerobot_train \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.video_backend=pyav \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=0 \
    --batch_size=2 \
    --steps=2 \
    --output_dir=outputs/test_direct
# 結果: FAIL
#   AttributeError: module 'torchvision.io' has no attribute 'VideoReader'
#   → video_backend=pyav でも内部で torchvision.io.VideoReader を要求するため動かない
```

### 4.5 train_realman.py ラッパー経由の学習テスト（smoke test）

```bash
.venv/bin/python train_realman.py \
    --dataset.repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.type=act \
    --policy.push_to_hub=false \
    --num_workers=0 \
    --batch_size=2 \
    --steps=2 \
    --output_dir=outputs/test_smoke
# 結果: PASS
#   Training: 100%|██████████| 2/2 [00:07<00:00, 3.80s/step]
#   INFO End of training
```

### 4.6 Makefile 経由のテスト

```bash
make test-import
# 結果: realman import OK / bi_realman import OK

make test-dataset
# 結果: Dataset loaded: 19083 frames, Features: [10 features]

make test-train
# 結果: Training 2/2 完了, End of training

make check-dataset-features
# 結果:
#   observation.images.cam_head_rgb: dtype=video, shape=[480, 640, 3]
#   observation.images.cam_left_wrist_rgb: dtype=video, shape=[480, 640, 3]
#   observation.images.cam_right_wrist_rgb: dtype=video, shape=[480, 640, 3]
#   observation.state: dtype=float32, shape=[28]
#   action: dtype=float32, shape=[28]
#   timestamp: dtype=float32, shape=[1]
#   frame_index: dtype=int64, shape=[1]
#   episode_index: dtype=int64, shape=[1]
#   index: dtype=int64, shape=[1]
#   task_index: dtype=int64, shape=[1]
```

---

## 5. 設計判断の記録

### なぜ `--robot.type=realman` ではないのか

LeRobot の `TrainPipelineConfig` には `robot` フィールドが**存在しない**。学習パイプラインはデータセットの `features`（`meta/info.json`）のみからポリシーの入出力次元を決定する。robot config が必要になるのはデプロイ（推論→実機制御）フェーズのみ。

### なぜ monkey-patch ラッパーを維持するのか

| アプローチ | 結果 | 備考 |
|-----------|------|------|
| `lerobot_train` 直接実行 + `--dataset.video_backend=pyav` | **FAIL** | `torchvision.io.VideoReader` 未搭載でクラッシュ |
| `train_realman.py`（monkey-patch） | **PASS** | PyAV を直接使い VideoReader を完全バイパス |

LeRobot の `video_backend="pyav"` は「PyAV を直接使う」のではなく「torchvision の pyav バックエンドを使う」意味であり、VideoReader が必須。この問題は LeRobot 本体の設計上の制約であるため、monkey-patch で外部から回避するのが現時点で唯一の解決策。

**リスクと対策:**
- monkey-patch は `video_utils.decode_video_frames` という関数名に依存しており、LeRobot のアップデートで壊れる可能性がある
- LeRobot アップデート時は当該関数のシグネチャが変更されていないか確認すること
- 長期的には LeRobot 本体に純粋な PyAV バックエンドを PR するのが望ましい

### なぜ sys.argv 注入ではなく Makefile か

| 方式 | メリット | デメリット |
|------|---------|-----------|
| sys.argv 注入（旧方式） | 実行コマンドが短い | コードがCLI引数を自己改変するハック |
| Makefile（現方式） | デフォルト値がgit管理可能、透明性が高い | `make` コマンドの知識が必要 |

Makefile の `COMMON_ARGS` でデフォルト値を一元管理することで、何のオプションが渡されているかが常に明示的になる。

---

## 6. トラブルシューティング

### video decoding エラー（torchcodec / VideoReader 非対応環境）

**症状:**
```
AttributeError: module 'torchvision.io' has no attribute 'VideoReader'
# または
ImportError: No module named 'torchcodec'
```

**原因:** LeRobot の video backend が `torchvision.io.VideoReader` を要求するが、環境に搭載されていない。`--dataset.video_backend=pyav` を指定しても、内部で VideoReader 経由のため解決しない。

**対処:** 必ず `train_realman.py` 経由で実行する。
```bash
# NG: 直接呼び出し
PYTHONPATH=lerobot/src python -m lerobot.scripts.lerobot_train ...

# OK: monkey-patch ラッパー経由
python train_realman.py ...

# OK: Makefile 経由
make train-act
```

### SHM不足エラー（DataLoader ワーカー関連）

**症状:**
```
RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Bus error
```

**原因:** `/dev/shm` のサイズ不足。コンテナ環境ではデフォルト64MB。

**対処:** `--num_workers=0` を指定。Makefile では `COMMON_ARGS` に含まれている。

```bash
df -h /dev/shm  # SHM サイズ確認
```

### push_to_hub エラー

**症状:**
```
ValueError: 'policy.repo_id' argument missing. Please specify it to push the model to the hub.
```

**対処:** `--policy.push_to_hub=false` を指定。Makefile では `COMMON_ARGS` に含まれている。

### robot_type が認識されない

**症状:**
```
ValueError: Error creating robot with config ...
```

**対処:** `lerobot/src/lerobot/robots/utils.py` の `make_robot_from_config()` に elif 分岐を追加する（セクション2の登録手順を参照）。

---

*最終更新: 2026-03-26*
