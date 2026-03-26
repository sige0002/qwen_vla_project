# RoboCOIN データセット → LeRobot 学習 パイプラインガイド

本ドキュメントは、RoboCOINデータセットを用いたLeRobotの学習パイプライン（データ変換・Config実装・学習実行・トラブルシューティング）をまとめたものです。

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
cat /path/to/dataset/meta/info.json | python3 -m json.tool | grep codebase_version
# "codebase_version": "v3.0" であれば変換済み
```

**action featureの注意点（28次元）:**

RoboCOINデータセットの `action` は28次元で構成されている:

| 次元 | フィールド名 | 説明 |
|---|---|---|
| 0–6 | `right_arm_joint_1_rad` 〜 `right_arm_joint_7_rad` | 右腕関節角度（ラジアン） |
| 7 | `right_gripper_open` | 右グリッパー開閉 |
| 8–13 | `right_eef_pos_{x,y,z}_m`, `right_eef_rot_euler_{x,y,z}_rad` | 右手先位置・姿勢 |
| 14–20 | `left_arm_joint_1_rad` 〜 `left_arm_joint_7_rad` | 左腕関節角度（ラジアン） |
| 21 | `left_gripper_open` | 左グリッパー開閉 |
| 22–27 | `left_eef_pos_{x,y,z}_m`, `left_eef_rot_euler_{x,y,z}_rad` | 左手先位置・姿勢 |

カメラは3台（`cam_head_rgb`, `cam_left_wrist_rgb`, `cam_right_wrist_rgb`）、解像度は 480×640、コーデックは AV1。

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
| `velocity` | `30` | 関節移動速度（0–100） |
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

### train_realman.py の構成（monkey-patchの説明）

`/workspace/qwen_vla_project/train_realman.py` は、`torchcodec` / `torchvision.io.VideoReader` が利用できない環境（aarch64, GPU非搭載など）向けに、LeRobotの動画デコード関数を起動時にすり替え（monkey-patch）するラッパースクリプトです。

**起動フロー:**

```
train_realman.py
├── 1. sys.path に lerobot/src を追加
├── 2. lerobot.datasets.video_utils をインポート
├── 3. decode_video_frames_av() を定義（PyAV使用）
├── 4. video_utils.decode_video_frames = decode_video_frames_av  ← monkey-patch
└── 5. lerobot.scripts.lerobot_train.train() を呼び出し
```

**decode_video_frames_av の動作:**
- `av.open()` でMP4を開き、シーク + フレームデコード
- リクエストされたタイムスタンプに最も近いフレームを返す
- 許容誤差 `tolerance_s` を超えた場合は `FrameTimestampError` を送出

### 実行コマンド例（ACTポリシー）

```bash
cd /workspace/qwen_vla_project

# uvの仮想環境を使用
python train_realman.py \
    --policy.type=act \
    --dataset.repo_id=local/robocoin \
    --dataset.root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --policy.push_to_hub=false \
    --num_workers=0 \
    --output_dir=outputs/act_realman
```

ハブにデータセットがある場合:
```bash
python train_realman.py \
    --policy.type=act \
    --dataset.repo_id=your_org/robocoin_dataset \
    --policy.push_to_hub=false \
    --num_workers=0
```

### 重要なオプション

| オプション | 推奨値 | 理由 |
|---|---|---|
| `--policy.push_to_hub=false` | `false` | HuggingFaceへのモデル自動アップロードを無効化。ネットワーク不要 or 認証なし環境で必須 |
| `--num_workers=0` | `0` | DataLoader のワーカープロセスを無効化。SHM（共有メモリ）不足エラーを回避するために必要 |

`train_realman.py` はこれらをデフォルト注入するため、明示指定しなくても自動で適用される:

```python
if not any(arg.startswith("--policy.push_to_hub") for arg in sys.argv[1:]):
    sys.argv.append("--policy.push_to_hub=false")
if not any(arg.startswith("--num_workers") for arg in sys.argv[1:]):
    sys.argv.append("--num_workers=0")
```

### データセット features の確認方法

```bash
python3 -c "
import sys; sys.path.insert(0, 'lerobot/src')
import json
with open('robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket/meta/info.json') as f:
    info = json.load(f)
for k, v in info['features'].items():
    print(f'{k}: dtype={v[\"dtype\"]}, shape={v[\"shape\"]}')
"
```

期待される出力（RoboCOIN）:
```
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

---

## 4. トラブルシューティング

### video decoding エラー（torchcodec非対応環境）

**症状:**
```
ImportError: No module named 'torchcodec'
# または
ImportError: No module named 'torchvision'
```

**原因:** LeRobotのデフォルトのビデオデコードバックエンドが `torchcodec` または `torchvision.io.VideoReader` を要求するが、aarch64環境やGPU非搭載環境ではインストールできない。

**対処:** `train_realman.py` を使用する。このスクリプトは起動時に `decode_video_frames` 関数をPyAVベースの実装に置き換える（monkey-patch）。直接 `lerobot_train.py` を呼ばず、必ず `train_realman.py` 経由で実行する。

```bash
# NG: 直接呼び出し（torchcodecが必要）
python lerobot/src/lerobot/scripts/lerobot_train.py ...

# OK: monkey-patchラッパー経由
python train_realman.py ...
```

### SHM不足エラー（DataLoader ワーカー関連）

**症状:**
```
RuntimeError: DataLoader worker (pid XXXXX) is killed by signal: Bus error
# または
OSError: [Errno 28] No space left on device (shared memory)
```

**原因:** DataLoaderがマルチプロセスでデータを読み込む際、共有メモリ（`/dev/shm`）を消費する。コンテナ環境などではSHMサイズが制限されている（デフォルト64MB）。

**対処:** `--num_workers=0` を指定してシングルプロセスに切り替える。`train_realman.py` はデフォルトで自動付与するため、明示指定しなくても適用される。

```bash
python train_realman.py ... --num_workers=0
```

SHMのサイズ確認:
```bash
df -h /dev/shm
```

### push_to_hub エラー

**症状:**
```
huggingface_hub.utils._errors.HfHubHTTPError: 401 Unauthorized
# または
ValueError: You must be logged in to push to the Hub
```

**原因:** デフォルトではモデルのチェックポイントをHuggingFace Hubに自動アップロードしようとする。

**対処:** `--policy.push_to_hub=false` を指定する（`train_realman.py` は自動付与）。

HuggingFaceにログインして使う場合:
```bash
huggingface-cli login
python train_realman.py ... --policy.push_to_hub=true
```

### robot_type が認識されない

**症状:**
```
ValueError: Error creating robot with config ...
```

**原因:** `robots/utils.py` の `make_robot_from_config()` に新しいロボットタイプの分岐が追加されていない。

**対処:** `utils.py` に `elif config.type == "my_robot":` 分岐を追加する（セクション2の登録手順を参照）。

---

*最終更新: 2026-03-26*
