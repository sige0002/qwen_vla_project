# Plan: RoboCOIN データセット → LeRobot v3.0 対応 & Realman 学習パイプライン構築

## 概要

RoboCOIN（LeRobot v2.1ベース）のデータセットをLeRobot v3.0形式に変換し、LeRobot本体にRealmanロボットのconfigを実装した上で、学習スクリプトを実行可能にする。

---

## Task 1: データセット変換（v2.1 → v3.0）

### 背景
- RoboCOIN は LeRobot v2.1 フォーマットでデータセットを管理している
- LeRobot 本体（サブモジュール）は v3.0 フォーマットを採用済み
- 変換スクリプト `lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py` が存在する

### 主な変更点（v2.1 → v3.0）

| 項目 | v2.1 | v3.0 |
|------|------|------|
| データファイル | `data/chunk-000/episode_000000.parquet`（1エピソード=1ファイル） | `data/chunk-000/file_000.parquet`（複数エピソード統合） |
| 動画ファイル | `videos/chunk-000/CAMERA/episode_000000.mp4` | `videos/CAMERA/chunk-000/file_000.mp4` |
| エピソードメタ | `meta/episodes.jsonl` | `meta/episodes/chunk-000/file_000.parquet` |
| タスクメタ | `meta/tasks.jsonl` | `meta/tasks.parquet` |
| 統計情報 | `meta/episodes_stats.jsonl` + `meta/stats.json` | エピソードParquetに統合 + `meta/stats.json` |
| info.json | `total_chunks`, `total_videos` あり | 削除、`data_path`/`video_path` テンプレート追加 |

### 手順

1. **RoboCOINデータセットの所在確認**
   - `robocoin/` ディレクトリ内のデータセット構造を確認
   - `meta/info.json` で codebase_version が `v2.1` であることを確認

2. **変換スクリプトの実行準備**
   - LeRobot の依存関係をインストール（`uv` 仮想環境を使用）
   - 必要パッケージ: `jsonlines`, `pyarrow`, `pandas`, `datasets`, `huggingface_hub`

3. **変換スクリプトの実行**
   ```bash
   uv run python lerobot/src/lerobot/datasets/v30/convert_dataset_v21_to_v30.py \
     --repo-id=<dataset-repo-id> \
     --root=<robocoinデータセットのローカルパス> \
     --push-to-hub=false
   ```

4. **RoboCOIN固有フィールドの互換性確認**
   - RoboCOIN v2.1 には独自の annotation フィールドがある（subtask, scene, eef_acc_mag 等）
   - これらが変換後も保持されるか、または変換スクリプトの拡張が必要か確認
   - v2.1の `episodes_stats.jsonl` 形式が LeRobot 標準と互換か検証

5. **変換結果の検証**
   - `meta/info.json` の `codebase_version` が `v3.0` に更新されていること
   - データの読み込みテスト（`LeRobotDataset` でロードできること）

### リスク・注意点
- RoboCOIN独自の annotation（subtask, scene等）は LeRobot v3.0 変換スクリプトでは想定外の可能性あり → スクリプト修正が必要になるかもしれない
- 修正が必要な場合は新しいファイルで作成
- v2.1 の `episodes_stats.jsonl` のスキーマが RoboCOIN 固有の拡張を含む場合、変換時にエラーが出る可能性あり

---

## Task 2: LeRobot に Realman ロボット Config を実装

### 背景
- RoboCOIN には `robots/realman/` が存在（`configuration_realman.py`, `realman.py`, `realman_end_effector.py`）
- LeRobot 本体には Realman のサポートがない
- RoboCOIN の Realman 実装は `BaseRobot` / `BaseRobotConfig` を継承（RoboCOIN独自の基底クラス）
- LeRobot 本体の `RobotConfig` は `draccus.ChoiceRegistry` ベース（構造が異なる）

### Realman ロボットの仕様（RoboCOINより）
- **7軸関節 + グリッパー**（合計8自由度）
- IP/ポート経由で接続（デフォルト: `169.254.128.18:8080`）
- 関節角度: degree 単位、グリッパー: meter 単位
- SDK: `Robotic_Arm` パッケージが必要
- エンドエフェクタ制御対応

### 手順

1. **LeRobot のロボット実装パターンを把握**
   - `so_follower/` の実装を参考にする（`config_so_follower.py`, `so_follower.py`）
   - `RobotConfig` 基底クラス（`draccus.ChoiceRegistry`）の仕組みを理解

2. **Realman Config の作成**
   - `lerobot/src/lerobot/robots/realman/` ディレクトリを作成
   - `config_realman.py`: `RobotConfig` を継承した `RealmanFollowerConfig` を実装
     - パラメータ: `ip`, `port`, `velocity`, `joint_names`, `cameras` 等
   - `@RobotConfig.register_subclass("realman_follower")` で登録

3. **Realman Robot クラスの作成**
   - `realman.py`: `Robot` 基底クラスを継承
   - RoboCOIN の `realman.py` を参考に、LeRobot のインターフェースに合わせる
   - 必須プロパティ/メソッド:
     - `observation_features` / `action_features`
     - `connect()` / `disconnect()`
     - `get_observation()` / `send_action()`
     - `calibrate()` / `is_calibrated`

4. **`__init__.py` の作成と登録**
   - ロボットモジュールの `__init__.py` に import を追加
   - `lerobot/src/lerobot/robots/__init__.py` に realman を登録

5. **Bimanual 版の検討**
   - RoboCOIN には `bi_realman/` も存在
   - 必要に応じて `bi_realman_follower` も実装（Task 3 で必要な場合）

### RoboCOIN → LeRobot 移植時の主な差異

| 項目 | RoboCOIN | LeRobot |
|------|----------|---------|
| 基底クラス | `BaseRobotConfig` (独自) | `RobotConfig` (draccus) |
| 登録方法 | `@RobotConfig.register_subclass("realman")` | `@RobotConfig.register_subclass("realman_follower")` |
| カメラ設定 | `BaseRobotConfig` 内に含む | `cameras: dict[str, CameraConfig]` |
| 単位系 | `units_transform.py` で変換 | `use_degrees` フラグ等 |
| 依存関係 | `Robotic_Arm` SDK | 同じ（オプション依存） |

---

## Task 3: Realman Config で学習スクリプトを実行

### 背景
- LeRobot の学習スクリプト: `lerobot/src/lerobot/scripts/lerobot_train.py`
- 学習にはデータセット（Task 1 で変換済み）とポリシー設定が必要
- 実機接続は不要（データセットからのオフライン学習）

### 手順

1. **学習に必要な設定の確認**
   - `TrainPipelineConfig`: dataset, policy, batch_size, steps 等
   - `DatasetConfig`: repo_id（ローカルパス指定可能）、episodes、image_transforms
   - ポリシー選択: ACT / Diffusion / Pi0 等から選択

2. **データセットの observation/action features の確認**
   - Realman の場合:
     - `observation.state`: 7関節 + グリッパー（8次元）
     - `action`: 7関節 + グリッパー（8次元）
     - `observation.images.*`: カメラ画像
   - これらが LeRobot v3.0 のデータセットに正しく含まれていることを確認

3. **ポリシー設定の調整**
   - 選択したポリシー（例: ACT）の `input_features` / `output_features` を Realman のデータセットに合わせる
   - 必要に応じて `delta_timestamps` を設定

4. **学習スクリプトの実行テスト**
   ```bash
   uv run python -m lerobot.scripts.lerobot_train \
     --dataset.repo_id=<変換済みデータセットID> \
     --dataset.root=<ローカルパス> \
     --policy.type=act \
     --batch_size=8 \
     --steps=100 \
     --output_dir=outputs/realman_act
   ```

5. **エラー対応と動作確認**
   - データセットロード時のエラー（features の不一致等）を修正
   - ポリシーの入出力次元がデータセットと合致しているか確認
   - 学習ループが正常に回ること（loss が減少すること）を確認

### 前提条件
- Task 1 のデータセット変換が完了していること
- Task 2 の Realman config が実装済みであること（ただし学習自体は実機不要のため、config が無くてもデータセットさえ正しければ動く可能性あり）

---

## 実行順序

```
Task 1（データセット変換 v2.1→v3.0）
    ↓
Task 2（Realman config 実装）← Task 1 と並行可能
    ↓
Task 3（学習スクリプト実行）← Task 1, 2 の完了が前提
```

## 未確認事項（要確認）

- [ ] `robocoin/` ディレクトリ内の実際のデータセット構造と内容
- [ ] RoboCOIN データセットで使用されているロボットが Realman かどうか（他ロボットのデータの可能性）　→realmanなので懸念なし
- [ ] 学習に使うポリシーの種類（ACT / Diffusion / Pi0 / SmolVLA 等）→今は気にしない　lerobotで実装できていれば何でもできるはず
- [ ] Bimanual（双腕）対応が必要かどうか→対応する
- [ ] GPU環境の確認（学習に必要なVRAM等）考えなくていい．この環境は128GBある
