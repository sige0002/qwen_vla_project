# トラブルシューティング

学習パイプラインの手順は → [PIPELINE_GUIDE.md](PIPELINE_GUIDE.md)

---

## SHM 不足 (DataLoader Bus error)

```
RuntimeError: DataLoader worker is killed by signal: Bus error
```

**原因**: コンテナの `/dev/shm` が小さい。

**対処**: `--num_workers=0` を指定する。または `docker run --shm-size=8g` で SHM を拡大。

---

## push_to_hub エラー

```
ValueError: 'policy.repo_id' argument missing.
```

**対処**: `--policy.push_to_hub=false` を指定する。

---

## VideoReader が見つからない

```
AttributeError: module 'torchvision.io' has no attribute 'VideoReader'
```

**原因**: lerobot が正しくインストールされていない。

**対処**:
```bash
uv pip install -e lerobot/ --python .lerobot_venv/bin/python
```

それでも解決しない場合は `scripts/train_realman.py` (monkey-patch ラッパー) を使用。

---

## v2.1 変換後のスキーマエラー

```
TypeError: Couldn't cast array of type list<int32> to int32
```

**原因**: RoboCOIN 固有の `scene_annotation` (shape=[1]) が parquet で `list<int32>` のまま残っている。

**対処**:
```bash
python scripts/fix_parquet_list_scalars.py --root=<dataset_root>
```

---

## ファイル権限エラー (Permission denied)

```
error: Cannot update time stamp of directory 'src/lerobot.egg-info'
error: failed to remove file `.lerobot_venv/...`: Permission denied
```

**原因**: 別ユーザー (root 等) で venv を作成したため、ファイルの所有者が異なる。

**対処**: venv を削除して作り直す。
```bash
sudo rm -rf .lerobot_venv lerobot/src/lerobot.egg-info
make setup
```

---

## コンテナ内ファイルのホスト権限

```
Permission denied  # ホスト側でファイルを操作しようとした場合
```

**対処**: コンテナ内で所有者を統一する。
```bash
sudo chown -R $(whoami):$(whoami) /workspace/qwen_vla_project/
```

---

## PyTorch が CPU 版になる

**原因**: `uv pip install` がデフォルトで PyPI の CPU 版をインストールする。

**対処**: `lerobot/pyproject.toml` に CUDA index を設定済み。

```toml
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu130" }
torchvision = { index = "pytorch-cu130" }
```

確認:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
# → True 13.0
```

別の CUDA バージョンの場合は `cu130` を変更 (例: `cu124`, `cu126`)。

---

*最終更新: 2026-03-26*
