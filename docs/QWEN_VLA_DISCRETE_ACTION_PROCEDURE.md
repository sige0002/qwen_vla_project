# Qwen VLA 離散アクション + サブタスク出力 学習手順書

**目的**: Qwen3.5 VLMをバックボーンとして、Pi0.5のPre-trainingステージ相当（離散アクショントークン + サブタスク予測）の最小構成を実装・学習する。

**スコープ**: Phase 1のみ。Action Expert / Flow Matching は Phase 2 以降。

---

## 全体アーキテクチャ概要

```
入力                                    出力
─────────────────────────────────────────────────────────
3カメラ画像 (480x640x3)  ─┐
テキスト指示             ─┼─→ Qwen3.5 VLM ─→ サブタスクテキスト (autoregressive)
固有受容感覚 (28次元)    ─┘    (Backbone)  ─→ 離散アクショントークン (FAST, autoregressive)

※ 実際のモデルスペック (hf_qwen/config.json):
  - LLM: hidden_size=2560, intermediate_size=9216, 32層 (24 linear_attention + 8 full_attention)
  - Vision: hidden_size=1024, out_hidden_size=2560, 27層
  - vocab_size=248320, tie_word_embeddings=true
```

Pi0.5 Pre-training (α=0) に相当:
- VLMのみ（Action Expertなし）
- Cross-Entropy損失のみ（Flow Matchingなし）
- 離散アクショントークンによるアクション予測

---

## Phase 1: 最小構成（離散アクション + サブタスク）

### 1. データ前処理パイプライン設計

#### 1.1 LeRobotデータセットからの読み込み

**なぜ**: 既存のLeRobot v3.0データセット `robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket` はACT/Diffusion用に動作確認済み。これをベースに拡張する。

**現状のデータセット仕様**:
- 50エピソード、19083フレーム、30fps
- action: float32, shape=[28]（右腕7+グリッパー+右手先6D + 左腕7+グリッパー+左手先6D）
- observation.state: float32, shape=[28]
- カメラ3台: cam_head_rgb, cam_left_wrist_rgb, cam_right_wrist_rgb (480x640, AV1)
- タスク: "the left gripper grasp the basket on the table, the right grippe pick up the blocks on the table and place it into the basket."

**必要な作業**:

1. **カスタムDatasetクラスの作成** (`src/qwen_vla/data/dataset.py`)
   - LeRobotDatasetをラップし、以下を追加で返す:
     - 離散化済みアクショントークン（FASTエンコード済み）
     - サブタスクラベルテキスト
     - 固有受容感覚のテキスト表現
   - LeRobotの `LeRobotDataset` クラスを内部で使い、画像のデコード・前処理は既存機構を流用

2. **画像前処理**
   - Pi0.5に準拠: RandomCrop(95%), Resize(224x224 or 384x384), Rotate(±5°), ColorJitter
   - Qwen3.5のVision Encoderはpatch_size=16, spatial_merge_size=2 → 入力解像度は384x384程度が妥当（後で実験で決定）
   - 3カメラ画像を個別にエンコードし、VLMの入力シーケンスに連結

**推定作業量**: 2-3日

#### 1.2 アクションの離散化（FAST Tokenizer）

**なぜ**: Pi0.5のPre-trainingでは、アクションチャンク（連続値）をFAST Action Tokenizerで離散トークン列に変換し、VLMのテキスト生成と同じCross-Entropy損失で学習する。これにより、VLMの事前学習済み言語生成能力をアクション予測に転用できる。

**FAST Action Tokenizerの仕組み**:
```
アクションチャンク [chunk_size, action_dim]
    ↓ DCT変換（離散コサイン変換、軸=時間方向）
DCT係数 [chunk_size, action_dim]
    ↓ スケーリング + 量子化 → 整数列
整数列 [chunk_size * action_dim]
    ↓ BPEトークン化
離散トークン列 [num_tokens]  (通常 50〜256トークン)
```

**実装方針**: LeRobotのpi0_fastにFAST実装が存在するが、プリトレイン版(`lerobot/fast-action-tokenizer`)は28次元/30Hzのアクション分布に最適化されていない。**Realmanデータセット固有のカスタムFAST tokenizerを学習する必要がある**。

**具体的な手順**:

1. **カスタムFAST Action Tokenizerの学習**
   ```bash
   # Realmanデータセット用のFAST tokenizerを学習
   lerobot-train-tokenizer \
       --repo_id=robocoin/Realman_RMC_AIDA_L_storage_block_basket \
       --root=robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
       --action_horizon=50 \
       --encoded_dims="0:28" \
       --vocab_size=1024 \
       --scale=10.0 \
       --normalization_mode="QUANTILES" \
       --output_dir="./fast_tokenizer_realman_28d"
   ```

2. **学習済みFAST Action Tokenizerのロード**
   ```python
   from transformers import AutoProcessor
   action_tokenizer = AutoProcessor.from_pretrained(
       "./fast_tokenizer_realman_28d", trust_remote_code=True
   )
   ```

2. **アクションチャンクの構成**
   - chunk_size: 50（Pi0.5準拠）
   - action_dim: 28（当データセット固有）
   - 各タイムステップから先頭50フレームのアクションを切り出してチャンク化
   - 正規化: データセット統計量でmean/std正規化（Pi0FastのNormalizationMode.MEAN_STDと同じ）

3. **トークン化パイプライン** (`src/qwen_vla/data/action_tokenizer.py`)
   ```python
   def tokenize_action_chunk(actions, tokenizer, scale):
       """
       actions: [chunk_size, action_dim] numpy array, normalized to ~[-1,1]
       Returns: list of integer token IDs
       """
       from scipy.fftpack import dct
       # DCT変換
       dct_coeffs = dct(actions, axis=0, norm="ortho") * scale
       # 整数化（0-255範囲にクリップ）
       int_coeffs = np.clip(np.round(dct_coeffs), 0, 255).astype(int)
       # フラット化
       flat = int_coeffs.flatten()
       # BPEエンコード
       chars = "".join(chr(v - tokenizer.min_token) for v in flat)
       token_ids = tokenizer.bpe_tokenizer.encode(chars)
       return token_ids
   ```

4. **Qwen3.5 vocabへのマッピング**
   - Pi0Fastの方式に準拠: FASTトークンIDをQwen3.5のvocab末尾にマッピング
   - `qwen_token_id = qwen_vocab_size - 1 - skip_offset - fast_token_id`
   - これによりvocab拡張（embed_tokens/lm_headのリサイズ）が不要
   - skip_offset: 128（Pi0Fast準拠、vocab末尾の未使用トークン領域を利用）

**推定作業量**: 2-3日

#### 1.3 サブタスクラベルの生成/抽出

**なぜ**: Pi0.5はサブタスク予測（高レベルプランニング）を行い、そのサブタスクに基づいてアクションを生成する階層的アーキテクチャ。サブタスクラベルがないとこの階層構造を学習できない。

**現状のデータセット状況**:
- v3.0データセット: `subtask_annotation` フィールドあり（shape=[5], int32）
  - 最初の要素のみ有効なサブタスクインデックス、残り4要素は `4`（null/パディング）
- `annotations/subtask_annotations.jsonl` にマッピング定義あり

**サブタスク定義（実データ確認済み）**:
| ID | テキスト | フレーム数 |
|---|---|---|
| 0 | End | 1,148 |
| 1 | Grasp the blue cube with the right gripper | 3,858 |
| 2 | Place the blue cube into the basket with the right gripper | 6,142 |
| 3 | Grasp the basket with the left gripper | 7,935 |
| 4 | null（パディング値） | - |

**実装方針（2段階）**:

**段階A: subtask_annotationを直接利用（推奨、初期実装）**
- v3.0データセットの `subtask_annotation[0]` を取得
- 整数IDをテキストラベルに変換:
  ```python
  SUBTASK_MAP = {
      0: "end",
      1: "Grasp the blue cube with the right gripper",
      2: "Place the blue cube into the basket with the right gripper",
      3: "Grasp the basket with the left gripper",
      4: "",  # null/padding
  }
  ```
- ID=0 (End) と ID=4 (null) のフレームは損失計算から除外するか、特殊トークンで処理

**段階B: VLMによる自動アノテーション（将来、データ拡張時）**
- 画像列をQwen3.5（または他のVLM）に入力し、サブタスクテキストを自動生成
- Hindsight Relabeling的なアプローチ
- 新規データセット追加時に必要

**推奨**: 段階Aを最初から実装（v3.0に直接利用可能なラベルが存在するため）。

**実装ファイル**: `src/qwen_vla/data/subtask_labeler.py`

**推定作業量**: 段階A: 1日 / 段階B: 3-5日

---

### 2. モデルアーキテクチャ（最小構成）

#### 2.1 全体構成

```
┌──────────────────────────────────────────────────────────────────────┐
│                     QwenVLAForDiscreteAction                         │
│                                                                      │
│  ┌──────────────────┐                                                │
│  │ Qwen3_5VisionModel│ ─── 画像特徴量 ───┐                          │
│  │ (27層, 1024dim     │                    │                          │
│  │  →2560dim射影,凍結) │                    │                          │
│  └──────────────────┘                    ▼                           │
│                               ┌──────────────────────┐               │
│  テキスト入力 ──────────────→ │ Qwen3_5TextModel     │               │
│  (タスク指示+固有受容感覚)    │ (32層, 2560dim)      │               │
│                               │ 75% linear_attention │               │
│  サブタスクトークン ─────────→ │ (GatedDeltaNet)      │               │
│  <subtask> ... </subtask>     │ 25% full_attention   │               │
│                               └──────────┬───────────┘               │
│                                          │                           │
│  FASTアクショントークン ──→              │                           │
│  <action_start> tok1 tok2 ...            ▼                           │
│                               ┌──────────────────────┐               │
│                               │ lm_head (2560→248320)│               │
│                               │ (tie_word_embeddings) │               │
│                               └──────────┬───────────┘               │
│                                          │                           │
│                                          ▼                           │
│                               サブタスクテキスト + 離散アクショントークン│
└──────────────────────────────────────────────────────────────────────┘
```

**重要な設計判断**: Phase 1ではQwen3.5の既存アーキテクチャをほぼそのまま使う。追加するのは特殊トークンと入力フォーマットのみ。

#### 2.2 Qwen3.5 VLMの改修箇所

**なぜ**: Qwen3.5は既にVLM（画像+テキスト→テキスト）として完成している。Phase 1ではこれをVLA（画像+テキスト→サブタスク+アクショントークン）に最小限の変更で適応させる。

**改修箇所一覧**:

| 改修 | ファイル | 内容 | 規模 |
|------|----------|------|------|
| 特殊トークン追加 | tokenizer設定 | `<action_start>`, `<action_end>`, `<subtask>`, `</subtask>` | 小 |
| 入力フォーマット定義 | processor | プロンプトテンプレート | 小 |
| Attention Mask | modeling | VLA用のカスタムマスク（Phase 1は因果マスクのまま可） | 中（後回し可） |
| FASTトークンマッピング | processor | アクショントークンのvocabマッピング | 小 |
| Policyラッパー | 新規 | LeRobot PreTrainedPolicy準拠のラッパー | 中 |

**Phase 1で改修しないもの**:
- Vision Encoder: そのまま使用（凍結 or LoRA）
- LLMレイヤー構造: 変更なし（linear_attention + full_attention のハイブリッドのまま）
- embed_tokens / lm_head: リサイズ不要（FASTトークンはvocab末尾にマッピング）
- MTP (Multi-Token Prediction): config.jsonに `mtp_num_hidden_layers=1` が存在。Phase 1では無効化（`use_mtp=False`）して標準的なnext-token predictionのみ使用。MTP活用はPhase 2以降で検討

#### 2.3 アクショントークンのvocab統合

**なぜ**: FASTトークンを別vocabとして扱うとembed_tokens/lm_headのリサイズが必要になり、事前学習済み重みの再利用が面倒になる。Pi0Fastの方式（既存vocab末尾へのマッピング）に従えば変更が最小限。

**方式**: Pi0Fastの `_act_tokens_to_paligemma_tokens` と同じ方式をQwen3.5に適用

```python
class ActionTokenMapper:
    """FASTアクショントークンIDをQwen3.5のvocab IDにマッピング"""

    def __init__(self, qwen_vocab_size=248320, skip_tokens=128):
        self.qwen_vocab_size = qwen_vocab_size
        self.skip_tokens = skip_tokens

    def fast_to_qwen(self, fast_token_ids):
        """FAST token ID → Qwen3.5 token ID"""
        return self.qwen_vocab_size - 1 - self.skip_tokens - fast_token_ids

    def qwen_to_fast(self, qwen_token_ids):
        """Qwen3.5 token ID → FAST token ID"""
        return self.qwen_vocab_size - 1 - self.skip_tokens - qwen_token_ids
```

Qwen3.5のvocab_size=248320は十分大きく、末尾領域にFASTトークンをマッピング可能。FASTトークンは通常1024個程度なので、skip_tokens=128を加えてもtoken ID 247168〜248191の範囲に収まる。

**vocab末尾衝突チェック（必須）**: Qwen3.5のconfig.jsonに `image_token_id=248056`, `video_token_id=248057` 等のspecial tokenが248000番台に存在する。FASTトークンのマッピング範囲（247168〜248191）と衝突しないことを実装前に確認すること。衝突する場合は `skip_tokens` を増やして回避する。

```python
# 衝突チェックコード
special_tokens = [248056, 248057]  # image_token_id, video_token_id 等
fast_range = range(248320 - 1 - 128 - 1024, 248320 - 1 - 128 + 1)  # 247167〜248191
conflicts = [t for t in special_tokens if t in fast_range]
assert len(conflicts) == 0, f"Vocab衝突: {conflicts}"
```

**推定作業量**: 0.5日

#### 2.4 入力シーケンスフォーマット

**なぜ**: VLMの入力をどう構成するかがモデルの学習品質に直結する。Pi0.5の入力フォーマットに準拠しつつ、Qwen3.5のマルチモーダル入力形式に合わせる。

**プロンプトテンプレート**:
```
<|im_start|>system
You are a robot control assistant. Given camera images and a task instruction,
predict the next subtask and action sequence.<|im_end|>
<|im_start|>user
<|vision_start|><image_1><|vision_end|>
<|vision_start|><image_2><|vision_end|>
<|vision_start|><image_3><|vision_end|>
Task: {task_instruction}
State: {proprioceptive_state_discretized}<|im_end|>
<|im_start|>assistant
<subtask>{subtask_text}</subtask>
<action_start>{fast_action_tokens}<action_end><|im_end|>
```

**固有受容感覚のテキスト化**（Pi0.5準拠）:
```python
def proprioception_to_text(state, stats):
    """
    state: [28] float tensor (normalized to [-1,1] via mean/std)
    Returns: discretized state string
    """
    # 256ビンに離散化
    bins = np.linspace(-1, 1, 257)[:-1]
    discretized = np.digitize(state.cpu().numpy(), bins) - 1
    return " ".join(map(str, discretized))
```

**推定作業量**: 1日

#### 2.5 Forward Passの変更

**なぜ**: 学習時のForward Passを定義する。Phase 1では標準的なautoregressive言語モデルのnext-token predictionと同じ構造。

**学習時のForward Pass**:
```python
def forward(self, batch):
    # 1. 入力の構成
    #    - 3カメラ画像 → Vision Encoder → image_features
    #    - テキスト（タスク指示 + 固有受容感覚）→ text_tokens
    #    - サブタスクテキスト → subtask_tokens
    #    - FASTアクショントークン → action_tokens (Qwen vocab IDにマッピング済み)

    # 2. 全トークンを連結
    #    [image_tokens | text_tokens | subtask_tokens | action_tokens]
    #    ※ Qwen3.5の既存メカニズムで画像トークンは自動的にVision Encoder出力に置換される

    # 3. Qwen3.5 ForConditionalGeneration.forward() を呼ぶ
    #    - input_ids: 連結されたトークン列
    #    - attention_mask: 因果マスク（Phase 1）
    #    - labels: サブタスク部分 + アクション部分にのみ損失を計算
    #              （入力プロンプト部分はlabel=-100でマスク）

    # 4. 損失を返す
    #    Cross-Entropy loss（VLMの標準損失をそのまま利用）
    return loss, {"loss": loss.item(), "ce_loss": loss.item()}
```

**ラベルマスキング（重要）**:
- システムプロンプト → label=-100（損失計算しない）
- ユーザー入力（画像+タスク+固有受容感覚）→ label=-100
- サブタスクテキスト → 損失計算する
- FASTアクショントークン → 損失計算する

**推定作業量**: 2-3日

---

### 3. 学習スクリプト設計

#### 3.1 損失関数

**なぜ**: Phase 1はPi0.5のPre-trainingステージ（α=0）に対応。VLMの標準的なCross-Entropy損失のみ。

```python
# 損失計算は Qwen3.5 ForConditionalGeneration の内部ロジックをそのまま利用
# labels テンソルで損失対象をマスク制御

loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
# logits: [batch, seq_len, vocab_size]
# labels: [batch, seq_len]  (損失対象外は-100)
loss = loss_fct(logits.view(-1, vocab_size), labels.view(-1))
```

サブタスク予測損失とアクション予測損失を別々にログする場合:
```python
subtask_loss = loss_fct(
    logits[:, subtask_start:subtask_end].reshape(-1, vocab_size),
    labels[:, subtask_start:subtask_end].reshape(-1)
)
action_loss = loss_fct(
    logits[:, action_start:action_end].reshape(-1, vocab_size),
    labels[:, action_start:action_end].reshape(-1)
)
```

#### 3.2 ハイパーパラメータ

**Pi0.5 Pre-training準拠 + Qwen3.5向け調整**:

| パラメータ | 値 | 根拠 |
|---|---|---|
| learning_rate | 2.5e-5 | Pi0Fast準拠 |
| warmup_steps | 300 | データ量に比例して削減 (19083frames/bs4≒4771step/epoch) |
| decay_steps | 15000 | 約3エポック相当（過学習防止のため控えめ） |
| total_steps | 15000 | 約3エポック（early stoppingで調整） |
| batch_size | 4-8 | GPUメモリに応じて（128GB VRAM前提で8程度） |
| optimizer | AdamW | β1=0.9, β2=0.95, ε=1e-8 |
| weight_decay | 0.01 | Pi0Fast準拠 |
| grad_clip_norm | 1.0 | Pi0Fast準拠 |
| chunk_size | 50 | Pi0.5準拠 |
| max_action_tokens | 256 | Pi0Fast準拠 |
| image_resolution | 384x384 | Qwen3.5 Vision向け（要実験） |
| dtype | bfloat16 | メモリ効率 |
| gradient_checkpointing | true | メモリ節約 |
| validation_episodes | 5-10 | 50エピソード中から確保、early stopping判定用 |

**データ量に関する注意**: 19083フレーム / batch_size=4 = 約4771ステップ/epoch。15000ステップは約3エポック相当。過学習を避けるためearly stoppingを導入し、validation lossをモニタリングすること。

**凍結戦略（重要）**:
| コンポーネント | Phase 1 初期 | Phase 1 後期 |
|---|---|---|
| Vision Encoder (27層, 1024dim) | 凍結 | LoRA (rank=16) |
| LLM レイヤー 0-15 | 凍結 | LoRA (rank=16) |
| LLM レイヤー 16-31 | 学習 | 学習 |
| lm_head (tie_word_embeddings) | 学習 | 学習 |

**レイヤー種別と凍結の関係**:
- 32層中、full_attention は層 3,7,11,15,19,23,27,31（4層間隔で8層）
- linear_attention（GatedDeltaNet）は残り24層
- 凍結(0-15): full_attention 4層 + linear_attention 12層
- 学習(16-31): full_attention 4層 + linear_attention 12層
- linear_attention層のfine-tuning安定性は未検証のため、学習初期に勾配norm/分散をモニタリングすること

**なぜ凍結**: Qwen3.5は hidden_size=2560, 32層のモデル（推定約3Bパラメータ）。50エピソード/19083フレームという少量データでは全パラメータの学習は過学習リスクが極めて高い。Vision Encoderと下位LLMレイヤーは凍結し、上位レイヤーのみをfine-tuneする。

**tie_word_embeddingsに関する注意**: embed_tokensとlm_headが重み共有されている。FASTトークンをvocab末尾にマッピングすると、該当領域のembeddingも同時に更新される。事前学習で意味のある表現を持たないため、ファインチューニングで学習が必要。

**推定作業量**: 1日

#### 3.3 学習ループ

**実装方針**: LeRobotの学習フレームワーク（`lerobot-train`）に統合する。

**ファイル構成**:
```
src/qwen_vla/
├── __init__.py
├── config.py                    # QwenVLAConfig (PreTrainedConfig継承)
├── modeling.py                  # QwenVLAPolicy (PreTrainedPolicy継承)
├── processor.py                 # QwenVLAProcessor
├── data/
│   ├── __init__.py
│   ├── dataset.py               # QwenVLADataset
│   ├── action_tokenizer.py      # FAST Action Tokenizer ラッパー
│   └── subtask_labeler.py       # サブタスクラベル生成
└── utils.py                     # ユーティリティ
```

**学習スクリプト**: LeRobotの `lerobot-train` CLI に統合するか、独自の `scripts/train_qwen_vla.py` を作成。

推奨: まず独自スクリプトで動作確認 → 後でLeRobot CLI統合

```bash
# 独自スクリプトで学習
uv run python scripts/train_qwen_vla.py \
    --model_name Qwen/Qwen3.5-VL-3B  \
    --dataset_path robocoin/RoboCOIN/Realman_RMC_AIDA_L_storage_block_basket \
    --output_dir outputs/qwen_vla_discrete_v1 \
    --batch_size 4 \
    --learning_rate 2.5e-5 \
    --steps 30000 \
    --chunk_size 50 \
    --freeze_vision_encoder true \
    --gradient_checkpointing true \
    --dtype bfloat16
```

**学習ループ疑似コード**:
```python
for step, batch in enumerate(dataloader):
    # 1. バッチの構成
    images = batch["images"]           # [B, 3, C, H, W] (3カメラ)
    task_text = batch["task"]          # list of str
    state = batch["observation.state"] # [B, 28]
    actions = batch["action"]          # [B, chunk_size, 28]
    subtask = batch["subtask"]         # list of str

    # 2. 前処理
    processed = processor(images, task_text, state, subtask, actions)
    # processed contains: input_ids, attention_mask, pixel_values, labels

    # 3. Forward
    outputs = model(**processed)
    loss = outputs.loss

    # 4. Backward
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    # 5. ログ
    if step % log_interval == 0:
        log({"loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

    # 6. チェックポイント
    if step % save_interval == 0:
        save_checkpoint(model, optimizer, scheduler, step)
```

**推定作業量**: 3-4日

---

### 4. 推論パイプライン

#### 4.1 サブタスク予測 → 離散アクション生成

**なぜ**: 学習済みモデルを使って、画像+タスク指示から実際にアクションを生成する推論パイプラインが必要。

**推論フロー**:
```
1. 入力構成
   - 3カメラ画像をVision Encoderに通す
   - タスク指示テキスト + 固有受容感覚（離散化テキスト）を構成
   - プロンプトの「assistant」ターンの先頭まで入力

2. サブタスク予測（Autoregressive生成）
   - <subtask> トークンから生成開始
   - </subtask> が出るまでトークンを1つずつ生成
   - temperature=0（greedy）or temperature=0.5

3. アクショントークン予測（Autoregressive生成）
   - <action_start> トークンを追加
   - max_action_tokens=256 個まで生成
   - <action_end> が出たら停止

4. アクションのデコード（FAST逆変換）
   - 生成されたトークンID → FASTトークンID に逆変換
   - FASTトークンID → BPEデコード → 整数列
   - 整数列 → reshape [chunk_size, action_dim]
   - 逆DCT変換 → 連続アクション [chunk_size, action_dim]
   - 逆正規化 → 実アクション値

5. アクション実行
   - chunk_size=50 ステップを30Hzで順次実行
   - 全て実行したら再度画像を取得して1.に戻る
```

**推論コード** (`src/qwen_vla/inference.py`):
```python
class QwenVLAInference:
    def __init__(self, model, processor, action_tokenizer, config):
        self.model = model
        self.processor = processor
        self.action_tokenizer = action_tokenizer
        self.config = config
        self.token_mapper = ActionTokenMapper(
            qwen_vocab_size=config.qwen_vocab_size,
            skip_tokens=config.skip_tokens
        )

    @torch.no_grad()
    def predict(self, images, task_instruction, proprioceptive_state):
        """
        images: dict of camera_name -> [H, W, 3] numpy array
        task_instruction: str
        proprioceptive_state: [28] numpy array
        Returns: [chunk_size, action_dim] numpy array
        """
        # 1. 入力構成
        inputs = self.processor.prepare_inference_inputs(
            images, task_instruction, proprioceptive_state
        )

        # 2. Autoregressive生成（サブタスク + アクショントークン）
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.config.tokenizer_max_length + self.config.max_action_tokens,
            temperature=self.config.temperature,
            do_sample=(self.config.temperature > 0),
        )

        # 3. サブタスクとアクションの分離
        subtask_text, action_token_ids = self.parse_generated(generated_ids)

        # 4. アクションのデコード
        fast_token_ids = self.token_mapper.qwen_to_fast(action_token_ids)
        actions = self.decode_actions(
            fast_token_ids,
            time_horizon=self.config.chunk_size,
            action_dim=self.config.action_dim,
        )

        return subtask_text, actions

    def decode_actions(self, fast_token_ids, time_horizon, action_dim):
        """FAST逆変換: トークン → 連続アクション"""
        # Pi0FastPolicy.decode_actions_with_fast() と同じロジック
        decoded_tokens = self.action_tokenizer.bpe_tokenizer.decode(fast_token_ids)
        dct_coeffs = np.array(list(map(ord, decoded_tokens))) + self.action_tokenizer.min_token
        dct_coeffs = dct_coeffs.reshape(time_horizon, action_dim)
        actions = idct(dct_coeffs / self.action_tokenizer.scale, axis=0, norm="ortho")
        return actions
```

**推定作業量**: 2日

#### 4.2 KVキャッシュ最適化

**なぜ**: 推論時、サブタスク予測とアクション生成で入力プレフィックス（画像+テキスト）のKVキャッシュを共有できる。Qwen3.5のGatedDeltaNetはrecurrent推論をサポートしており、長いコンテキストでも高速。

**Phase 1での実装**: Qwen3.5の `generate()` メソッドが内部でKVキャッシュを自動管理するため、特別な実装は不要。`use_cache=True` を設定するだけ。

---

### 5. 具体的な実装ステップ（ファイル単位）

#### Step 1: プロジェクト骨格の作成（0.5日）

```
/workspace/qwen_vla_project/src/qwen_vla/
├── __init__.py
├── config.py
├── modeling.py
├── processor.py
├── data/
│   ├── __init__.py
│   ├── dataset.py
│   ├── action_tokenizer.py
│   └── subtask_labeler.py
├── inference.py
└── utils.py
```

**何が必要か**: ディレクトリ構造の作成、`__init__.py` の配置、`pyproject.toml` への依存追加（transformers, qwen3_5モデル）

#### Step 2: Configuration クラス（1日）

**ファイル**: `src/qwen_vla/config.py`

```python
@PreTrainedConfig.register_subclass("qwen_vla")
@dataclass
class QwenVLAConfig(PreTrainedConfig):
    # VLMバックボーン
    vlm_model_name: str = "Qwen/Qwen3.5-VL-3B"
    load_vlm_weights: bool = True

    # アクション空間
    action_dim: int = 28            # 当データセット
    chunk_size: int = 50            # Pi0.5準拠
    n_action_steps: int = 50
    max_action_tokens: int = 256

    # 状態空間
    max_state_dim: int = 32

    # FAST Tokenizer
    action_tokenizer_name: str = "lerobot/fast-action-tokenizer"
    fast_skip_tokens: int = 128

    # 画像
    image_resolution: tuple[int, int] = (384, 384)

    # テキスト
    tokenizer_max_length: int = 200
    temperature: float = 0.0

    # 凍結設定
    freeze_vision_encoder: bool = True
    freeze_llm_layers_below: int = 16  # 0-15を凍結

    # 学習設定
    optimizer_lr: float = 2.5e-5
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_weight_decay: float = 0.01
    optimizer_grad_clip_norm: float = 1.0
    scheduler_warmup_steps: int = 1000
    scheduler_decay_steps: int = 30000
    gradient_checkpointing: bool = True
    dtype: str = "bfloat16"

    # 正規化
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )
```

**なぜ**: LeRobotのポリシーフレームワーク（`PreTrainedConfig` 継承）に統合するため。

#### Step 3: FAST Action Tokenizer ラッパー（1日）

**ファイル**: `src/qwen_vla/data/action_tokenizer.py`

**内容**:
- FAST tokenizerのロードとキャッシュ
- アクションチャンクのエンコード（連続値 → トークンID列）
- アクションチャンクのデコード（トークンID列 → 連続値）
- Qwen3.5 vocabへのマッピング/逆マッピング

**何が必要か**:
- `transformers` の `AutoProcessor` （FAST tokenizerのロード）
- `scipy` （DCT/IDCT変換）
- Pi0FastPolicy の `decode_actions_with_fast` メソッドのロジックを移植

#### Step 4: データセットクラス（2日）

**ファイル**: `src/qwen_vla/data/dataset.py`

**内容**:
- LeRobotDataset のラッパー
- `__getitem__` で以下を返す:
  - 3カメラ画像（前処理済み）
  - タスク指示テキスト
  - 固有受容感覚（normalized state）
  - アクションチャンク（normalized actions, chunk_size分）
  - FASTアクショントークン（エンコード済み）
  - サブタスクテキスト
- データ拡張（画像のRandomCrop, Resize, ColorJitter）

**何が必要か**:
- LeRobotDataset クラスのインポートと使用方法の理解
- action_tokenizer.py のエンコード機能
- subtask_labeler.py のラベル生成機能

#### Step 5: Processor（入力フォーマッタ）（2日）

**ファイル**: `src/qwen_vla/processor.py`

**内容**:
- Qwen3.5のprocessor（`AutoProcessor.from_pretrained("Qwen/Qwen3.5-VL-3B")`）をラップ
- 学習時: バッチデータ → model入力（input_ids, attention_mask, pixel_values, labels）
- 推論時: 生の画像+テキスト → model入力

**主要メソッド**:
```python
class QwenVLAProcessor:
    def prepare_training_inputs(self, images, task, state, subtask, action_tokens):
        """学習用入力を構成"""
        # プロンプトテンプレートにデータを埋め込み
        # Qwen3.5 Processorでトークン化
        # labelsを構成（入力部分は-100、出力部分にトークンID）
        ...

    def prepare_inference_inputs(self, images, task, state):
        """推論用入力を構成"""
        # プロンプトのアシスタントターン直前まで
        ...
```

**何が必要か**:
- Qwen3.5のProcessor/Tokenizerの使い方
- マルチ画像入力の構成方法
- labelsのマスキングロジック

#### Step 6: Policy クラス（2-3日）

**ファイル**: `src/qwen_vla/modeling.py`

**内容**:
- `QwenVLAPolicy(PreTrainedPolicy)` クラス
- `__init__`: Qwen3.5モデルのロード、凍結設定、FAST tokenizerロード
- `forward(batch)`: 学習時のForward Pass（損失計算）
- `select_action(batch)`: 推論時のアクション生成
- `reset()`: エピソードリセット

**コア実装**:
```python
class QwenVLAPolicy(PreTrainedPolicy):
    config_class = QwenVLAConfig
    name = "qwen_vla"

    def __init__(self, config: QwenVLAConfig):
        super().__init__(config)

        # Qwen3.5 VLMのロード
        from transformers import Qwen3_5_VLForConditionalGeneration, AutoProcessor
        self.vlm = Qwen3_5_VLForConditionalGeneration.from_pretrained(
            config.vlm_model_name,
            torch_dtype=torch.bfloat16 if config.dtype == "bfloat16" else torch.float32,
        )

        # 凍結
        if config.freeze_vision_encoder:
            for p in self.vlm.visual.parameters():
                p.requires_grad = False
        for i, layer in enumerate(self.vlm.model.layers):
            if i < config.freeze_llm_layers_below:
                for p in layer.parameters():
                    p.requires_grad = False

        # Gradient Checkpointing
        if config.gradient_checkpointing:
            self.vlm.gradient_checkpointing_enable()

        # Processor & Tokenizer
        self.processor = QwenVLAProcessor(config)
        self.action_tokenizer = FastActionTokenizerWrapper(config)

    def forward(self, batch):
        """学習時: Cross-Entropy損失を計算"""
        inputs = self.processor.prepare_training_inputs(batch)
        outputs = self.vlm(**inputs)
        loss = outputs.loss
        return loss, {"loss": loss.item()}

    @torch.no_grad()
    def select_action(self, batch):
        """推論時: アクション生成"""
        inputs = self.processor.prepare_inference_inputs(batch)
        generated = self.vlm.generate(**inputs, max_new_tokens=512)
        subtask, actions = self.decode_output(generated)
        return actions
```

#### Step 7: 学習スクリプト（1-2日）

**ファイル**: `scripts/train_qwen_vla.py`

**内容**:
- コマンドライン引数の定義
- データセットのロード
- モデルの初期化
- 学習ループ（Step 3.3参照）
- WandB/TensorBoardログ
- チェックポイント保存

**何が必要か**:
- `uv` 仮想環境でのtransformers + Qwen3.5モデルの利用可能性確認
- GPUメモリのプロファイリング（Qwen3.5-VL-3B + batch_size=4 で VRAM使用量の見積もり）

#### Step 8: 推論スクリプト（1日）

**ファイル**: `scripts/eval_qwen_vla.py`

**内容**:
- チェックポイントのロード
- テストエピソードでの推論
- サブタスク予測の定性評価
- アクション予測精度の定量評価（L2距離、成功率等）

---

### 実装順序とマイルストーン

```
Week 1:
  Step 1 (骨格) → Step 2 (Config) → Step 3 (FAST Tokenizer)
  マイルストーン: アクションのエンコード/デコードがend-to-endで動作

Week 2:
  Step 4 (Dataset) → Step 5 (Processor)
  マイルストーン: データパイプラインが動作、1バッチを構成して中身を確認

Week 3:
  Step 6 (Policy) → Step 7 (学習スクリプト)
  マイルストーン: 学習ループが回り、lossが減少する

Week 4:
  Step 8 (推論) → 評価 → バグ修正 → ハイパーパラメータ調整
  マイルストーン: 推論パイプラインが動作、定性的にアクションが妥当
```

**合計推定作業量**: 約3-4週間

---

## Phase 2以降のロードマップ概要

### Phase 2: Action Expert + Flow Matching追加

**目的**: Pi0.5のPost-trainingステージを実装。離散トークン予測に加え、連続アクション空間でのFlow Matching損失を追加する。

**主要タスク**:

1. **Action Expert Transformer の実装** (`action_expert.py`)
   - 初期: 既存のgemma_300m構成を流用（hidden=1024, depth=18, intermediate=4096, 約300Mパラメータ）
     - VLM hidden(2560) → Action Expert hidden(1024) は線形射影層で吸収（pi0_fastと同方式）
   - 拡張案: hidden=1280（VLM hidden/2比率、約420Mパラメータ）
   - いずれもfull_attentionのみ（linear_attentionは双方向attention非対応のため不使用）
   - 推定規模: 大（2-3週間）

2. **Adaptive RMSNorm の実装**
   - Flow Matchingのtimestepで条件付けされたRMSNorm
   - Action Expertの各レイヤーのRMSNormを置換
   - 推定規模: 小（2-3日）

3. **Flow Matching モジュール** (`flow_matching.py`)
   - ノイズスケジュール: `a_noisy = τ * a_clean + (1-τ) * ω`
   - タイムステップサンプリング: Beta分布
   - 推論: 10ステップEuler積分
   - 推定規模: 中（1-2週間）

4. **Attention Maskの本格実装**
   - 画像=双方向、テキスト=因果的、FASTアクション=因果的、Action Expert=双方向
   - VLM→Action Expert一方向のみ
   - 推定規模: 中（1週間）

5. **KVキャッシュ拡張**
   - VLMプレフィックスのキャッシュをAction Expertの10 denoising stepで再利用
   - 推定規模: 中（1週間）

6. **損失関数の拡張**
   ```python
   total_loss = ce_loss + α * flow_matching_loss  # α=10.0
   ```

**推定期間**: 6-8週間

### Phase 3: Vision Encoder交換実験

**目的**: Qwen3.5のデフォルトVision Encoder（Qwen3_5VisionModel）をRADIO等の高性能Vision Encoderに交換し、性能を比較する。

**主要タスク**:

1. **RADIOの統合テスト**
   - RADIO (AM-RADIO): マルチ教師蒸留済みVision Encoder
   - Qwen3.5のPatch Mergerとの接続方法の検討
   - 出力次元の調整（Linear projection）

2. **A/B比較実験**
   - 同じデータ・同じ学習設定で、Vision Encoderのみ変更
   - 評価指標: アクション予測精度、サブタスク予測精度、推論速度

**推定期間**: 3-4週間

### Phase 4: スケーリングと最適化

1. **データ拡張**: 追加データセットの統合（マルチロボット、マルチタスク）
2. **分散学習**: DeepSpeed/FSDP対応
3. **推論最適化**: vLLM等による高速推論
4. **実機デプロイ**: Realmanロボットでのリアルタイム推論テスト

---

## 補足: 既存コードベースとの関係

### LeRobot ポリシー実装パターン

LeRobotの既存VLAポリシー（pi0_fast, smolvla, xvla）から学べるパターン:

| パターン | pi0_fast | smolvla | 本実装 |
|---|---|---|---|
| VLMバックボーン | PaliGemma (Gemma 2B) | SmolVLM (500M) | Qwen3.5-VL (3B) |
| アクション表現 | FAST離散トークン | Flow Matching連続値 | FAST離散トークン (Phase 1) |
| 損失 | Cross-Entropy | MSE (Flow Matching) | Cross-Entropy (Phase 1) |
| Config基底 | PreTrainedConfig | PreTrainedConfig | PreTrainedConfig |
| Policy基底 | PreTrainedPolicy | PreTrainedPolicy | PreTrainedPolicy |
| Processor | カスタム (pi0_fast固有) | カスタム | カスタム |

### transformers-qwen3_5 サブモジュール

`/workspace/qwen_vla_project/transformers-qwen3_5/` にQwen3.5のtransformers実装がある:
- `modeling_qwen3_5.py` (2268行): 本体モデル
- `configuration_qwen3_5.py`: Config
- `tokenization_qwen3_5.py`: Tokenizer

Phase 1ではこれらを直接改修せず、HuggingFaceの公式transformersパッケージ経由でQwen3.5を利用する。Phase 2以降でAttention Mask等のカスタマイズが必要になった場合にforkして改修する。

### 注意: linear_attention (GatedDeltaNet) のfine-tuning特性

Qwen3.5の32層中24層がlinear_attention（GatedDeltaNet + Conv1d kernel=4）であり、標準的なtransformer attentionとは異なる勾配特性を持つ。VLA fine-tuningにおける安定性は未検証であるため、以下のモニタリングを推奨:

1. 学習初期にlinear_attention層とfull_attention層の勾配normを個別にログ
2. 異常な勾配爆発/消失が見られた場合、層ごとの学習率調整を検討
3. `partial_rotary_factor=0.25`（RoPEがhead_dimの25%にのみ適用）の影響で、長いアクショントークン列（最大256トークン）の位置表現に制約がある可能性がある

---

*作成日: 2026-03-27*
*最終更新: 2026-03-27*
