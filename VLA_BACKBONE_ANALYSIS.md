# Qwen3.5 を pi0.5 型 VLA Backbone にするための設計文書

## 目次
1. [pi0.5 アーキテクチャ要約](#1-pi05-アーキテクチャ要約)
2. [Qwen3.5 現状アーキテクチャ](#2-qwen35-現状アーキテクチャ)
3. [対応関係マッピング](#3-対応関係マッピング)
4. [必要な実装変更の詳細タスク一覧](#4-必要な実装変更の詳細タスク一覧)
5. [各タスクの技術的詳細](#5-各タスクの技術的詳細)
6. [未解決の設計判断事項](#6-未解決の設計判断事項)

---

## 1. pi0.5 アーキテクチャ要約

### 全体構成
pi0.5 は **VLM Backbone + Action Expert** の2コンポーネント構成:

```
┌────────────────────────────────────────────────────────────────┐
│                        VLM Backbone                            │
│  ┌──────────┐   ┌──────────────────────────┐                   │
│  │ SigLIP   │   │    Gemma 2B (LLM)        │                   │
│  │ (400M)   │──>│  18層, hidden=2048        │──> テキスト出力   │
│  │ Vision   │   │  MLP=16384, heads=18      │    (サブタスク予測)│
│  │ Encoder  │   │  KV heads=1, head_dim=256 │                   │
│  └──────────┘   └──────────┬───────────────┘                   │
│                             │ (hidden states)                   │
│                             ▼                                   │
│  ┌──────────────────────────────────────────┐                   │
│  │       Action Expert (300M)              │                   │
│  │  18層, hidden=1024, MLP=4096            │                   │
│  │  heads=18, KV heads=1, head_dim=256     │                   │
│  │  + Adaptive RMSNorm (timestep注入)       │                   │
│  │  + Flow Matching (10 denoising steps)   │                   │
│  └──────────────────────────┬──────────────┘                   │
│                             ▼                                   │
│                     連続アクション出力                            │
│                     (H=50ステップ chunk)                         │
└────────────────────────────────────────────────────────────────┘
```

### pi0.5 の主要パラメータ
| パラメータ | VLM Backbone | Action Expert |
|---|---|---|
| hidden_size | 2048 | 1024 |
| num_layers | 18 | 18 |
| mlp_dim | 16384 | 4096 |
| num_heads | 18 | 18 |
| num_kv_heads | 1 | 1 |
| head_dim | 256 | 256 |
| パラメータ数 | ~2.6B (SigLIP含む) | ~300M |

### 入力・出力仕様
- **画像入力**: 最大4台カメラ (前方/後方/左手首/右手首)
- **テキスト入力**: タスク指示 + 制御モード指定
- **固有受容感覚**: テキストトークンとして離散化して入力
- **アクション出力**: 連続値, H=50ステップ, 最大18-19DoF, [-1,1]正規化
- **制御周波数**: 50Hz

### 訓練戦略
| ステージ | ステップ数 | 内容 |
|---|---|---|
| Pre-training | 280k | 離散トークンのみ (FAST tokenizer), α=0 |
| Post-training | 80k | 離散 + Flow Matching 併用, α=10.0, Action Expert追加 |

### Flow Matching 詳細
- ノイズ付加: `a^{τ,ω} = τ * a + (1-τ) * ω`, ω ~ N(0,I)
- 学習対象: ベクトル場 `ω - a`
- タイムステップ分布: `Beta((s-τ)/s; α=1.5, β=1)`, s=0.999
- 推論: 10 denoising steps

### Attention パターン
- 画像トークン: **双方向 (bidirectional)**
- テキストトークン: サブタスク予測は **因果的 (causal/autoregressive)**
- FASTアクショントークン: prefix + 前のアクションへの因果的attention
- Action Expert トークン: prefix + 他のaction expert トークンへの **双方向** attention
- **VLM → Action Expert のみ一方向** (Action Expert は VLM の hidden states を参照するが逆はない)

---

## 2. Qwen3.5 現状アーキテクチャ

### 構成概要
```
Qwen3_5ForConditionalGeneration
├── Qwen3_5VisionModel (Vision Encoder)
│   ├── Qwen3_5VisionPatchEmbed (Conv3d: [2,16,16])
│   ├── position_embedding (nn.Embedding: 2304 x 1152)
│   ├── 27 x Qwen3_5VisionBlock (full attention, 16 heads)
│   └── Qwen3_5VisionPatchMerger → out_hidden_size=3584
│
├── Qwen3_5TextModel (LLM Backbone)
│   ├── embed_tokens (248320 x 4096)
│   ├── 32 x Qwen3_5DecoderLayer
│   │   ├── [75% linear_attention]: Qwen3_5GatedDeltaNet
│   │   └── [25% full_attention]: Qwen3_5Attention (GQA)
│   │   └── Qwen3_5MLP (SwiGLU)
│   └── norm (RMSNorm)
│
└── lm_head (4096 → 248320)
```

### Qwen3.5 主要パラメータ

**テキストモデル (LLM)**:
| パラメータ | 値 |
|---|---|
| vocab_size | 248320 |
| hidden_size | 4096 |
| intermediate_size (MLP) | 12288 |
| num_hidden_layers | 32 |
| num_attention_heads | 16 |
| num_key_value_heads | 4 (GQA) |
| head_dim | 256 |
| max_position_embeddings | 32768 |
| partial_rotary_factor | 0.25 (64/256 dim にのみ RoPE) |
| mrope_section | [11,11,10] (T/H/W 周波数配分) |
| layer_types | 75% linear (GatedDeltaNet) + 25% full attention |

**ビジョンモデル**:
| パラメータ | 値 |
|---|---|
| depth | 27 |
| hidden_size | 1152 |
| num_heads | 16 |
| patch_size | 16 |
| temporal_patch_size | 2 |
| spatial_merge_size | 2 |
| out_hidden_size | 3584 |

### Qwen3.5 の特徴的な機構
1. **ハイブリッド Attention**: 75% が GatedDeltaNet (線形attention, O(1)推論), 25% が full GQA
2. **MRoPE**: 4D位置ID [text, temporal, height, width] でマルチモーダル位置エンコーディング
3. **Partial RoPE**: head_dim の 25% (64次元) のみにRoPE適用、残り75%は位置無関係
4. **ゲート付きAttention出力**: Q投影が2倍幅で、半分がゲートとして機能
5. **Vision Patch Merger**: 4隣接パッチ → 3584次元に統合してLLMへ注入
6. **GatedDeltaNet の Conv1d**: kernel_size=4 の因果的畳み込みをQKV前に適用

---

## 3. 対応関係マッピング

| pi0.5 コンポーネント | Qwen3.5 での対応/代替 | ギャップ |
|---|---|---|
| SigLIP (400M) Vision Encoder | Qwen3_5VisionModel (27層, 1152dim) | ✅ そのまま利用可能。ただし出力次元が異なる (3584 vs 2048) |
| Gemma 2B LLM | Qwen3_5TextModel (32層, 4096dim) | ✅ より大きく高性能。ハイブリッドattentionは高速推論に有利 |
| PaliGemma統合 | Qwen3_5ForConditionalGeneration | ✅ 既にVLM統合済み |
| Action Expert (300M) | **未実装 → 新規追加必要** | ❌ コア実装が必要 |
| Flow Matching | **未実装 → 新規追加必要** | ❌ コア実装が必要 |
| Adaptive RMSNorm (timestep注入) | **未実装 → 新規追加必要** | ❌ 新規モジュール |
| FAST Action Tokenizer | **未実装 → 新規追加必要** | ❌ pre-training用 |
| アクション埋め込み (Linear) | **未実装 → 新規追加必要** | ❌ |
| 固有受容感覚のトークン化 | テキストトークンとして入力可能 | ⚠️ トークン化ロジックの追加 |
| カスタム Attention Mask | 部分的に対応可能 | ⚠️ VLA用のマスクパターン追加が必要 |
| 階層的推論 (高レベル→低レベル) | 推論パイプライン未実装 | ❌ 推論ロジック全体 |

---

## 4. 必要な実装変更の詳細タスク一覧

### Phase A: 基盤モジュールの追加 (新規ファイル・クラス)

#### A-1. Action Expert Transformer の実装
- **ファイル**: `modeling_qwen3_5.py` に新クラス追加 or 新ファイル `action_expert.py`
- **内容**:
  - `Qwen3_5ActionExpert(nn.Module)` クラス
  - VLM の DecoderLayer と同じ構造だがサイズを縮小
  - 推奨パラメータ (pi0.5 比率に基づくスケーリング):

| パラメータ | pi0.5 Action Expert | Qwen3.5 VLA Action Expert (提案) |
|---|---|---|
| hidden_size | 1024 (VLM の 1/2) | 2048 (VLM 4096 の 1/2) |
| num_layers | 18 (VLM と同じ) | 18〜24 (要実験) |
| mlp_dim | 4096 (VLM の 1/4) | 6144 (VLM 12288 の 1/2) |
| num_heads | 18 | 16 (VLM と揃える) |
| num_kv_heads | 1 | 4 (VLM と揃える or 1に削減) |
| head_dim | 256 | 256 (VLM と同じ) |

  - **設計判断**: GatedDeltaNet (線形attention) を Action Expert にも採用するか → 50Hz リアルタイム推論には有利

#### A-2. Adaptive RMSNorm の実装
- **ファイル**: `modeling_qwen3_5.py` 内に新クラス
- **内容**:
  ```python
  class Qwen3_5AdaptiveRMSNorm(nn.Module):
      """Flow Matching の timestep τ で条件付けされた RMSNorm"""
      def __init__(self, hidden_size, eps=1e-6):
          self.norm = Qwen3_5RMSNorm(hidden_size, eps)
          # timestep MLP: sinusoidal_encoding → swish(W1) → swish(W2) → scale, shift
          self.timestep_mlp = nn.Sequential(
              SinusoidalPositionalEncoding(hidden_size),
              nn.Linear(hidden_size, hidden_size),
              nn.SiLU(),
              nn.Linear(hidden_size, hidden_size * 2),  # scale + shift
          )

      def forward(self, x, timestep):
          scale, shift = self.timestep_mlp(timestep).chunk(2, dim=-1)
          return self.norm(x) * (1 + scale) + shift
  ```
- pi0.5 では各 Action Expert レイヤーの RMSNorm を Adaptive 版に置換

#### A-3. Flow Matching モジュールの実装
- **ファイル**: 新規 `flow_matching.py`
- **内容**:
  - `FlowMatchingScheduler`: ノイズスケジュール管理
    - 順方向: `a_noisy = τ * a_clean + (1-τ) * ω`
    - タイムステップサンプリング: `Beta((s-τ)/s; α=1.5, β=1)`, s=0.999
  - `FlowMatchingLoss`: MSE損失 `||ω - a - f_θ(a_noisy, o, l)||²`
  - `FlowMatchingInference`: 10ステップ Euler 積分による推論
    ```python
    def sample(self, model, observation, language, num_steps=10):
        a = torch.randn(batch, H, action_dim)  # 初期ノイズ
        dt = 1.0 / num_steps
        for i in range(num_steps):
            tau = i * dt
            v = model(a, observation, language, tau)  # ベクトル場予測
            a = a + v * dt
        return a
    ```

#### A-4. アクション埋め込み/投影レイヤー
- **ファイル**: `modeling_qwen3_5.py`
- **内容**:
  - `action_input_proj`: `nn.Linear(action_dim, action_expert_hidden_size)` — ノイジーアクションを Action Expert の hidden_size に投影
  - `action_output_proj`: `nn.Linear(action_expert_hidden_size, action_dim)` — 予測されたベクトル場をアクション空間に投影
  - アクション正規化: 各次元を [-1, 1] に正規化するための統計量管理

#### A-5. FAST Action Tokenizer の統合
- **ファイル**: 新規 `action_tokenizer.py`
- **内容**:
  - アクションチャンク (H=50, d=action_dim) を離散トークン列に変換
  - 圧縮ベースのトークン化 (FAST: Pertsch et al.)
  - vocab に FAST トークンを追加 (Qwen3.5 の vocab_size=248320 に追記)
  - Pre-training ステージで使用、Post-training では補助損失として併用

---

### Phase B: 既存モジュールの改修

#### B-1. Configuration の拡張
- **ファイル**: `configuration_qwen3_5.py`
- **変更内容**:
  ```python
  class Qwen3_5VLAConfig(Qwen3_5Config):
      """VLA用の拡張Config"""
      # Action Expert 設定
      action_expert_hidden_size: int = 2048
      action_expert_num_layers: int = 18
      action_expert_intermediate_size: int = 6144
      action_expert_num_heads: int = 16
      action_expert_num_kv_heads: int = 4
      action_expert_head_dim: int = 256

      # Action 空間設定
      action_dim: int = 19            # 最大 DoF
      action_chunk_size: int = 50     # H=50

      # Flow Matching 設定
      flow_matching_num_steps: int = 10  # 推論時 denoising steps
      flow_matching_beta_alpha: float = 1.5
      flow_matching_beta_beta: float = 1.0
      flow_matching_s: float = 0.999
      flow_matching_loss_weight: float = 10.0  # α=10.0

      # FAST Tokenizer 設定
      fast_vocab_size: int = 1024     # FAST トークン数

      # 推論設定
      num_cameras: int = 4            # 最大カメラ数
      control_frequency: float = 50.0  # Hz
  ```

#### B-2. Attention Mask のカスタマイズ
- **ファイル**: `modeling_qwen3_5.py`
- **変更箇所**: `Qwen3_5TextModel.forward()` 内のマスク生成
- **必要なマスクパターン**:

```
             | 画像 | テキスト | Propri | FAST Act | Action Expert |
画像          | BI   | -        | -      | -        | -             |
テキスト      | ✓    | CAUSAL   | ✓      | -        | -             |
Propri       | ✓    | ✓        | BI     | -        | -             |
FAST Action  | ✓    | ✓        | ✓      | CAUSAL   | -             |
Action Expert| ✓    | ✓        | ✓      | -        | BI            |
```

  - BI = 双方向 (Bidirectional)
  - CAUSAL = 因果的
  - ✓ = 参照可能
  - `-` = 参照不可

  **具体的な変更点**:
  - `_update_causal_mask()` メソッドを拡張し、トークン種別ごとにマスクを切り替え
  - `mm_token_type_ids` に新しいトークンタイプを追加 (ACTION_EXPERT_TYPE, FAST_ACTION_TYPE, PROPRIOCEPTION_TYPE)

#### B-3. Forward Pass の拡張
- **ファイル**: `modeling_qwen3_5.py`
- **変更箇所**: `Qwen3_5ForConditionalGeneration.forward()`
- **変更内容**:
  1. 新しい入力引数の追加:
     - `action_tokens`: `(batch, H, action_dim)` — ノイジーアクション入力
     - `flow_timestep`: `(batch,)` — flow matching のタイムステップ τ
     - `proprioceptive_state`: `(batch, propri_dim)` — ロボット固有受容感覚
     - `target_actions`: `(batch, H, action_dim)` — 教師アクション (学習時)
  2. Forward の流れ:
     ```
     a) テキスト・画像・固有受容感覚を通常通り埋め込み
     b) VLM Backbone を forward → hidden_states を取得
     c) hidden_states の prefix 部分を Action Expert に渡す
     d) ノイジーアクションを action_input_proj で埋め込み
     e) Action Expert を forward (prefix hidden + action embeddings)
     f) action_output_proj でベクトル場を出力
     g) テキスト損失 + Flow Matching 損失を合算して返す
     ```

#### B-4. VLM → Action Expert の hidden state 受け渡し
- **ファイル**: `modeling_qwen3_5.py`
- **設計オプション**:

  **オプション1: 共有 Attention (pi0.5 方式)**
  - Action Expert のトークンを VLM のシーケンスに追加し、カスタムマスクで一方向のみ許可
  - 利点: 実装がシンプル、pi0.5 忠実
  - 欠点: VLM の全レイヤーを Action Expert トークンも通過する (計算コスト)

  **オプション2: Cross-Attention (分離方式)**
  - Action Expert に cross-attention レイヤーを追加し、VLM の最終 hidden states をキーバリューとして参照
  - 利点: VLM と Action Expert を完全分離でき、独立にバッチ処理可能
  - 欠点: pi0.5 と異なるアーキテクチャ

  **推奨**: pi0.5 論文に忠実に **オプション1 (共有 Attention)** を採用

#### B-5. 位置エンコーディングの拡張
- **ファイル**: `modeling_qwen3_5.py`
- **変更内容**:
  - MRoPE の 4D position_ids `[text, T, H, W]` に Action Expert トークン用の位置割り当てを追加
  - Action Expert トークンは時間的な順序を持つため、temporal 次元のみに連番を割り当て
  - `get_rope_index()` メソッドを拡張して action token の位置計算を追加

#### B-6. KV Cache の拡張
- **ファイル**: `modeling_qwen3_5.py`
- **変更内容**:
  - `Qwen3_5DynamicCache` を拡張し、Action Expert のキャッシュも管理
  - Flow Matching の各 denoising step で VLM prefix のキャッシュを再利用
  - **重要**: 10回の denoising step で VLM を再計算せず、キャッシュされた prefix hidden states のみ参照

---

### Phase C: 訓練パイプライン

#### C-1. 2段階訓練ループの実装
- **ファイル**: 新規 `train_vla.py`
- **内容**:

  **Stage 1: Pre-training**
  - VLM のみ (Action Expert なし)
  - 離散トークン予測 (FAST tokenizer)
  - Cross-entropy 損失のみ (α=0)
  - データ: ロボット操作データ + Web データ (画像キャプション, VQA 等)

  **Stage 2: Post-training**
  - VLM + Action Expert (Action Expert はランダム初期化)
  - Cross-entropy + Flow Matching MSE 損失 (α=10.0)
  - 損失: `L = CE(text_pred, text_target) + 10.0 * MSE(v_pred, v_target)`
  - データ: 成功エピソードのみ、長さ閾値以下にフィルタ

#### C-2. データローダーの実装
- **ファイル**: 新規 `data/`
- **内容**:
  - マルチカメラ画像の読み込みとaugmentation
    - RandomCrop (95%), Resize, Rotate (±5°), ColorJitter (b=0.3, c=0.4, s=0.5)
  - アクションチャンクの構成 (H=50)
  - 固有受容感覚のテキスト化
  - アクション正規化 (1%/99% quantile per dimension)
  - FAST tokenizer による離散化 (pre-training 用)

#### C-3. 損失関数の統合
- **ファイル**: `modeling_qwen3_5.py` の forward 内
- **内容**:
  ```python
  # テキスト損失 (Cross-Entropy)
  text_loss = F.cross_entropy(text_logits, text_targets)

  # Flow Matching 損失 (MSE)
  # target = ω - a (ノイズ - クリーンアクション)
  flow_target = noise - clean_actions
  flow_loss = F.mse_loss(predicted_vector_field, flow_target)

  # 合計損失
  total_loss = text_loss + alpha * flow_loss  # alpha=10.0
  ```

---

### Phase D: 推論パイプライン

#### D-1. 階層的推論の実装
- **ファイル**: 新規 `inference_vla.py`
- **内容**:

  ```
  ステップ1: 高レベル推論 (サブタスク予測)
    入力: 4カメラ画像 + タスク指示文
    処理: VLM で autoregressive にテキスト生成
    出力: サブタスク記述 (例: "pick up the plate")

  ステップ2: 低レベル推論 (アクション生成)
    入力: 3カメラ画像 + サブタスク記述 + 固有受容感覚
    処理:
      a) VLM で prefix hidden states を計算 (1回のみ)
      b) ランダムノイズ a ~ N(0,I) を生成
      c) 10回の denoising step:
         for i in range(10):
           τ = i / 10
           v = action_expert(a, prefix_cache, τ)
           a = a + v * (1/10)
      d) アクションチャンク a を出力
    出力: H=50 ステップの連続アクション

  ステップ3: アクション実行
    50Hz で順次実行、必要に応じてアクションチャンクを再生成
  ```

#### D-2. リアルタイム推論の最適化
- **内容**:
  - VLM prefix の KV キャッシュ再利用 (10 denoising steps で VLM は1回のみ計算)
  - Action Expert のみが 10回 forward (300M相当なので高速)
  - GatedDeltaNet の O(1) recurrent 推論の活用
  - 推論時は FAST トークン不要 (flow matching のみ)

---

## 5. 各タスクの技術的詳細

### 5.1 GatedDeltaNet を Action Expert に採用する場合の考慮事項

Qwen3.5 の特徴的な GatedDeltaNet (線形 attention) は、Action Expert にも採用する価値がある:

**利点**:
- O(1) per-step 推論 → 50Hz リアルタイム制御に最適
- recurrent state が implicit memory として機能 → 過去の文脈を効率的に保持
- Conv1d (kernel=4) が短期的なアクション系列のパターン認識に有効

**課題**:
- Flow Matching の各 denoising step で recurrent state をリセットする必要があるか?
- Action Expert のトークンは双方向 attention → GatedDeltaNet は因果的 → full attention レイヤーのみ使用?

**推奨**:
- Action Expert は **full attention のみ** (GatedDeltaNet なし) にする
- 理由: Action Expert のトークン間は双方向 attention が必要で、GatedDeltaNet の因果的性質と合わない

### 5.2 Vision Encoder の出力次元の調整

| | pi0.5 | Qwen3.5 |
|---|---|---|
| Vision 出力次元 | 2048 (= VLM hidden_size) | 3584 (out_hidden_size) |
| VLM hidden_size | 2048 | 4096 |

- Qwen3.5 の Vision Patch Merger は 3584 次元を出力し、hidden_size=4096 との間にギャップ
- `Qwen3_5Model._get_image_features()` 内で隠れ層への投影が行われているはず
- 追加対応は不要 (既存の仕組みを利用)

### 5.3 Vocab 拡張

Pre-training で FAST tokenizer を使う場合:
- 現行 vocab_size: 248320
- FAST トークン追加: +1024 (要調整)
- 特殊トークン追加:
  - `<action_start>`, `<action_end>`: アクションシーケンスの区切り
  - `<subtask>`, `</subtask>`: サブタスク予測の区切り
  - `<proprioception>`: 固有受容感覚入力の識別
- embed_tokens と lm_head のリサイズが必要

### 5.4 固有受容感覚の処理

pi0.5 方式: テキストトークンとして離散化
```
例: "joint_1: 0.32, joint_2: -0.15, gripper: 0.95, ..."
→ 通常のテキストトークンとして tokenize して入力
```

Qwen3.5 での実装:
- tokenizer でテキスト化 → embed_tokens で埋め込み → 通常のテキスト列として扱う
- 特別な実装変更は不要 (プロンプトエンジニアリングで対応)

---

## 6. 未解決の設計判断事項

### 6.1 Action Expert のサイズと構造
- [ ] hidden_size: 1024 (pi0.5比率) vs 2048 (半分) vs 実験で決定
- [ ] レイヤー数: 18 vs 32 (VLMと同じ) vs それ以外
- [ ] GatedDeltaNet を使うか full attention のみか
- [ ] VLM と重み共有するか完全独立か

### 6.2 VLM → Action Expert の接続方式
- [ ] 共有 Attention (pi0.5 準拠) vs Cross-Attention (分離)
- [ ] 共有の場合: VLM の全レイヤーか特定レイヤーのみか

### 6.3 Pre-training の要否
- [ ] Qwen3.5 は既に強力な VLM → Stage 1 をスキップして直接 Post-training から始められるか?
- [ ] FAST tokenizer による離散 pre-training の価値 vs 直接 flow matching

### 6.4 Qwen3.5 モデルサイズの選択
- [ ] デフォルト設定 (4096 hidden, 32層) は pi0.5 の VLM (2048, 18層) より大幅に大きい
- [ ] 小型バリアント (hidden=2048, layers=18 等) を使うか、そのまま大型モデルを使うか
- [ ] リアルタイム制御 (50Hz) の計算制約

### 6.5 マルチロボット対応
- [ ] アクション次元のパディング戦略 (最大 DoF に合わせてゼロパディング)
- [ ] 制御モードの切り替え (joint vs end-effector)

---

## 付録: 実装優先度マトリクス

| 優先度 | タスクID | タスク | 推定規模 |
|---|---|---|---|
| P0 (必須) | A-1 | Action Expert Transformer | 大 |
| P0 (必須) | A-2 | Adaptive RMSNorm | 小 |
| P0 (必須) | A-3 | Flow Matching モジュール | 中 |
| P0 (必須) | A-4 | アクション埋め込み/投影 | 小 |
| P0 (必須) | B-1 | Config 拡張 | 小 |
| P0 (必須) | B-2 | Attention Mask カスタマイズ | 中 |
| P0 (必須) | B-3 | Forward Pass 拡張 | 大 |
| P0 (必須) | B-4 | Hidden State 受け渡し | 中 |
| P0 (必須) | C-3 | 損失関数統合 | 小 |
| P1 (重要) | B-5 | 位置エンコーディング拡張 | 中 |
| P1 (重要) | B-6 | KV Cache 拡張 | 中 |
| P1 (重要) | C-1 | 2段階訓練ループ | 大 |
| P1 (重要) | C-2 | データローダー | 大 |
| P1 (重要) | D-1 | 階層的推論 | 中 |
| P1 (重要) | D-2 | リアルタイム推論最適化 | 中 |
| P2 (後回し可) | A-5 | FAST Action Tokenizer | 中 |
| P2 (後回し可) | 5.3 | Vocab 拡張 | 小 |

---

## 付録: ファイル変更マップ

```
/workspace/qwen/transformers-qwen3_5/src/transformers/models/qwen3_5/
├── configuration_qwen3_5.py   [B-1: VLAConfig追加]
├── modeling_qwen3_5.py        [A-1,A-2,A-4,B-2,B-3,B-4,B-5,B-6,C-3: 大幅改修]
├── modular_qwen3_5.py         [参照のみ / 必要に応じて更新]
├── tokenization_qwen3_5.py    [5.3: 特殊トークン追加]
├── __init__.py                [新クラスのexport追加]
├── action_expert.py           [A-1: 新規ファイル] (or modeling内に統合)
├── flow_matching.py           [A-3: 新規ファイル]
└── action_tokenizer.py        [A-5: 新規ファイル]
```
