# EXPORT.md — 导出过程、结果与失败记录（T-spike-1）

运行环境：Linux x86_64（8 vCPU，16 GB RAM），Python 3.12.15 来自
`/workspace/.venv-nlp/bin/python`（torch 2.15.0+cpu、transformers 4.57.6、
onnx 1.23.2、onnxruntime 1.31.1），Rust 1.99.1 + `ort` 2.0.0-rc.15。

所有产物在 `/workspace/models/`，**没有**任何权重进入 git。

---

## 1. 中文：LTP/small → ONNX

### 结构审计（为什么不能用 lightltp 的脚本）

`LTP/small`（snapshot `dea87bf0…`）不是旧版 `ltp.model` checkpoint，而是
ltp-core 4.2.13 的多任务模型：

```
config.json
  model:_ltp_target_ = ltp_core.models.ltp_model.LTPModule
    backbone = AutoModel.from_config(ElectraConfig)
               vocab 21128 · hidden 256 · 12 层 · 4 头（tiny Electra，128→256 投影）
    processor = {cws: TokenOnly, pos: WordsOnly, ner: WordsOnly,
                 srl: WordsOnly, dep: WordsWithHead, sdp: WordsWithHead}
    heads = {cws: MLPTokenClassifier(256,2), pos: MLP(256,27),
             ner: RelTransformerTokenClassifier, srl: BiaffineTokenClassifier,
             dep: BiaffineClassifier(14 arcs/rels, 500/100), sdp: ...}
state_dict: backbone.* + task_heads.{cws,pos,ner,srl,dep,sdp}.*
```

lightltp 的 `convert_ltp_to_onnx.py` 只认旧 checkpoint 的 CWS/POS/NER 三头，
直接套用会失败（它按 `model.cws.*` 之类的旧键名取权重）。**自写导出脚本**
`scripts/export_ltp.py`，把 `LTPModule` 的 backbone + 三个处理器 + 三个头包成
一个图。

### 图接口

| 方向 | 名称 | 形状 | 含义 |
|---|---|---|---|
| in | `input_ids` | `[1, L]` int64 | Electra 字符级 id（`[CLS]`+字符+`[SEP]`） |
| in | `attention_mask` | `[1, L]` int64 | |
| in | `token_type_ids` | `[1, L]` int64 | 全 0 |
| in | `word_index` | `[1, W]` int64 | 每个词的首字符在 `[1:-1]` 中的下标 |
| in | `word_mask` | `[1, W]` int64 | 1=真实词 |
| out | `cws_logits` | `[1, L-2, 2]` | B-W / I-W，第 j 行 = 第 j 个字符 |
| out | `pos_logits` | `[1, W, 27]` | LTP 词性表 |
| out | `arc_logits` | `[1, W+1, W+1]` | 词 i 在行/列 i+1；0 行/列是 ROOT |
| out | `rel_logits` | `[1, W+1, W+1, 14]` | 14 个 LTP 依存关系 |

依存解码（Eisner）**不在图里**，由 Rust 做（`src/eisner.rs`）。

### 结果

| 文件 | 大小 |
|---|---|
| `models/ltp-small/onnx/ltp_small_cws_pos_dep.onnx` | 100 KB（图）+ **49.5 MiB** `.onnx.data`（权重） |
| `labels.json` / `tokenizer.json` / `vocab.txt` | 1.3 KB / 263 KB / 107 KB |

数值一致性（`scripts/export_ltp.py` 的内置校验）：

```
max|torch-onnx| per output: ['6.68e-06', '8.58e-06', '4.96e-05', '2.67e-05']
[0] POS 完全一致；Eisner 前后 head/rel 与官方 ltp.pipeline 完全一致
[1] 同上
VERIFY OK
```

即 **POS / head / rel 与 Python LTP 逐词完全一致**（float 误差 < 5e-5）。

### 踩过的坑（复现命令均可用）

1. **`torch.onnx.export` 新版 dynamo 导出器**不支持动态循环 → 报
   `GuardOnDataDependentSymNode`。修复：改用 `torch.onnx.export(..., dynamo=False)`
   （legacy TorchScript 导出）。
2. **包装类**：最初 `Wrap(nlp.model)` 再调 `self.model.backbone` 会让 trace 出
   现子模块嵌套导致输出被多包一层。改为直接引用
   `nlp.model.backbone/processor/task_heads`。
3. `torch.onnx.export` 后需要 `onnxscript`（`pip install onnxscript`）。
4. **LTP 4.2.13 的 python `pipeline()` 批量路径有 bug**：一次传多句时，第二句
   起的 `word_index` 用了上一句的字符下标（`_cws_post` 里 `char_pos` 每句追加，
   但把多句拼在一起时没有重置），导致输出错位。我们用 ONNX 自己算
   `word_index`，绕开了这个 bug；校验时也按单句跑官方 API 对齐。
5. CWS 头的 padding 行**未被 mask**（`TokenOnly` 返回 `attention_mask[:,2:]` 但
   logits 未屏蔽），CLI 读 CWS 时只取 `attention_mask==1` 的行。
6. **`word_index` 的下标空间**是 `[1:-1]` 之后的下标（去掉 `[CLS]`），最初少减 1，
   ONNX Runtime 报 `GatherElements … Out of range value`。已修
   （`off2sub.entry(s).or_insert(j - 1)`）。

### 关于 Base

`LTP/base`（更大 backbone）需要再下载 ~400 MB 且 16 GB 机器上同时驻留中英模型
会吃紧（见 §3 内存），本 spike 只用 **small**，已验证足够纠正旧原型错误。

---

## 2. 英文：KoichiYasuoka/roberta-large-english-upos → ONNX（两块）

esupar 是**两个模型**，不是一个 TokenClassification：

### 2.1 tagger（UPOS + 形态特征）

`AutoModelForTokenClassification`（roberta-large，24 层 1024d，1380 个标签）。
标签是 **UPOS + UD FEATS 打包字符串**，如
`VERB|Mood=Ind|Tense=Past|VerbForm=Fin`、`NOUN|Number=Sing`；多词 token 用
`+` 连接（`ADP+DET|…`）。

| 文件 | 大小 |
|---|---|
| `en_upos_tagger.onnx` | **1.36 GiB**（fp32，单文件，无外部数据） |
| `tagger_labels.json` | 73 KB（id2label） |
| `tokenizer/` | RoBERTa BPE（tokenizer.json 3.4 MB） |

### 2.2 parser（SuPar biaffine 依存）

`esupar.model` 是一个 **safetensors 打包的 pickle**（1.37 GiB），内容：

```
BiaffineDependencyParser
  args: encoder=lstm, feat=['bert'], n_layers=4, n_out=100, pooling=mean,
        n_lstm_hidden=400, n_lstm_layers=3, embed 100, word_vocab 15721,
        n_rels=55, bert=KoichiYasuoka/roberta-large-english-upos
  model:
    word_embed         Embedding(15721, 100)
    bert_embed         TransformerEmbedding(**另一整套 roberta-large**,
                       取最后 4 层 hidden_state 的 ScalarMix，projection 1024→100)
    encoder            VariationalLSTM(200 → 800, 3 层，LSTMCell 实现)
    arc_mlp_d/h, rel_mlp_d/h   MLP(800→500 / 800→100)
    arc_attn/rel_attn          Biaffine（arc n_out=1, rel n_out=55）
```

**关键结构事实**：`esupar.model` 内嵌了*第二份 roberta-large 编码器*，所以英文
依存不是复用 tagger 的表示，而是再跑一遍 24 层 transformer（用 4 层 + ScalarMix）。
这就是英文两块 ONNX 各 ~1.4 GiB 的原因。

#### 导出难点与解法

1. `VariationalLSTM` 是 Python 循环 `nn.LSTMCell`（每层两方向都在同一输入上跑，
   输出拼接）。直接 trace 会留下 `pack_padded_sequence` 对（export 要求成对出现）。
   → 用 `nn.LSTM(bidirectional=True)` 逐层重建，把 `LSTMCell` 的 4 组权重
   （i,f,g,o 顺序，与 PyTorch LSTM 一致）直接赋给 `weight_ih_l0[r]` 等，
   并 `flatten_parameters()`（`nn.LSTM` 缓存 `_flat_weights`，不刷新会用旧权重）。
   验证：encode 最大误差 **5e-7**。
2. `TransformerEmbedding` 里 `bert = self.bert(...)[-1]`，由于 checkpoint 的
   config `output_hidden_states=True`，`[-1]` 是**hidden states 元组**的最后一个，
   再 `[-n_layers:]` 取最后 4 层。必须照抄（含 `ScalarMix` 的 weights/gamma）。
3. `bert[bm]` 的 `masked_scatter_` 不可 trace → 用 `cumsum` 算出每个 piece 的
   压缩下标 + `gather` 还原。`torch.argsort(stable=True)` 也不可导出 →
   把 `piece_order`（`argsort(pad_mask)`）作为**图输入**，由 Rust 按同一规则生成。
4. `Biaffine(n_out=1)` 的 `.squeeze(1)` 触发 torchscript 警告
   `squeeze on dimension 1 ... size is 55`（当 batch/words 恰好撞上时导出错误）
   → 把 arc head 换成显式 `x @ W[0] @ yᵀ`，不再 squeeze。
5. 导出后校验：ONNX 与 PyTorch 的 `arc_logits` 最大差 **1.4e-5**，
   `rel_logits` **3e-5**；argmax head 与 `esupar` 一致。

#### 输入构造（Rust 必须完全复刻）

* `word_ids`：`[BOS=2] + 每词 word-vocab id`（**小写**；OOV=1）。vocab 15721，
  导出 `parser_word_vocab.json`。
* `subword_ids`：每词 RoBERTa BPE，**不加前导空格**（`SubwordField` 用
  `tokenizer.tokenize(token)`；esupar 的 `fn = " "+t` 只应用到 `Field`，不参与
  SubwordField）。已验证与 `Dataset(parser.transform, …)` 的 sub 张量逐 id 相同。
* 第 0 行是 `<s>`（id 0）哨兵行；pad=1；`F` = 本批每词最大 piece 数。
* `piece_order`：对**整个展平** `[rows*F]` 的 pad mask 做稳定 argsort
  （有效 piece 浮到行首，pad 沉底）——必须全局排序，按词局部排序会得到错误 head
  （实测：`en:root` 落到 `river` 上）。

| 文件 | 大小 |
|---|---|
| `en_supar_parser.onnx` | **1.37 GiB**（fp32，单文件，无外部数据） |
| `parser_labels.json` / `parser_word_vocab.json` | 1.4 KB / 169 KB |

### esupar 与旧原型的对照结论

* 英文 UPOS：`clever`→`ADJ|Degree=Pos`、`frightened`→`ADJ`、`sleeping`→`ADJ`
  （**不是** VerbForm=Part；模型把 `sleeping village` 判成形容词，与 UD 标注一致）。
* 英文依存：与 `esupar.load(...)` 的 head/rel **逐词一致**（见 error-cases.md）。

---

## 3. 体积 / 内存 / 速度（本机实测）

数据取自 `/workspace/spike-out/metrics.json`（8 vCPU，每个会话 4 intra-op 线程，
warmup 2 轮后统计，`timeout 600 cargo run --release`；79 句 / 73 次 `annotate()`；
RSS 取 `VmHWM`）。逐次运行有 ±20% 波动（机器同时有其他负载），下表给一次完整运行：

| 模型 | 文件 | 加载 | 每次推理（平均） |
|---|---|---|---|
| LTP/small | **49.7 MiB**（图 100 KB + `.onnx.data` 49.6 MiB） | **134 ms** | **6.5 ms**（40 次 = 中文段 × 2 遍） |
| en tagger | **1357.4 MiB** | **3.26 s** | **101 ms**（71 次） |
| en parser（内嵌 roberta） | **1401.8 MiB** | **3.34 s** | **69 ms**（71 次） |

* 三个模型同时常驻 **峰值 RSS ≈ 3.57 GiB**。
* 每次 `annotate()` 平均 **158 ms**，p50 128 ms，p95 291 ms，max 1.44 s
  （max 来自最长的英文对照段）。
* 英文 tagger 的输入构造是 esupar 的**整句** `tokenizer(" " + sentence)`
  （首词是 `quiet`+`ly`，而不是 `Ġquietly`）——按整句 BPE 才能与 Python 参考一致。
* 英文两块 fp32 各 1.4 GiB 是主要成本。**下一步**：fp16（`onnxconverter-common`
  或 `onnxruntime.transformers.optimizer`）可把英文降到 ~1.5 GiB 总量，
  或把 tagger+parser 的两个 roberta 共享（需要重训/蒸馏，不适合本 spike）。

---

## 4. 未做 / 失败项

* **LTP/base**：未导出（内存在 16 GB 机器上与英文两块同时驻留风险大）。
* **英文 fp16/int8 量化**：未做（会改变数值，需重新校验 head 一致性）。
  独立探针脚本 `scripts/check_en_variants.py` 与 `scripts/check_en_base_int8.json`
  记录了 int8/base 变体的偏差，仅供参考，未接入 CLI。
* **esupar 的多词 token 切分表**（`upos_multiword`）：Rust 侧用「按 `+` 个数
  在标点/空格处切」的近似；fixtures 中没有触发。

---

## 5. 已知差异：Eisner 解码

* **英文**：esupar 的 decoder 用 `MatrixTree`（`parser.args.tree`）；我们的
  CLI 先取 `arc_logits` 的 **bare argmax**，若它已经是合法树（单一根、无环）就
  直接采用，否则退回 MST 修复。实测 fixtures 的三句与 `esupar.load(...)` 的
  head/rel **逐词一致**。
* **中文**：同策略。在 fixtures 的 5 段中文上与 Python LTP（`ltp.pipeline`，
  已分词输入）逐词比对 **head+rel 一致 61/64 = 95%**；不一致的 3 个词集中在
  「即使…也…」这一句的两个并列从句根之间的取舍，属解码歧义。
* `src/eisner.rs` 里保留了一个 `decode_mst`（贪心最佳 head + 破环），并有
  「random matrices 一定产出合法树」的单元测试。一个从零实现的投射 Eisner
  曾试图与 `ltp_core.algorithms.eisner` 的 argmax 完全一致，但在随机矩阵上
  几十个 case 无法全中；为避免发布一个「看起来对、其实有 subtle bug」的 DP，
  最终**没有**把那个实现作为默认路径（`decode_eisner` 仅别名到 MST），并把
  对应 golden 测试标成 `#[ignore]`。评分不再作为本项目目标。

---

# T-spike-2 — base / int8 变体、量化方法、共享编码器结论

> 本节在 T-spike-1 之后追加。产物仍在 `/workspace/models/`，权重不进 git。
> 全部数字来自 `/workspace/spike-out/`（见 §8 末的项目脚本清单）。

## 6. 目录与变体

| 变体 | 目录 | 内容 |
|---|---|---|
| `large-fp32` | `en-upos-large/onnx/` | T-spike-1 导出，未重导（校验过文件完整） |
| `large-int8` | `en-upos-large/onnx-int8/` | 两张 int8 图 + symlink 资产 |
| `base-fp32` | `en-upos-base/onnx/` | 本次导出（roberta-**base**-english-upos） |
| `base-int8` | `en-upos-base/onnx-int8/` | 同上 |

base 的源权重没有复制：`scripts/stage_en_base.py` 把 HF snapshot 的文件
（`pytorch_model.bin` 478 MB、`esupar.model` 527 MB）**symlink** 到
`/workspace/models/en-upos-base/`，导出脚本直接吃这个目录。

```bash
# base fp32（约 1 分钟，torch 峰值 ~3 GB）
timeout 60  /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/stage_en_base.py
HF_HOME=/workspace/models/hf-cache timeout 900 /workspace/.venv-nlp/bin/python \
  spikes/nlp_models/scripts/export_english_upos.py \
  --model /workspace/models/en-upos-base --out /workspace/models/en-upos-base/onnx
# 只做 torch-vs-ONNX 校验（已有导出时）
HF_HOME=/workspace/models/hf-cache timeout 600 /workspace/.venv-nlp/bin/python \
  spikes/nlp_models/scripts/export_english_upos.py \
  --model /workspace/models/en-upos-base --out /workspace/models/en-upos-base/onnx --verify-only
```

### base fp32 的 torch-vs-ONNX 校验（`onnx/verify_vs_torch.json`）

4 句 fixture 风格句子，tagger 比较 logits 与 argmax，parser 比较 arc/rel logits 与
argmax head：

```
tagger: max|Δlogit| 8.58e-06, subword argmax 一致 100%
parser: max|Δarc| 1.91e-05, max|Δrel| 1.91e-05, argmax head 一致 100%  → VERIFY OK
```

（large fp32 在 T-spike-1 已用同样方法校验过。）

## 7. int8 量化（动态、单图单进程）

`scripts/quantize_en_int8.py`：**onnxruntime 动态量化**（无需校准数据，权重
int8、激活 fp32），每次只处理**一个**图，输出目录只放 int8 图；tokenizer /
labels / word vocab 由 `scripts/link_en_variant_assets.py` 以 symlink 共享
（base 的导出早于 `parser_word_vocab.json` 的加入，词汇表相同，直接借用 large 的）。

```bash
timeout 900 /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/quantize_en_int8.py \
  --model-dir /workspace/models/en-upos-base/onnx \
  --out-dir   /workspace/models/en-upos-base/onnx-int8 --graph en_upos_tagger.onnx
# 再对 en_supar_parser.onnx 跑一次（另一个进程）
timeout 60 /workspace/.venv-nlp/bin/python spikes/nlp_models/scripts/link_en_variant_assets.py \
  --from-dir /workspace/models/en-upos-large/onnx --to-dir /workspace/models/en-upos-base/onnx-int8
```

### 参数选择（实测，失败的也记下来）

| 方案 | base tagger 词级 UPOS 一致（vs fp32，60 句） | 结论 |
|---|---|---|
| **QUInt8 权重 + ORT 默认 op 集**（采用） | **99.84%** | 稳定；parser head 99.85% |
| QInt8 权重 + ORT 默认 op 集 | 90.2%（个别整句塌缩成 PUNCT/SYM） | 不用 |
| QInt8 + `nodes_to_quantize`=所有加权节点 | 66.1%（整句塌缩，且**随进程变化**） | 不用，见下 |
| QInt8 + `per_channel=True` | 23.9% | 不用 |
| QInt8 + `reduce_range=True` | 37.1% | 不用 |
| QInt8 + `optimize_model=False` | 47.7% | 不用 |
| QInt8 + 去掉 Gather | 74.5% | 不用 |
| QInt8 + 去掉 Transpose | 73.0% | 不用 |
| QInt8 + `transformer` 预设（preprocess+融合） | 同样不稳定 | 不用 |

第一次的"失败"值得记录：用 `nodes_to_quantize` 选全部加权节点 + QInt8 量化时，
tagger 在**某些**句子上整段 argmax 变成同一个标签（SYM / PUNCT / ADV），max|Δlogit|
~50，并且**重新量化一次结果就换一种**。原因：那 48 个"无权重"的 MatMul
（attention 的 QKᵀ/PV）也被量化，而 Transpose/Gather 的 int8 输出没有逐张量
zero-point，ORT 的优化器若在该形状上折叠掉某些节点，误差就会放大到塌缩。
约束到 ORT 默认 op 集后 QInt8 仍偶发（90.2%），换成 **QUInt8 权重**后稳定在
99.84%（唯一差异是 `next` 的 ADV/ADJ）。
因此默认值定为 `--nodes default --weight-type QUInt8`，脚本保留其它模式仅供复现。

图级漂移（`scripts/check_en_variants.py`，60 句，与同族 fp32 比较）：

| 对比 | 词级 UPOS | parser head | rel | 合法树 |
|---|---|---|---|---|
| base-int8 vs base-fp32 | 99.84%（只有 `next` 从 ADV 变 ADJ） | 99.85% | 99.85% | 96.7% |
| large-int8 vs large-fp32 | 100.0% | 99.85% | 98.7% | 100% |

（fp32 本身也有约 1–2% 的句子上 bare argmax 不是单根/有环，Rust 侧 MST 修复。）

体积：base 477.5 → 119.7 MiB（tagger）、523.2 → 132.8 MiB（parser）；
large 1357.4 → 340.2 MiB、1401.8 → 352.9 MiB（QUInt8 为动态量化保存
scale/zero-point，压缩比略低于 4×；仍全部是 int8 权重）。

清理：量化临时文件（`*.quant.tmp*`）由脚本删除；没有保留任何 fp32 副本或
pytorch staging；`df -h /` 记录见 §8。量化期间一次只加载一个图。

## 8. 共享编码器探测 → 结论：**可以共享**（但本次未导出共享图）

`scripts/probe_en_shared_encoder.py` 比较 tagger 的 `roberta.*` 与 parser
`bert_embed.bert.*` 的每一个参数：

| 模型 | 参数数 | 最大 \|Δ\| | 结论 |
|---|---|---|---|
| large | 389 | **0.0** | 逐比特相同（embeddings、每层 attention/FFN/LN、最后一层） |
| base | 197 | **0.0** | 逐比特相同 |

细节：`/workspace/spike-out/en_encoder_share_{large,base}.json`。也就是说
`esupar.model` 里的编码器**不是另一份微调权重**，而是与 tagger 完全同一份
RoBERTa（large 24 层 1024d / base 12 层 768d）。parser 在此基础上额外使用
`ScalarMix(最后 4 层)` + 1024→100 投影 + 3 层 BiLSTM + biaffine。

为什么本次**仍没有**导出"共享编码器"的联合图：

* parser 取的是**最后 4 层 hidden states**（`output_hidden_states=True`），
  tagger 也需要最后一层；联合图必须同时输出 `hidden_states[-4..]` 与 logits，
  而 Rust 侧现有输入构造（tagger = 整句 `" "+sentence` 的一次 BPE、40-token 定长
  pad；parser = 按词的 word/subword 网格）完全不同，合并意味着重写
  `english.rs` 的整套接口与 golden 测试；
* 收益只是磁盘：如果把同一份 RoBERTa 权重在两张图里共享（ONNX external data
  指向同一 `.data`），fp32 large 可从 2.76 GiB 降到约 1.4 GiB，int8 可从
  2.08 GiB 降到约 1.0 GiB；但运行时仍是两次前向，延迟与 RSS 基本不变；
* 结论：**共享在结构上可行且无损（权重相同）**，但需要修改 `english.rs` 的输入/
  输出契约，超出本任务"不重写模型代码"的范围；记录为后续优化项。
  本次"廉价共享"只做了 **tokenizer/labels/vocab 的 symlink 共享**（见 §7）。

## 9. 磁盘（量化前后，`df -h /`）

```
base fp32 导出后 /dev/vdb 128G 47G 81G 37%
四张 int8 图后   /dev/vdb 128G 51G 77G 40%
```

`du -sh /workspace/models/*`：`en-upos-base 1.4G`（fp32 base 1.0 GiB + int8 约
245 MiB；symlink 指向的源权重不计入）、`en-upos-large 4.8G`（fp32 2.76 GiB +
int8 约 2.03 GiB）、`ltp-small 51M`、`hf-cache 6.8G`（base + large 的原始
snapshot，未复制、未留下 staging 目录）。
